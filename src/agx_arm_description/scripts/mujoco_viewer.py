#!/usr/bin/env python3
"""
mujoco_viewer.py — Puente ROS2 <-> MuJoCo para el brazo Nero
==============================================================
Simula el brazo con la misma interfaz ROS2 que agx_arm_ctrl (brazo real), para que
el mismo codigo/launch sirva en real y en simulacion.

Comandos (suscripciones):
  control/joint_states  JointState   posiciones directas (streaming, sin planificar)
  control/move_j        JointState   objetivo articular (arm + 'gripper'); admite flujos a 100 Hz
  control/move_p        PoseStamped  punto a punto: IK + interpolacion articular
  control/move_l        PoseStamped  linea recta cartesiana
  control/move_c        PoseArray    arco circular inicio -> medio -> fin

Feedback (publicaciones):
  feedback/joint_states JointState   joint1..7, 'gripper' (ancho total, m), y despues los
                                     joints de MuJoCo (gripper_joint1/2, por dedo)
  feedback/tcp_pose     PoseStamped  pose de link7 respecto a base_link
  sim/cube_state        Float64MultiArray  estado real del cubo [x,y,z,vx,vy,vz] (si la escena tiene 'cube')
  distance              Range        sensor ultrasonico simulado (solo con --sensor-pose)

Convenciones:
  - Poses en el marco de base_link, cuaternion ROS (x, y, z, w). Por defecto la pose
    es la de link7; el marco de la brida del SDK real puede estar girado/desplazado
    respecto a link7: ver --flange-offset y --flange-zero-identity (o las variables
    NERO_FLANGE_OFFSET='x y z qx qy qz qw' y NERO_FLANGE_ZERO_IDENTITY=1).
    tcp_offset (herramienta) NO esta soportado en simulacion.
  - 'gripper' = ANCHO total en metros (0.0 a 0.1); el actuador de MuJoCo mueve un
    dedo (0 a 0.05) -> dedo = ancho / 2.
  - Un movimiento sin solucion de IK se rechaza con un aviso (como el brazo real).
  - Gravedad compensada por defecto (el brazo real lo hace); --no-gravcomp la desactiva.

Uso (lo invoca display_mujoco.launch.py):
    python3 mujoco_viewer.py <escena.xml> [--rate 200] [--joint-speed 0.8] [--lin-speed 0.1]
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray

ARM_JOINTS = [f'joint{i}' for i in range(1, 8)]
FLANGE_BODY = 'link7'
BASE_BODY = 'base_link'


# --------------------------------------------------------------------------------------
# Utilidades de rotaciones (cuaterniones en orden MuJoCo: w, x, y, z)
# --------------------------------------------------------------------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_error(q_target, q_current):
    """Vector de rotacion (marco mundo) que lleva q_current a q_target."""
    q_err = quat_mul(q_target, quat_conj(q_current))
    if q_err[0] < 0.0:
        q_err = -q_err
    v = q_err[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return 2.0 * v
    return v / s * 2.0 * math.atan2(s, q_err[0])


def quat_slerp(q0, q1, t):
    t = min(max(t, 0.0), 1.0)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def arc_waypoints(p0, pm, p1, n):
    """Puntos de un arco circular que pasa por p0 -> pm -> p1.

    Devuelve (puntos (n+1, 3), fraccion del arco donde esta pm, longitud del arco).
    Si los tres puntos son colineales, devuelve una recta.
    """
    b = pm - p0
    c = p1 - p0
    bxc = np.cross(b, c)
    den = 2.0 * float(np.dot(bxc, bxc))
    if den < 1e-12:
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        d01 = float(np.dot(c, c))
        fm = float(np.clip(np.dot(b, c) / max(d01, 1e-12), 0.0, 1.0))
        return p0 + c * t, fm, math.sqrt(d01)

    center = p0 + np.cross(np.dot(b, b) * c - np.dot(c, c) * b, bxc) / den
    r = np.linalg.norm(p0 - center)
    u = (p0 - center) / r
    nrm = bxc / np.linalg.norm(bxc)  # p0 -> pm -> p1 es antihorario respecto a nrm
    v = np.cross(nrm, u)

    def angle_of(p):
        d = p - center
        return math.atan2(float(np.dot(d, v)), float(np.dot(d, u))) % (2.0 * math.pi)

    th1 = angle_of(p1)
    thm = angle_of(pm)
    th = np.linspace(0.0, th1, n + 1)
    pts = center + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    return pts, thm / th1, r * th1


# --------------------------------------------------------------------------------------
# Nodo puente
# --------------------------------------------------------------------------------------
class MujocoBridge(Node):
    def __init__(self, scene_path: str, control_topic: str, feedback_topic: str,
                 rate_hz: float, move_j_topic: str = 'control/move_j',
                 joint_speed: float = 0.8, lin_speed: float = 0.1, gravcomp: bool = True,
                 flange_offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                 flange_zero_identity: bool = False,
                 cube_velocity=(0.0, 0.05, 0.0), sensor_pose=None,
                 sensor_rate: float = 20.0, sensor_noise: float = 0.0,
                 sensor_max: float = 4.0, cube_delay: float = 0.0,
                 cube_source: str = 'velocity', sensor_topic: str = 'distance',
                 cube_track_tau: float = 0.1, sensor_timeout: float = 0.5):
        super().__init__('mujoco_bridge')

        self.joint_speed = joint_speed  # rad/s (media, norma sobre los 7 joints)
        self.lin_speed = lin_speed      # m/s (media)

        self.get_logger().info(f'Cargando escena MuJoCo: {scene_path}')
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        if gravcomp and hasattr(self.model, 'body_gravcomp'):
            # El brazo real compensa su propio peso. Con los kp bajos de los actuadores de
            # MuJoCo el brazo se hundia ~5 cm por debajo de lo ordenado; anular la gravedad
            # en todos los cuerpos reproduce el comportamiento del real.
            self.model.opt.gravity[:] = 0.0  # body_gravcomp asignado tras compilar no actua (ngravcomp=0)
            self.get_logger().info('Compensacion de gravedad activada (--no-gravcomp la desactiva).')
        self.data = mujoco.MjData(self.model)

        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self._actuator_ids = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_ids[name] = i

        self._joint_info = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not name:
                continue
            self._joint_info[name] = {
                'id': i,
                'qpos_adr': self.model.jnt_qposadr[i],
                'qvel_adr': self.model.jnt_dofadr[i],
            }

        self.get_logger().info(f'Actuadores encontrados: {list(self._actuator_ids.keys())}')
        self.get_logger().info(f'Joints encontrados: {list(self._joint_info.keys())}')

        # Gripper: 'gripper' (ancho total, m) -> dedo = ancho / 2, limitado al rango del joint.
        self._gripper_max = None
        g_id = self._actuator_ids.get('gripper')
        if g_id is not None:
            g_joint = int(self.model.actuator_trnid[g_id, 0])
            self._gripper_max = float(self.model.jnt_range[g_joint, 1])

        # Indices del brazo (joint1..joint7).
        missing = [n for n in ARM_JOINTS if n not in self._actuator_ids or n not in self._joint_info]
        if missing:
            raise RuntimeError(f'Faltan joints/actuadores del brazo en el modelo: {missing}')
        self._arm_aid = np.array([self._actuator_ids[n] for n in ARM_JOINTS])
        self._arm_qadr = np.array([self._joint_info[n]['qpos_adr'] for n in ARM_JOINTS])
        self._arm_dadr = np.array([self._joint_info[n]['qvel_adr'] for n in ARM_JOINTS])
        lo, hi = [], []
        for n in ARM_JOINTS:
            jid = self._joint_info[n]['id']
            if self.model.jnt_limited[jid]:
                lo.append(float(self.model.jnt_range[jid, 0]))
                hi.append(float(self.model.jnt_range[jid, 1]))
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        self._q_lo, self._q_hi = np.array(lo), np.array(hi)

        # Marcos: efector (link7) y base (base_link).
        self._flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)
        if self._flange_id < 0:
            raise RuntimeError(f'No existe el cuerpo "{FLANGE_BODY}" en el modelo')
        mujoco.mj_forward(self.model, self.data)
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        if base_id >= 0:
            self._base_p = self.data.xpos[base_id].copy()
            self._base_R = self.data.xmat[base_id].reshape(3, 3).copy()
        else:
            self._base_p = np.zeros(3)
            self._base_R = np.eye(3)
        self._base_q = mat_to_quat(self._base_R)

        # Datos auxiliares solo para cinematica/IK (no tocan la simulacion).
        self._ik_data = mujoco.MjData(self.model)

        # Offset link7 -> brida del SDK, expresado en el marco de link7:
        #   p_brida = p_link7 + R_link7 @ t_off ;  R_brida = R_link7 @ R_off
        self._t_off = np.array(flange_offset[:3], dtype=float)
        q_off = np.array([flange_offset[6], flange_offset[3], flange_offset[4], flange_offset[5]])
        self._q_off = q_off / np.linalg.norm(q_off)
        if flange_zero_identity:
            # El SDK mide la orientacion relativa a la pose en cero: con q=0 reporta identidad.
            _, q_zero = self._fk_base(np.zeros(7))
            self._q_off = quat_conj(q_zero)
            self.get_logger().info(
                'Marco de brida: orientacion relativa a la pose en cero (identidad con q=0).')

        # Mantener la pose inicial.
        self.data.ctrl[self._arm_aid] = self.data.qpos[self._arm_qadr]

        self._traj = None
        self._jgoal = None   # objetivo articular de move_j (seguidor con limite de velocidad)
        self._data_lock = threading.Lock()

        self.control_sub = self.create_subscription(
            JointState, control_topic, self._on_control_msg, 10)
        self.move_j_sub = self.create_subscription(
            JointState, move_j_topic, self._on_move_j, 10)
        self.move_p_sub = self.create_subscription(
            PoseStamped, 'control/move_p', self._on_move_p, 10)
        self.move_l_sub = self.create_subscription(
            PoseStamped, 'control/move_l', self._on_move_l, 10)
        self.move_c_sub = self.create_subscription(
            PoseArray, 'control/move_c', self._on_move_c, 10)

        self.feedback_pub = self.create_publisher(JointState, feedback_topic, 10)
        self.tcp_pub = self.create_publisher(PoseStamped, 'feedback/tcp_pose', 10)
        self.create_timer(1.0 / rate_hz, self._publish_feedback)

        self._setup_cube(cube_velocity, sensor_pose, sensor_rate, sensor_noise, sensor_max,
                         cube_delay, cube_source, sensor_topic, cube_track_tau,
                         sensor_timeout)

        self.get_logger().info(
            f'Comandos: "{control_topic}" (directo), "{move_j_topic}", control/move_p, '
            f'control/move_l, control/move_c. Feedback: "{feedback_topic}" y '
            f'feedback/tcp_pose a {rate_hz} Hz.')

    # ------------------------------------------------------------------ pinza
    def _set_gripper_locked(self, width):
        g_id = self._actuator_ids.get('gripper')
        if g_id is None or self._gripper_max is None:
            return
        self.data.ctrl[g_id] = min(max(width / 2.0, 0.0), self._gripper_max)

    # --------------------------------------------------------- comandos directos
    def _on_control_msg(self, msg: JointState):
        """control/joint_states: posicion directa a los actuadores (sin planificar)."""
        with self._data_lock:
            for name, pos in zip(msg.name, msg.position):
                if name in ARM_JOINTS:
                    self._traj = None  # el streaming directo manda sobre una trayectoria
                    self._jgoal = None
                if name == 'gripper':
                    self._set_gripper_locked(pos)
                    continue
                actuator_id = self._actuator_ids.get(name)
                if actuator_id is None:
                    continue
                self.data.ctrl[actuator_id] = pos

    # ----------------------------------------------------------- trayectorias
    def _q_now(self):
        with self._data_lock:
            return self.data.qpos[self._arm_qadr].copy()

    def _start_trajectory(self, Q, path_len=0.0):
        """Programa una trayectoria articular suave por los waypoints Q (N x 7)."""
        Q = np.asarray(Q, dtype=float)
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-6:
            self.get_logger().info('Ya esta en la posicion objetivo.')
            return 0.0
        T = max(0.5, total / self.joint_speed, path_len / self.lin_speed)
        with self._data_lock:
            self._traj = {'Q': Q, 'cum': cum, 'T': T, 't0': None}
            self._jgoal = None
        return T

    def _advance_trajectory(self):
        """Se llama desde step() con el lock tomado."""
        tr = self._traj
        if tr is None:
            self._follow_joint_goal()
            return
        if tr['t0'] is None:
            tr['t0'] = self.data.time
        s = (self.data.time - tr['t0']) / tr['T']
        done = s >= 1.0
        s = min(max(s, 0.0), 1.0)
        s = s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)  # perfil minimum-jerk
        cum, Q = tr['cum'], tr['Q']
        u = s * cum[-1]
        i = int(np.clip(np.searchsorted(cum, u, side='right') - 1, 0, len(cum) - 2))
        seg = cum[i + 1] - cum[i]
        a = 0.0 if seg < 1e-12 else (u - cum[i]) / seg
        self.data.ctrl[self._arm_aid] = (1.0 - a) * Q[i] + a * Q[i + 1]
        if done:
            self._traj = None

    def _follow_joint_goal(self):
        """Seguidor de objetivo articular para move_j (con el lock tomado).

        Acerca el comando hacia el objetivo con una exponencial suave limitada a
        joint_speed. Como no reinicia nada con cada mensaje, tolera flujos de move_j a
        100 Hz (p. ej. los de ik_node) y llega al ultimo objetivo recibido.
        """
        if self._jgoal is None:
            return
        dt = float(self.model.opt.timestep)
        cur = self.data.ctrl[self._arm_aid].copy()
        err = self._jgoal - cur
        n = float(np.linalg.norm(err))
        if n < 1e-6:
            self.data.ctrl[self._arm_aid] = self._jgoal
            self._jgoal = None
            return
        step = min(n * (1.0 - math.exp(-dt / 0.1)), self.joint_speed * dt)
        self.data.ctrl[self._arm_aid] = cur + err / n * step

    # --------------------------------------------------------------- cinematica
    def _fk_base(self, q):
        """Pose (p, quat wxyz) de link7 en base_link para los joints q."""
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self._arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        p_w = d.xpos[self._flange_id]
        R_w = d.xmat[self._flange_id].reshape(3, 3)
        return self._base_R.T @ (p_w - self._base_p), mat_to_quat(self._base_R.T @ R_w)

    def _solve_ik(self, p_b, q_b, q_seed, max_iter=400, tol_p=2e-3, tol_r=2e-2):
        """IK por Jacobiano (minimos cuadrados amortiguados). Objetivo en marco base."""
        m, d = self.model, self._ik_data
        p_t = self._base_p + self._base_R @ p_b
        q_t = quat_mul(self._base_q, q_b)
        d.qpos[:] = self.data.qpos
        q = np.clip(np.asarray(q_seed, dtype=float), self._q_lo, self._q_hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        def errors():
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = p_t - d.xpos[self._flange_id]
            e_r = quat_error(q_t, mat_to_quat(d.xmat[self._flange_id].reshape(3, 3)))
            return e_p, e_r

        for _ in range(max_iter):
            e_p, e_r = errors()
            if np.linalg.norm(e_p) < tol_p and np.linalg.norm(e_r) < tol_r:
                return q, True, (float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r)))
            mujoco.mj_jacBody(m, d, jacp, jacr, self._flange_id)
            J = np.vstack([jacp[:, self._arm_dadr], jacr[:, self._arm_dadr]])
            e = np.concatenate([e_p, e_r])
            # Amortiguamiento adaptativo: alto si el error es grande (estabilidad cerca de
            # singularidades), bajo si ya estamos cerca (convergencia rapida).
            lam2 = 1e-4 + 0.0025 * min(1.0, float(np.linalg.norm(e)) / 0.05)
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self._q_lo, self._q_hi)

        e_p, e_r = errors()
        ok = np.linalg.norm(e_p) < tol_p and np.linalg.norm(e_r) < tol_r
        return q, bool(ok), (float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r)))

    def _solve_path(self, poses, q_seed):
        """IK secuencial (cada waypoint parte de la solucion anterior).

        Devuelve (soluciones, None) o (None, (indice, n, err_pos, err_rot)) si falla.
        """
        sols = []
        q = q_seed
        for idx, (p, qt) in enumerate(poses):
            q, ok, (ep, er) = self._solve_ik(p, qt, q)
            if not ok:
                return None, (idx, len(poses), ep, er, q)
            sols.append(q)
        return sols, None

    def _plan_path(self, poses, q_now, max_jump=np.inf, n_seeds=30, max_branches=6):
        """Resuelve la IK de toda la trayectoria, probando otras configuraciones si hace falta.

        1) Desde la configuracion actual (lo normal: movimiento continuo).
        2) Si falla, busca soluciones alternativas del primer punto desde semillas aleatorias
           (otras 'ramas' del brazo, como hace el firmware) y prueba la trayectoria completa
           desde las mas cercanas a la configuracion actual. max_jump limita (rad) cuanto
           puede alejarse la rama elegida de la configuracion actual.
        """
        sols, fail = self._solve_path(poses, q_now)
        if sols is not None:
            return sols

        rng = np.random.default_rng(0)
        lo = np.maximum(self._q_lo, -math.pi)
        hi = np.minimum(self._q_hi, math.pi)
        p0, qt0 = poses[0]
        cands = []
        for _ in range(n_seeds):
            q, ok, _ = self._solve_ik(p0, qt0, rng.uniform(lo, hi))
            if ok and np.linalg.norm(q - q_now) <= max_jump and \
                    all(np.linalg.norm(q - c) > 0.3 for c in cands):
                cands.append(q)
        cands.sort(key=lambda c: float(np.linalg.norm(c - q_now)))

        tried = 0
        for c in cands[:max_branches]:
            tried += 1
            rest, _ = self._solve_path(poses[1:], c) if len(poses) > 1 else ([], None)
            if rest is not None:
                self.get_logger().info(
                    f'IK: se uso una configuracion alternativa del brazo (rama {tried}).')
                return [c] + rest

        idx, n, ep, er, qf = fail
        self.get_logger().warn(
            f'Sin solucion IK en el punto {idx + 1}/{n} (error pos {ep * 1000:.1f} mm, '
            f'rot {er:.3f} rad; {tried} configuraciones alternativas probadas): '
            f'movimiento rechazado.')
        en_limite = [f'joint{k + 1}={qf[k]:.3f}' for k in range(len(qf))
                     if min(qf[k] - self._q_lo[k], self._q_hi[k] - qf[k]) < 0.02]
        self.get_logger().warn(f'IK fallo con joints en el limite (<0.02 rad): {en_limite or "ninguno"}; '
                               f'q = {np.round(qf, 3).tolist()}')
        return None

    def _link7_to_flange(self, p, q):
        """Pose de link7 -> pose de la brida del SDK (marco base, q = wxyz)."""
        return p + quat_to_mat(q) @ self._t_off, quat_mul(q, self._q_off)

    def _flange_to_link7(self, p, q):
        """Pose de la brida del SDK -> pose de link7 (lo que resuelve la IK)."""
        q7 = quat_mul(q, quat_conj(self._q_off))
        return p - quat_to_mat(q7) @ self._t_off, q7

    def _pose_to_arrays(self, pose):
        """geometry_msgs/Pose (convencion del SDK) -> (p, quat wxyz) de link7."""
        p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        q = np.array([pose.orientation.w, pose.orientation.x,
                      pose.orientation.y, pose.orientation.z], dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-9:
            raise ValueError('cuaternion nulo')
        return self._flange_to_link7(p, q / n)

    # -------------------------------------------------------- callbacks de movimiento
    def _on_move_j(self, msg: JointState):
        """control/move_j: objetivo articular; admite mensajes sueltos o flujos rapidos."""
        with self._data_lock:
            base = self._jgoal if self._jgoal is not None else self.data.ctrl[self._arm_aid]
            q_goal = np.array(base, dtype=float)
            starting = self._jgoal is None and self._traj is None
            moved = False
            for name, pos in zip(msg.name, msg.position):
                if name in ARM_JOINTS:
                    k = ARM_JOINTS.index(name)
                    clipped = min(max(pos, self._q_lo[k]), self._q_hi[k])
                    if clipped != pos:
                        self.get_logger().warn(
                            f'{name}={pos:.3f} fuera de rango [{self._q_lo[k]:.3f}, '
                            f'{self._q_hi[k]:.3f}]: se limita a {clipped:.3f}.')
                    q_goal[k] = clipped
                    moved = True
                elif name == 'gripper':
                    self._set_gripper_locked(pos)
            if moved:
                self._traj = None
                self._jgoal = q_goal
        if moved and starting:
            self.get_logger().info('move_j: nuevo objetivo articular.')

    def _on_move_p(self, msg: PoseStamped):
        try:
            p, q = self._pose_to_arrays(msg.pose)
        except ValueError as e:
            self.get_logger().warn(f'move_p rechazado: {e}')
            return
        q0 = self._q_now()
        sols = self._plan_path([(p, q)], q0)
        if sols is None:
            return
        T = self._start_trajectory([q0, sols[0]])
        if T:
            self.get_logger().info(f'move_p: {T:.1f} s')

    def _on_move_l(self, msg: PoseStamped):
        try:
            p1, qt1 = self._pose_to_arrays(msg.pose)
        except ValueError as e:
            self.get_logger().warn(f'move_l rechazado: {e}')
            return
        q0 = self._q_now()
        p0, qt0 = self._fk_base(q0)
        dist = float(np.linalg.norm(p1 - p0))
        ang = float(np.linalg.norm(quat_error(qt1, qt0)))
        n = int(np.clip(math.ceil(max(dist / 0.005, ang / 0.05)), 2, 400))
        poses = []
        for i in range(1, n + 1):
            t = i / n
            poses.append((p0 + (p1 - p0) * t, quat_slerp(qt0, qt1, t)))
        self.get_logger().info(f'move_l: planificando {n} puntos...')
        sols = self._plan_path(poses, q0, max_jump=0.8)
        if sols is None:
            return
        T = self._start_trajectory([q0] + sols, path_len=dist)
        if T:
            self.get_logger().info(f'move_l: {T:.1f} s')

    def _on_move_c(self, msg: PoseArray):
        q0 = self._q_now()
        try:
            arrs = [self._pose_to_arrays(p) for p in msg.poses]
        except ValueError as e:
            self.get_logger().warn(f'move_c rechazado: {e}')
            return
        if len(arrs) == 2:  # sin inicio explicito: se parte de la pose actual
            arrs = [self._fk_base(q0)] + arrs
        if len(arrs) != 3:
            self.get_logger().warn(
                f'move_c espera 3 poses (inicio, medio, fin), llegaron {len(arrs)}: rechazado.')
            return
        (ps, qs), (pm, qm), (pe, qe) = arrs
        _, _, length = arc_waypoints(ps, pm, pe, 2)
        ang = float(np.linalg.norm(quat_error(qm, qs)) + np.linalg.norm(quat_error(qe, qm)))
        n = int(np.clip(math.ceil(max(length / 0.005, ang / 0.05)), 8, 400))
        pts, fm, length = arc_waypoints(ps, pm, pe, n)
        poses = []
        for i in range(n + 1):
            f = i / n
            if fm > 1e-6 and f <= fm:
                qq = quat_slerp(qs, qm, f / fm)
            else:
                qq = quat_slerp(qm, qe, (f - fm) / max(1.0 - fm, 1e-6))
            poses.append((pts[i], qq))
        self.get_logger().info(f'move_c: planificando {n + 1} puntos...')
        sols = self._plan_path(poses, q0)
        if sols is None:
            return
        T = self._start_trajectory([q0] + sols, path_len=length)
        if T:
            self.get_logger().info(f'move_c: {T:.1f} s')

    # ------------------------------------------------------------------ feedback
    def _publish_feedback(self):
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        with self._data_lock:
            def add(name, info):
                js.name.append(name)
                js.position.append(float(self.data.qpos[info['qpos_adr']]))
                js.velocity.append(float(self.data.qvel[info['qvel_adr']]))

            # Mismo orden que el brazo real (joint1..7, gripper): hay nodos (ik_node) que
            # leen las posiciones por indice, sin mirar los nombres. Los joints propios de
            # MuJoCo (gripper_joint1/2...) van despues.
            for name in ARM_JOINTS:
                add(name, self._joint_info[name])
            g = self._joint_info.get('gripper_joint1')
            if g is not None:
                # 'gripper' = ancho total en metros (como el real).
                js.name.append('gripper')
                js.position.append(2.0 * float(self.data.qpos[g['qpos_adr']]))
                js.velocity.append(0.0)
            for name, info in self._joint_info.items():
                if name not in ARM_JOINTS:
                    add(name, info)
            p_w = self.data.xpos[self._flange_id].copy()
            R_w = self.data.xmat[self._flange_id].reshape(3, 3).copy()
        self.feedback_pub.publish(js)

        p_b = self._base_R.T @ (p_w - self._base_p)
        q_b = mat_to_quat(self._base_R.T @ R_w)
        p_b, q_b = self._link7_to_flange(p_b, q_b)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = BASE_BODY
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p_b)
        ps.pose.orientation.w = float(q_b[0])
        ps.pose.orientation.x = float(q_b[1])
        ps.pose.orientation.y = float(q_b[2])
        ps.pose.orientation.z = float(q_b[3])
        self.tcp_pub.publish(ps)

    # ------------------------------------------------------------- cubo + sensor
    def _setup_cube(self, cube_velocity, sensor_pose, sensor_rate, sensor_noise, sensor_max,
                    cube_delay=0.0, cube_source='velocity', sensor_topic='distance',
                    cube_track_tau=0.1, sensor_timeout=0.5):
        """Cubo opcional (body 'cube' + freejoint 'cube_free' en la escena).

        - Se mueve a velocidad constante hasta que el robot lo toca; desde ahi solo
          actua la fisica (contactos con los dedos).
        - /sim/cube_state: estado real del cubo [x, y, z, vx, vy, vz] (marco base_link),
          solo para depurar / comparar con la estimacion.
        - Sensor ultrasonico simulado (sensor_pose = [px, py, pz, dx, dy, dz], marco
          base_link): lanza un rayo y publica sensor_msgs/Range en 'distance', igual que
          ultrasonic_sensor_node con el Arduino real.
        - cube_source='sensor': no hay sensor simulado; el cubo sigue al sensor REAL
          (sensor_topic, sensor_pose obligatorio): centro = sensor + dir * (rango + a/2),
          suavizado con cube_track_tau. Lecturas fuera de [min_range, sensor_max] o sin
          datos por mas de sensor_timeout s: el cubo se queda donde esta.
        """
        self._cube_enabled = False
        try:
            cube_body = self.model.body('cube').id
            self._cube_geom = self.model.geom('cube_geom').id
            joint = self.model.joint('cube_free')
        except KeyError:
            return
        self._cube_qadr = int(joint.qposadr[0])
        self._cube_dadr = int(joint.dofadr[0])
        # The 'home' keyframe (nero.xml) lists only the arm joints; depending on the
        # MuJoCo version the missing cube_free values come out as zeros (cube at the
        # origin, inside the base) instead of qpos0. Restore the pose from the XML.
        self.data.qpos[self._cube_qadr:self._cube_qadr + 7] = \
            self.model.qpos0[self._cube_qadr:self._cube_qadr + 7]
        self.data.qvel[self._cube_dadr:self._cube_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._cube_vel = np.asarray(cube_velocity, dtype=float)
        if cube_source == 'sensor' and sensor_pose is None:
            self.get_logger().warn('cube_source=sensor necesita --sensor-pose: uso "velocity".')
            cube_source = 'velocity'
        self._cube_source = cube_source
        self._cube_half = float(self.model.geom_size[self._cube_geom][0])
        self._cube_released = False
        self._cube_delay = float(cube_delay)   # s of simulated time the cube waits still
        root = self.model.body_rootid[self.model.body(BASE_BODY).id]
        self._robot_bodies = {i for i in range(1, self.model.nbody)
                              if self.model.body_rootid[i] == root}
        self._cube_enabled = True
        self._cube_pub = self.create_publisher(Float64MultiArray, 'sim/cube_state', 10)
        self.create_timer(0.02, self._publish_cube_state)
        if sensor_pose is not None:
            p = np.asarray(sensor_pose[:3], dtype=float)
            d = np.asarray(sensor_pose[3:], dtype=float)
            # de base_link a mundo
            self._sensor_p = self._base_p + self._base_R @ p
            self._sensor_d = self._base_R @ (d / np.linalg.norm(d))

        if self._cube_source == 'sensor':
            self._sensor_max = sensor_max
            self._track_tau = float(cube_track_tau)
            self._sensor_timeout = float(sensor_timeout)
            self._measured = None    # (rango en m, time.monotonic()) de la ultima lectura valida
            self._track_pos = None   # centro suavizado del cubo (mundo)
            self.create_subscription(Range, sensor_topic, self._on_real_range, 10)
            self.get_logger().info(
                f'Cubo: sigue al sensor REAL en "{sensor_topic}" (sensor en {p.tolist()} '
                f'mirando {d.tolist()}, lecturas validas hasta {sensor_max} m). '
                'Sin sensor simulado.')
            return

        self.get_logger().info(f'Cubo: velocidad constante {self._cube_vel.tolist()} m/s '
                               f'desde t = {self._cube_delay:.1f} s')

        if sensor_pose is not None:
            self._sensor_noise = sensor_noise
            self._sensor_max = sensor_max
            self._rng = np.random.default_rng()
            self._range_pub = self.create_publisher(Range, 'distance', 10)
            self.create_timer(1.0 / sensor_rate, self._publish_range)
            self.get_logger().info(
                f'Sensor ultrasonico simulado en {p.tolist()} mirando {d.tolist()} '
                f'({sensor_rate} Hz, ruido {sensor_noise} m) -> topic "distance"')

    def _update_cube(self):
        """Llamar con _data_lock tomado, antes de mj_step."""
        if not self._cube_enabled or self._cube_released:
            return
        for c in self.data.contact[:self.data.ncon]:
            if self._cube_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._cube_geom else c.geom1
                if int(self.model.geom_bodyid[other]) in self._robot_bodies:
                    self._cube_released = True
                    self.get_logger().info('El robot toco el cubo: se libera la velocidad constante.')
                    return
        if self._cube_source == 'sensor':
            self._track_real_sensor()
            return
        moving = self.data.time >= self._cube_delay
        self.data.qvel[self._cube_dadr:self._cube_dadr + 3] = self._cube_vel if moving else 0.0
        self.data.qvel[self._cube_dadr + 3:self._cube_dadr + 6] = 0.0

    def _on_real_range(self, msg):
        r = float(msg.range)
        # -1 / NaN / fuera de rango (sin cubo en la cinta): se ignora
        if np.isfinite(r) and max(msg.min_range, 0.0) <= r <= self._sensor_max:
            self._measured = (r, time.monotonic())

    def _track_real_sensor(self):
        """Lleva el cubo a la posicion medida por el sensor real (suavizada).
        Llamar con _data_lock tomado, antes de mj_step."""
        q = slice(self._cube_qadr, self._cube_qadr + 3)
        if self._track_pos is None:
            self._track_pos = self.data.qpos[q].copy()
        dt = self.model.opt.timestep
        meas = self._measured
        vel = np.zeros(3)
        if meas is not None and time.monotonic() - meas[1] <= self._sensor_timeout:
            target = self._sensor_p + self._sensor_d * (meas[0] + self._cube_half)
            alpha = 1.0 - np.exp(-dt / self._track_tau) if self._track_tau > 0 else 1.0
            new = self._track_pos + alpha * (target - self._track_pos)
            vel = (new - self._track_pos) / dt
            self._track_pos = new
        self.data.qpos[q] = self._track_pos
        self.data.qpos[self._cube_qadr + 3:self._cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._cube_dadr:self._cube_dadr + 3] = vel
        self.data.qvel[self._cube_dadr + 3:self._cube_dadr + 6] = 0.0

    def _publish_cube_state(self):
        with self._data_lock:
            p = self.data.qpos[self._cube_qadr:self._cube_qadr + 3].copy()
            v = self.data.qvel[self._cube_dadr:self._cube_dadr + 3].copy()
        msg = Float64MultiArray()
        msg.data = np.concatenate((self._base_R.T @ (p - self._base_p),
                                   self._base_R.T @ v)).tolist()
        self._cube_pub.publish(msg)

    def _publish_range(self):
        geomid = np.zeros(1, dtype=np.int32)
        with self._data_lock:
            dist = mujoco.mj_ray(self.model, self.data, self._sensor_p, self._sensor_d,
                                 None, 1, -1, geomid)
        if dist < 0 or dist > self._sensor_max:
            dist = self._sensor_max   # sin eco: el sensor real reporta su maximo
        elif self._sensor_noise > 0:
            dist += self._rng.normal(0.0, self._sensor_noise)
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ultrasonic_link'
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.26
        msg.min_range = 0.02
        msg.max_range = float(self._sensor_max)
        msg.range = float(dist)
        self._range_pub.publish(msg)

    def step(self):
        with self._data_lock:
            self._advance_trajectory()
            self._update_cube()
            mujoco.mj_step(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description='Puente ROS2 <-> MuJoCo para el brazo Nero')
    parser.add_argument('scene', help='Ruta absoluta al archivo de escena MuJoCo (.xml)')
    parser.add_argument('--control-topic', default='control/joint_states')
    parser.add_argument('--move-j-topic', default='control/move_j')
    parser.add_argument('--feedback-topic', default='feedback/joint_states')
    parser.add_argument('--rate', type=float, default=200.0, help='Tasa de publicacion de feedback (Hz)')
    parser.add_argument('--joint-speed', type=float, default=0.8,
                        help='Velocidad articular media de move_j/p/l/c (rad/s)')
    parser.add_argument('--lin-speed', type=float, default=0.1,
                        help='Velocidad lineal media de move_l/c (m/s)')
    parser.add_argument('--no-gravcomp', action='store_true',
                        help='Desactiva la compensacion de gravedad del brazo simulado')
    env_off = os.environ.get('NERO_FLANGE_OFFSET', '').split()
    parser.add_argument('--flange-offset', type=float, nargs=7,
                        metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
                        default=[float(v) for v in env_off] if len(env_off) == 7
                        else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        help='Transformacion link7 -> brida del SDK (en el marco de link7)')
    parser.add_argument('--flange-zero-identity', action='store_true',
                        default=os.environ.get('NERO_FLANGE_ZERO_IDENTITY', '') in ('1', 'true', 'True'),
                        help='Orientacion de la brida relativa a la pose en cero (identidad con q=0)')
    parser.add_argument('--cube-velocity', type=float, nargs=3, default=[0.0, 0.05, 0.0],
                        metavar=('VX', 'VY', 'VZ'),
                        help='Velocidad constante del cubo (m/s, marco mundo), si la escena lo tiene')
    parser.add_argument('--sensor-pose', type=float, nargs=6, default=None,
                        metavar=('PX', 'PY', 'PZ', 'DX', 'DY', 'DZ'),
                        help='Sensor ultrasonico simulado: posicion y direccion (marco base_link)')
    parser.add_argument('--cube-delay', type=float, default=0.0,
                        help='Segundos (simulados) que el cubo espera quieto antes de moverse')
    parser.add_argument('--sensor-rate', type=float, default=20.0, help='Hz del sensor simulado')
    parser.add_argument('--sensor-noise', type=float, default=0.003,
                        help='Ruido gaussiano del sensor simulado (m, desviacion estandar)')
    parser.add_argument('--sensor-max', type=float, default=4.0,
                        help='Alcance maximo (m). Con --cube-source sensor: lecturas mas '
                             'lejanas se ignoran (no hay cubo en la cinta)')
    parser.add_argument('--cube-source', choices=['velocity', 'sensor'], default='velocity',
                        help='velocity: el cubo avanza a --cube-velocity (sensor simulado); '
                             'sensor: el cubo sigue al sensor real (requiere --sensor-pose)')
    parser.add_argument('--sensor-topic', default='distance',
                        help='Topic del sensor real (--cube-source sensor)')
    parser.add_argument('--cube-track-tau', type=float, default=0.1,
                        help='Suavizado (s) de la posicion medida (--cube-source sensor)')
    parser.add_argument('--sensor-timeout', type=float, default=0.5,
                        help='Sin lecturas por mas de esto (s) el cubo se detiene '
                             '(--cube-source sensor)')
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    bridge = MujocoBridge(args.scene, args.control_topic, args.feedback_topic, args.rate,
                          args.move_j_topic, args.joint_speed, args.lin_speed,
                          not args.no_gravcomp, args.flange_offset, args.flange_zero_identity,
                          args.cube_velocity, args.sensor_pose, args.sensor_rate,
                          args.sensor_noise, args.sensor_max, args.cube_delay,
                          args.cube_source, args.sensor_topic, args.cube_track_tau,
                          args.sensor_timeout)

    # rclpy.spin en un hilo aparte; el hilo principal corre el viewer + la fisica
    # (mujoco.viewer.launch_passive debe llamarse desde el hilo principal).
    spin_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    spin_thread.start()

    try:
        with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
            sim_ref = bridge.data.time
            wall_ref = time.perf_counter()
            last_sync = wall_ref
            last_report = wall_ref
            report_sim = bridge.data.time

            while viewer.is_running() and rclpy.ok():
                bridge.step()
                now = time.perf_counter()

                # Refrescar la ventana ~60 Hz (no en cada paso de fisica: sync es caro).
                if now - last_sync >= 1.0 / 60.0:
                    viewer.sync()
                    last_sync = now

                # Ritmo real sin deriva: tiempo simulado vs tiempo de pared acumulado.
                ahead = (bridge.data.time - sim_ref) - (time.perf_counter() - wall_ref)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.5:
                    sim_ref = bridge.data.time
                    wall_ref = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()