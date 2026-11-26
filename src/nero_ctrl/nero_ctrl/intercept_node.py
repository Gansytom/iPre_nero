#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

"""Intercepts the cube moving on the conveyor and picks it up (real arm or MuJoCo bridge).

Inputs
  /cube_state            [x, y, z, vx, vy, vz] (cube_tracker_node), robot base frame
  feedback/joint_states  joint1..7 + 'gripper' (total width, m)  -- real arm and bridge
  ik/ik_result           last joint goal computed by ik_node (to know when the arm arrived)

Outputs (can_module is the only node that talks to the arm)
  joint_goal             JointState, joint-space goal (ready pose)
  input_obj_position     [x, y, z, roll, pitch, yaw (deg), gripper] -> main_node (deg->rad)
                         -> ik_node -> can_module (gripper value ignored: forward_ik_gripper=false)
  gripper_command        Float64, gripper width in m

Unlike the simulation-only version (ros2_to_mujoco_nero), every step waits until the
arm actually reached its goal (feedback), because the real arm speed is not known
in advance. If the arm gets to the conveyor too late it re-plans a later intercept
point, and if the grasp fails it retreats and tries again with the next estimate.

States
  WAIT_ROBOT -> MOVE_READY -> WAIT_CUBE -> PREGRASP -> DESCEND -> WAIT_ARRIVAL
  -> CLOSE -> LIFT -> DONE            (RETREAT -> MOVE_READY on a missed grasp)
"""

ARM_JOINTS = [f'joint{i}' for i in range(1, 8)]


class InterceptNode(Node):
    def __init__(self):
        super().__init__('intercept_node')

        # ---------------- parameters (see config/cube_intercept.yaml) ----------------
        self.declare_parameter('ready_joints', [0.724, 0.69, -1.208, 1.49, 0.743, 0.527, 1.382])
        self.declare_parameter('grasp_rpy', [0.0, np.pi / 2, np.radians(-150.0)])
        self.declare_parameter('intercept_x_range', [-0.45, -0.30])
        self.declare_parameter('intercept_y_range', [-0.10, 0.15])
        self.declare_parameter('pregrasp_z', 0.16)
        self.declare_parameter('grasp_z_offset', -0.004)
        self.declare_parameter('lift_z', 0.16)
        self.declare_parameter('gripper_open', 0.10)       # total width, m
        self.declare_parameter('gripper_closed', 0.0)
        self.declare_parameter('grasp_min_width', 0.015)   # narrower than this = missed
        # Time from "start moving to the conveyor" until the gripper waits at grasp
        # height (IK + pregrasp + descend). Too short is recovered by re-planning.
        self.declare_parameter('approach_time', 5.0)
        self.declare_parameter('descend_time', 2.0)
        self.declare_parameter('time_margin', 0.5)
        self.declare_parameter('close_lead_time', 0.15)    # s the fingers need to close
        self.declare_parameter('close_wait', 1.5)          # s before checking the grasp
        self.declare_parameter('min_speed', 0.01)          # m/s, slower = static cube
        # arrival detection
        self.declare_parameter('joint_tolerance', 0.02)    # rad
        self.declare_parameter('settle_time', 0.3)         # s without new IK goals
        self.declare_parameter('move_timeout', 20.0)       # s
        self.declare_parameter('cube_timeout', 3.0)        # s to keep predicting without data
        self.declare_parameter('max_attempts', 3)

        p = lambda name: self.get_parameter(name).value
        self.ready_joints = np.asarray(p('ready_joints'), dtype=float)
        self.grasp_rpy = [float(v) for v in p('grasp_rpy')]
        self.x_range = [float(v) for v in p('intercept_x_range')]
        self.y_range = [float(v) for v in p('intercept_y_range')]
        self.pregrasp_z = float(p('pregrasp_z'))
        self.grasp_z_offset = float(p('grasp_z_offset'))
        self.lift_z = float(p('lift_z'))
        self.gripper_open = float(p('gripper_open'))
        self.gripper_closed = float(p('gripper_closed'))
        self.grasp_min_width = float(p('grasp_min_width'))
        self.approach_time = float(p('approach_time'))
        self.descend_time = float(p('descend_time'))
        self.time_margin = float(p('time_margin'))
        self.close_lead_time = float(p('close_lead_time'))
        self.close_wait = float(p('close_wait'))
        self.min_speed = float(p('min_speed'))
        self.joint_tol = float(p('joint_tolerance'))
        self.settle_time = float(p('settle_time'))
        self.move_timeout = float(p('move_timeout'))
        self.cube_timeout = float(p('cube_timeout'))
        self.max_attempts = int(p('max_attempts'))

        # ---------------- state ----------------
        self.state = None
        self.state_start = 0.0
        self.q = None                 # current arm joints
        self.gripper_width = None
        self.cube = None              # (pos, vel, receive time)
        self.goal_q = None            # joint goal the arm is moving to
        self.goal_time = None         # when that goal was last updated
        self.waiting_ik = False       # a Cartesian target was sent, its IK result is pending
        self.goal_source = None       # 'joint' (joint_goal) or 'ik' (Cartesian target)
        self.intercept = None
        self.motion_dir = None
        self.attempts = 0
        self.gripper_target = self.gripper_open   # last commanded width

        # ---------------- ROS interfaces ----------------
        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.joint_goal_pub = self.create_publisher(JointState, 'joint_goal', 10)
        self.target_pub = self.create_publisher(Float64MultiArray, 'input_obj_position', 10)
        self.gripper_pub = self.create_publisher(Float64, 'gripper_command', 10)

        self.create_subscription(JointState, 'feedback/joint_states', self.feedback_callback, 10)
        self.create_subscription(JointState, 'ik/ik_result', self.ik_result_callback, latch_qos)
        self.create_subscription(Float64MultiArray, '/cube_state', self.cube_callback, 10)

        self.set_state('WAIT_ROBOT')
        self.create_timer(0.02, self.step)
        self.get_logger().info('Intercept node started.')

    # ======================= callbacks =======================
    def feedback_callback(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in ARM_JOINTS):
            self.q = np.array([pos[n] for n in ARM_JOINTS])
        if 'gripper' in pos:
            self.gripper_width = float(pos['gripper'])

    def ik_result_callback(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if not all(n in pos for n in ARM_JOINTS):
            return
        if self.goal_source != 'ik':
            return   # the arm is following a joint-space goal, not an IK result
        self.goal_q = np.array([pos[n] for n in ARM_JOINTS])
        self.goal_time = self.now()
        self.waiting_ik = False

    def cube_callback(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.shape == (6,):
            self.cube = (data[:3], data[3:], self.now())

    # ======================= helpers =======================
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def elapsed(self):
        return self.now() - self.state_start

    def set_state(self, state):
        self.state = state
        self.state_start = self.now()
        self.get_logger().info(f'State -> {state}')

    def cube_prediction(self, max_age=None):
        """Predicted cube (pos, vel) now, or None if there is no recent estimate."""
        if self.cube is None:
            return None
        pos, vel, t = self.cube
        age = self.now() - t
        if age > (self.cube_timeout if max_age is None else max_age):
            return None
        return pos + vel * age, vel

    def send_joint_goal(self, q):
        msg = JointState()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(v) for v in q]
        self.joint_goal_pub.publish(msg)
        self.goal_q = np.asarray(q, dtype=float)
        self.goal_time = self.now()
        self.goal_source = 'joint'
        self.waiting_ik = False

    def send_pose(self, xyz):
        msg = Float64MultiArray()
        # main_node expects roll/pitch/yaw in degrees and a 7th gripper element; the
        # gripper is driven by gripper_command, the value here is only informative.
        msg.data = ([float(v) for v in xyz] + [float(np.degrees(a)) for a in self.grasp_rpy]
                    + [float(self.gripper_target)])
        self.goal_source = 'ik'
        self.waiting_ik = True
        self.goal_q = None
        self.target_pub.publish(msg)
        self.get_logger().info(f'Target pose: {np.round(msg.data, 3).tolist()}')

    def send_gripper(self, width):
        self.gripper_target = float(width)
        self.gripper_pub.publish(Float64(data=float(width)))

    def arrived(self):
        """True when the arm is at its goal and no new IK goals arrived for settle_time."""
        if self.q is None or self.goal_q is None or self.waiting_ik:
            return False
        if self.now() - self.goal_time < self.settle_time:
            return False
        return np.max(np.abs(self.q - self.goal_q)) < self.joint_tol

    def move_timed_out(self):
        if self.elapsed() > self.move_timeout:
            self.get_logger().error(f'{self.state}: the arm did not reach its goal in '
                                    f'{self.move_timeout:.0f} s (IK failed or arm blocked).')
            return True
        return False

    def in_region(self, point):
        return (self.x_range[0] <= point[0] <= self.x_range[1]
                and self.y_range[0] <= point[1] <= self.y_range[1])

    def plan_intercept(self, t_needed):
        """Earliest point of the predicted straight path inside the reachable region
        that the cube reaches after t_needed seconds. Static cube: its own position."""
        pred = self.cube_prediction()
        if pred is None:
            return None
        pos, vel = pred
        speed = np.linalg.norm(vel)
        if speed < self.min_speed:
            return (pos, 0.0, None) if self.in_region(pos) else None
        for t in np.arange(t_needed, 60.0, 0.05):
            point = pos + vel * t
            if self.in_region(point):
                return point, t, vel / speed
        return None

    def time_to_intercept(self):
        """Seconds until the cube reaches the intercept point (negative = it passed)."""
        pred = self.cube_prediction()
        if pred is None or self.motion_dir is None:
            return None
        pos, vel = pred
        speed = max(np.linalg.norm(vel), 1e-6)
        return np.dot(self.intercept - pos, self.motion_dir) / speed

    def go_pregrasp(self, t_needed, reason=''):
        plan = self.plan_intercept(t_needed)
        if plan is None:
            if self.state == 'WAIT_CUBE':
                # already at the ready pose: keep waiting (cube still, or out of reach)
                self.get_logger().info('No reachable intercept point yet; waiting.',
                                       throttle_duration_sec=3.0)
                return
            self.get_logger().warn(f'{reason}No reachable intercept point; back to ready.')
            self.send_joint_goal(self.ready_joints)
            self.set_state('MOVE_READY')
            return
        self.intercept, t_arrival, self.motion_dir = plan
        self.get_logger().info(
            f'{reason}Intercept point {np.round(self.intercept, 3).tolist()}, '
            f'cube arrives in {t_arrival:.2f} s')
        self.send_pose([self.intercept[0], self.intercept[1], self.pregrasp_z])
        self.set_state('PREGRASP')

    # ======================= state machine =======================
    def step(self):
        s = self.state

        if s == 'WAIT_ROBOT':
            # Commands are not latched: wait until the arm feedback arrives AND can_module /
            # main_node are subscribed, otherwise the first commands are lost.
            connected = (self.joint_goal_pub.get_subscription_count() > 0
                         and self.gripper_pub.get_subscription_count() > 0
                         and self.target_pub.get_subscription_count() > 0)
            if self.q is None or not connected:
                if self.elapsed() > 5.0:
                    self.get_logger().warn(
                        'Waiting for feedback/joint_states and for can_module/main_node...',
                        throttle_duration_sec=5.0)
                return
            self.send_gripper(self.gripper_open)
            self.send_joint_goal(self.ready_joints)
            self.set_state('MOVE_READY')

        elif s == 'MOVE_READY':
            if self.arrived():
                self.set_state('WAIT_CUBE')
            elif self.move_timed_out():
                self.set_state('DONE')

        elif s == 'WAIT_CUBE':
            if self.attempts >= self.max_attempts:
                self.get_logger().error(f'Giving up after {self.attempts} attempts.')
                self.set_state('DONE')
                return
            if self.cube_prediction(max_age=0.3) is None:
                return   # wait for a fresh estimate
            self.go_pregrasp(self.approach_time)

        elif s == 'PREGRASP':
            if self.move_timed_out():
                self.set_state('RETREAT')
                return
            if not self.arrived():
                return
            if self.motion_dir is None:   # static cube
                self.descend()
                return
            t_left = self.time_to_intercept()
            if t_left is None or t_left < self.descend_time + self.time_margin:
                # arrived too late for this point: choose a later one
                self.go_pregrasp(self.descend_time + self.time_margin,
                                 reason='Too late for the intercept point. ')
                return
            self.descend()

        elif s == 'DESCEND':
            if self.move_timed_out():
                self.set_state('RETREAT')
                return
            if self.motion_dir is not None:
                t_left = self.time_to_intercept()
                if t_left is not None and t_left < -0.05:
                    self.get_logger().warn('The cube passed before the gripper got down.')
                    self.retreat()
                    return
            if self.arrived():
                self.set_state('WAIT_ARRIVAL')

        elif s == 'WAIT_ARRIVAL':
            if self.motion_dir is None:
                self.close()
                return
            t_left = self.time_to_intercept()
            if t_left is None:
                self.get_logger().warn('Lost the cube estimate.')
                self.retreat()
            elif t_left <= self.close_lead_time:
                self.close()

        elif s == 'CLOSE':
            if self.elapsed() < self.close_wait:
                return
            width = self.gripper_width
            if width is not None and width < self.grasp_min_width:
                self.get_logger().warn(f'Missed the cube (gripper width {width:.3f} m).')
                self.retreat()
                return
            self.get_logger().info(
                f'Cube grasped (gripper width {width if width is not None else float("nan"):.3f} m). Lifting.')
            self.send_pose([self.intercept[0], self.intercept[1], self.lift_z])
            self.set_state('LIFT')

        elif s == 'LIFT':
            if self.arrived() or self.move_timed_out():
                self.get_logger().info('Done: cube lifted.')
                self.set_state('DONE')

        elif s == 'RETREAT':
            if self.arrived() or self.elapsed() > self.move_timeout:
                self.send_joint_goal(self.ready_joints)
                self.set_state('MOVE_READY')

    def descend(self):
        p = self.intercept
        self.send_pose([p[0], p[1], p[2] + self.grasp_z_offset])
        self.set_state('DESCEND')

    def close(self):
        self.get_logger().info('Cube at the intercept point: closing the gripper.')
        self.send_gripper(self.gripper_closed)
        self.attempts += 1
        self.set_state('CLOSE')

    def retreat(self):
        """Open the gripper and go up; then back to the ready pose to try again."""
        self.send_gripper(self.gripper_open)
        if self.intercept is not None:
            self.send_pose([self.intercept[0], self.intercept[1], self.pregrasp_z])
        self.set_state('RETREAT')


def main(args=None):
    rclpy.init(args=args)
    node = InterceptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
