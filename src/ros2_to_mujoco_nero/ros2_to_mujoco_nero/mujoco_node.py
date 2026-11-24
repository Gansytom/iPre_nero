#!/usr/bin/env python3

import rclpy
import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool, Float64, Float64MultiArray
from sensor_msgs.msg import JointState, Range
from rosgraph_msgs.msg import Clock

"""This node handle the mujoco simulator"""


def find_scene_xml():
    """Locate scene.xml without hardcoded paths.

    Priority: env var NERO_SCENE_XML, then search upwards from this file
    for src/models_for_simulation/agilex_nero/scene.xml.
    """
    import os
    from pathlib import Path
    env = os.environ.get('NERO_SCENE_XML')
    if env and Path(env).is_file():
        return env
    rel = Path('src/models_for_simulation/agilex_nero/scene.xml')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / rel
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        'scene.xml not found. Set NERO_SCENE_XML to its full path.')


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')
        self.get_logger().info('Mujoco node has been started.')
        
         # QoS profile 
        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                
        # atributes
        self.timer_pr = 0.01 # 100Hz
        self.start_requested = False
        self.ready_sent = False
        self.last_command = None
        # This may change if needed
        self.model_path = find_scene_xml()

        # Moving cube (part 2) and gripper (part 3)
        # cube_velocity: constant linear velocity [vx, vy, vz] in m/s, world frame
        self.declare_parameter('cube_velocity', [0.0, 0.05, 0.0])
        self.cube_velocity = np.asarray(
            self.get_parameter('cube_velocity').value, dtype=float)
        # cube_source: 'velocity' -> constant cube_velocity (default)
        #              'sensor'   -> cube position follows the ultrasonic /distance topic
        self.declare_parameter('cube_source', 'velocity')
        self.declare_parameter('distance_topic', '/distance')
        # Ultrasonic sensor pose in the world frame: cube center =
        # sensor_position + sensor_direction * (range + cube half size)
        self.declare_parameter('sensor_position', [-0.38, -0.40, 0.084])
        self.declare_parameter('sensor_direction', [0.0, 1.0, 0.0])
        self.declare_parameter('cube_track_tau', 0.1)   # s, smoothing of the measured position
        self.declare_parameter('sensor_timeout', 0.5)   # s, hold the cube if no fresh reading
        self.cube_source = self.get_parameter('cube_source').value
        if self.cube_source not in ('velocity', 'sensor'):
            self.get_logger().warn(f'Unknown cube_source "{self.cube_source}", using "velocity".')
            self.cube_source = 'velocity'
        self.sensor_position = np.asarray(
            self.get_parameter('sensor_position').value, dtype=float)
        direction = np.asarray(self.get_parameter('sensor_direction').value, dtype=float)
        self.sensor_direction = direction / np.linalg.norm(direction)
        self.cube_track_tau = float(self.get_parameter('cube_track_tau').value)
        self.sensor_timeout = float(self.get_parameter('sensor_timeout').value)
        self.measured_range = None       # m, last valid reading
        self.measured_time = None        # wall time of the last valid reading
        self.cube_track_pos = None       # smoothed cube center while tracking the sensor

        self.cube_enabled = False
        self.cube_released = False   # True once the robot touches the cube
        self.cube_grasped = False    # True once both fingers touch the cube
        self.gripper_command = 0.05  # gripper starts open (range 0.0 closed .. 0.05 open)
        self.gripper_actuator_id = None

        # publishers
        self.publisher_state_ready = self.create_publisher(Bool, '/mujoco_ready', latch_qos)
        self.publisher_joint_state = self.create_publisher(JointState, '/joint_states', latch_qos)
        self.publisher_clock_sim = self.create_publisher(Clock, '/clock', latch_qos)
        # [x, y, z, vx, vy, vz] of the cube center, world frame
        self.publisher_cube_state = self.create_publisher(Float64MultiArray, '/cube_state', 10)
        self.publisher_cube_grasped = self.create_publisher(Bool, '/cube_grasped', latch_qos)

        #subscriptions
        self.subscription = self.create_subscription(
            Bool,
            '/start_running_nodes',
            self.start_running_nodes_callback,
            latch_qos,
        )
        self.subscription = self.create_subscription(Float64MultiArray, '/inverse_kinematics_result', self.inverse_kinematics_result_callback, 10)
        # gripper opening command, overrides the 8th value of the IK result
        self.gripper_subscription = self.create_subscription(Float64, '/gripper_command', self.gripper_command_callback, 10)
        if self.cube_source == 'sensor':
            self.distance_subscription = self.create_subscription(
                Range, self.get_parameter('distance_topic').value, self.distance_callback, 10)

    def distance_callback(self, msg):
        # Arduino sends -1 cm for invalid readings; also drop anything outside the sensor range
        if not np.isfinite(msg.range) or not (msg.min_range <= msg.range <= msg.max_range):
            return
        self.measured_range = float(msg.range)
        self.measured_time = time.monotonic()

    def gripper_command_callback(self, msg):
        self.gripper_command = float(np.clip(msg.data, 0.0, 0.05))
        self.get_logger().info(f'Gripper command: {self.gripper_command:.3f}')

    def setup_scene(self, model):
        """Look up the ids of the cube and the gripper inside the loaded model."""
        try:
            self.gripper_actuator_id = model.actuator('gripper').id
        except KeyError:
            self.get_logger().warn('No "gripper" actuator in the model.')
        try:
            self.cube_body_id = model.body('cube').id
            self.cube_geom_id = model.geom('cube_geom').id
            cube_joint = model.joint('cube_free')
        except KeyError:
            self.get_logger().warn('No cube in the scene; moving-cube features disabled.')
            return
        self.cube_qpos_adr = int(cube_joint.qposadr[0])   # 7 values: pos(3) + quat(4)
        self.cube_dof_adr = int(cube_joint.dofadr[0])     # 6 values: lin vel(3) + ang vel(3)
        self.finger_body_ids = {model.body('gripper_link1').id, model.body('gripper_link2').id}
        root = model.body('base_link').id
        # every body whose root ancestor is the robot base
        self.robot_body_ids = {
            i for i in range(model.nbody)
            if model.body_rootid[i] == model.body_rootid[root] and i != 0
        }
        # distance from the cube face seen by the sensor to the cube center
        half_size = model.geom_size[self.cube_geom_id]
        self.cube_half_extent = float(np.abs(self.sensor_direction) @ half_size)
        self.cube_enabled = True
        if self.cube_source == 'sensor':
            self.get_logger().info(
                f'Cube found. Position follows {self.get_parameter("distance_topic").value} '
                f'(sensor at {self.sensor_position.tolist()}, '
                f'pointing {self.sensor_direction.tolist()}).')
        else:
            self.get_logger().info(
                f'Cube found. Constant velocity: {self.cube_velocity.tolist()} m/s')

    def track_sensor(self, model, data):
        """Move the cube to the position measured by the ultrasonic sensor (smoothed)."""
        pos_adr = slice(self.cube_qpos_adr, self.cube_qpos_adr + 3)
        if self.cube_track_pos is None:
            self.cube_track_pos = data.qpos[pos_adr].copy()

        fresh = (self.measured_time is not None
                 and time.monotonic() - self.measured_time <= self.sensor_timeout)
        if fresh:
            target = self.sensor_position + self.sensor_direction * (
                self.measured_range + self.cube_half_extent)
            dt = model.opt.timestep
            alpha = 1.0 - np.exp(-dt / self.cube_track_tau) if self.cube_track_tau > 0 else 1.0
            new_pos = self.cube_track_pos + alpha * (target - self.cube_track_pos)
            velocity = (new_pos - self.cube_track_pos) / dt
            self.cube_track_pos = new_pos
        else:
            velocity = np.zeros(3)   # no fresh reading: hold the cube where it is

        data.qpos[pos_adr] = self.cube_track_pos
        data.qpos[self.cube_qpos_adr + 3:self.cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.cube_dof_adr:self.cube_dof_adr + 3] = velocity
        data.qvel[self.cube_dof_adr + 3:self.cube_dof_adr + 6] = 0.0

    def update_cube(self, model, data):
        """Impose the constant velocity until the robot touches the cube, detect the grasp."""
        if not self.cube_enabled:
            return
        # Bodies in contact with the cube
        touching = set()
        for contact in data.contact[:data.ncon]:
            if self.cube_geom_id in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self.cube_geom_id else contact.geom1
                touching.add(int(model.geom_bodyid[other]))

        if not self.cube_released and touching & self.robot_body_ids:
            # From now on the cube is free: only physics (contacts with the gripper) acts on it
            self.cube_released = True
            self.get_logger().info('Robot touched the cube: constant velocity released.')

        if not self.cube_grasped and self.finger_body_ids <= touching:
            # Both fingers hold the cube: give it back its weight, the gripper must carry it
            self.cube_grasped = True
            model.body_gravcomp[self.cube_body_id] = 0.0
            self.publisher_cube_grasped.publish(Bool(data=True))
            self.get_logger().info('Cube grasped by both fingers: gravity enabled on the cube.')

        if not self.cube_released and self.cube_source == 'sensor':
            self.track_sensor(model, data)
        elif not self.cube_released:
            # Constant velocity: linear part = cube_velocity, no rotation
            data.qvel[self.cube_dof_adr:self.cube_dof_adr + 3] = self.cube_velocity
            data.qvel[self.cube_dof_adr + 3:self.cube_dof_adr + 6] = 0.0

    def publish_cube_state(self, data):
        if not self.cube_enabled:
            return
        pos = data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3]
        vel = data.qvel[self.cube_dof_adr:self.cube_dof_adr + 3]
        msg = Float64MultiArray()
        msg.data = np.concatenate((pos, vel)).tolist()
        self.publisher_cube_state.publish(msg)

    def inverse_kinematics_result_callback(self, msg):
        # get the data form the msg
        self.get_logger().info(f"Received inverse kinematics result: {msg.data}")
        command = np.asarray(msg.data, dtype=float)
        if command.shape != (8,) or not np.all(np.isfinite(command)):
            self.get_logger().error('Ignoring invalid IK command; expected 8 finite actuator positions.')
            return
        self.last_command = command
        
        
    def start_running_nodes_callback(self, msg):
        if msg.data and not self.start_requested:
            self.get_logger().info('Starting Mujoco simulator.')
            self.start_requested = True
    
    def update_simulation(self, model, data):
        # apply the last command received from the inverse kinematics node
        if self.last_command is not None:
            num_actuators = min(len(self.last_command), len(data.ctrl))
            data.ctrl[:num_actuators] = self.last_command[:num_actuators]
        # the gripper is commanded separately (/gripper_command)
        if self.gripper_actuator_id is not None:
            data.ctrl[self.gripper_actuator_id] = self.gripper_command

        # moving cube: constant velocity / grasp detection
        self.update_cube(model, data)

        # update simulation
        mujoco.mj_step(model, data)
        
        # share clock
        clock_msg = Clock()
        # convert units
        sec = int(data.time)
        nanosec = int((data.time - sec) * 1e9)
        clock_msg.clock.sec = sec
        clock_msg.clock.nanosec = nanosec
        self.publisher_clock_sim.publish(clock_msg)
        
        # share joint states
        joint_state_msg = JointState()
        joint_state_msg.position = data.qpos[:7].tolist()
        self.publisher_joint_state.publish(joint_state_msg)

        # share cube state
        self.publish_cube_state(data)

        
            
def main(args=None):
    rclpy.init(args=args)
    mujoco_node = MujocoNode()
    try:
        model = mujoco.MjModel.from_xml_path(mujoco_node.model_path)
        data = mujoco.MjData(model)
        mujoco_node.setup_scene(model)

        mujoco_node.get_logger().info(
            'Waiting for /start_running_nodes...'
        )

        while rclpy.ok() and not mujoco_node.start_requested:
            rclpy.spin_once(mujoco_node, timeout_sec=0.1)

        if not rclpy.ok():
            return

        mujoco_node.get_logger().info(
            'Start message received. Launching MuJoCo.'
        )

        with mujoco.viewer.launch_passive(model, data) as viewer:
            if not mujoco_node.ready_sent:
                mujoco_node.publisher_state_ready.publish(Bool(data=True))
                mujoco_node.ready_sent = True

            # Real-time pacing: the simulation advances one timestep per iteration and
            # waits until the wall clock catches up. The viewer is refreshed at ~60 FPS
            # (syncing it every 2 ms step made the simulation ~4x slower than real time).
            wall_start = time.perf_counter()
            sim_start = data.time
            last_sync = 0.0
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(mujoco_node, timeout_sec=0.0)

                mujoco_node.update_simulation(model, data)

                now = time.perf_counter()
                if now - last_sync >= 1.0 / 60.0:
                    viewer.sync()
                    last_sync = now

                ahead = (data.time - sim_start) - (time.perf_counter() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
    except Exception as e:
        mujoco_node.get_logger().error(f"Failed to load Mujoco model or launch viewer: {e}")
       
    finally:
        mujoco_node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
    
# Mensaje de prueba para levantar el nodo de manera individual
# ros2 topic pub /start_running_nodes std_msgs/msg/Bool "{data: true}" -1
# publisher: beginning loop
# publishing #1: std_msgs.msg.Bool(data=True)
