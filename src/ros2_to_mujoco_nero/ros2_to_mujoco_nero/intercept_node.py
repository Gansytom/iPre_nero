#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray

"""Intercepts the cube that moves at constant velocity and picks it up.

Reads the cube state published by mujoco_node (/cube_state), predicts its
straight-line trajectory p(t) = p0 + v*t, chooses an intercept point that the
arm can reach before the cube does, and drives the arm through the existing
chain:  input_obj_position -> main_node -> /obj_point -> inverse_kinematics_node
-> /inverse_kinematics_result -> mujoco_node.  The gripper is commanded through
/gripper_command.

State machine:
  WAIT_CUBE -> READY -> PREGRASP -> DESCEND -> WAIT_ARRIVAL -> CLOSE -> LIFT -> DONE
"""

GRIPPER_OPEN = 0.05
GRIPPER_CLOSED = 0.0


class InterceptNode(Node):
    def __init__(self):
        super().__init__('intercept_node')

        # --- parameters -----------------------------------------------------
        # Joint configuration with the gripper above the conveyor pointing down.
        # The IK (Newton-Raphson from the current pose) cannot reach a top-down pose
        # starting from the initial "home" pose (it gets stuck at the joint limits),
        # so the arm is first sent here in joint space. From this seed the IK converges.
        self.declare_parameter('ready_joints', [0.724, 0.69, -1.208, 1.49, 0.743, 0.527, 1.382])
        # End-effector orientation (roll, pitch, yaw): pitch = pi/2 -> pinch_site points
        # down; yaw rotates the fingers around the vertical so they open almost
        # perpendicular to the cube motion and it can pass between them.
        self.declare_parameter('grasp_rpy', [0.0, np.pi / 2, np.radians(-150.0)])
        # Region of the trajectory where the arm can grasp (validated offline with the IK)
        self.declare_parameter('intercept_x_range', [-0.45, -0.30])
        self.declare_parameter('intercept_y_range', [-0.10, 0.15])
        self.declare_parameter('pregrasp_z', 0.16)       # gripper above the conveyor
        self.declare_parameter('grasp_z_offset', -0.004)  # pinch_site relative to cube center
        self.declare_parameter('lift_z', 0.16)
        # Durations (simulated time, s)
        self.declare_parameter('t_ready', 2.5)
        self.declare_parameter('t_pregrasp', 1.5)
        self.declare_parameter('t_descend', 1.0)
        self.declare_parameter('t_close', 0.8)
        self.declare_parameter('time_margin', 0.5)
        # Start closing the gripper before the cube reaches the intercept point:
        # lead = close_lead + speed * close_lead_time (the fingers need ~0.1 s to close)
        self.declare_parameter('close_lead', 0.0)
        self.declare_parameter('close_lead_time', 0.1)

        p = lambda name: self.get_parameter(name).value
        self.ready_joints = np.asarray(p('ready_joints'), dtype=float)
        self.grasp_rpy = np.asarray(p('grasp_rpy'), dtype=float)
        self.x_range = p('intercept_x_range')
        self.y_range = p('intercept_y_range')
        self.pregrasp_z = p('pregrasp_z')
        self.grasp_z_offset = p('grasp_z_offset')
        self.lift_z = p('lift_z')
        self.t_ready = p('t_ready')
        self.t_pregrasp = p('t_pregrasp')
        self.t_descend = p('t_descend')
        self.t_close = p('t_close')
        self.time_margin = p('time_margin')
        self.close_lead = p('close_lead')
        self.close_lead_time = p('close_lead_time')

        # --- state ------------------------------------------------------------
        self.state = 'WAIT_CUBE'
        self.state_start = None
        self.cube_pos = None
        self.cube_vel = None
        self.intercept_point = None
        self.motion_dir = None
        self.grasped = False

        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # --- publishers -------------------------------------------------------
        self.target_publisher = self.create_publisher(Float64MultiArray, 'input_obj_position', 10)
        self.joint_publisher = self.create_publisher(Float64MultiArray, '/inverse_kinematics_result', 10)
        self.gripper_publisher = self.create_publisher(Float64, '/gripper_command', 10)

        # --- subscriptions ----------------------------------------------------
        self.create_subscription(Float64MultiArray, '/cube_state', self.cube_state_callback, 10)
        self.create_subscription(Bool, '/cube_grasped', self.cube_grasped_callback, latch_qos)

        # Uses /clock when use_sim_time is true
        self.timer = self.create_timer(0.02, self.step)  # 50 Hz
        self.get_logger().info('Intercept node started. Waiting for /cube_state...')

    # --- callbacks ------------------------------------------------------------
    def cube_state_callback(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.shape != (6,):
            return
        self.cube_pos = data[:3]
        self.cube_vel = data[3:]

    def cube_grasped_callback(self, msg):
        if msg.data and not self.grasped:
            self.grasped = True
            self.get_logger().info('mujoco_node confirms: cube grasped.')

    # --- helpers --------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def elapsed(self):
        return self.now() - self.state_start

    def set_state(self, state):
        self.state = state
        self.state_start = self.now()
        self.get_logger().info(f'State -> {state}')

    def send_pose(self, position):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in position] + self.grasp_rpy.tolist()
        self.target_publisher.publish(msg)
        self.get_logger().info(f'Target pose: {np.round(msg.data, 3).tolist()}')

    def send_gripper(self, value):
        self.gripper_publisher.publish(Float64(data=float(value)))

    def plan_intercept(self):
        """Earliest point of p(t) = p0 + v*t, inside the reachable region, that the
        cube reaches only after the arm has had time to get there."""
        speed = np.linalg.norm(self.cube_vel)
        if speed < 1e-6:
            return None, None
        # time the arm still needs: pregrasp + descend + margin
        t_needed = self.t_pregrasp + self.t_descend + self.time_margin
        for t in np.arange(t_needed, 60.0, 0.05):
            point = self.cube_pos + self.cube_vel * t
            if (self.x_range[0] <= point[0] <= self.x_range[1]
                    and self.y_range[0] <= point[1] <= self.y_range[1]):
                return point, t
        return None, None

    # --- state machine ----------------------------------------------------------
    def step(self):
        if self.state == 'WAIT_CUBE':
            if self.cube_pos is None:
                return
            # Joint-space move to the ready pose, gripper open
            self.send_gripper(GRIPPER_OPEN)
            msg = Float64MultiArray()
            msg.data = self.ready_joints.tolist() + [GRIPPER_OPEN]
            self.joint_publisher.publish(msg)
            self.set_state('READY')

        elif self.state == 'READY':
            if self.elapsed() < self.t_ready:
                return
            point, t_arrival = self.plan_intercept()
            if point is None:
                self.get_logger().error(
                    'The cube trajectory does not pass through the reachable region '
                    'in time. Nothing to intercept.')
                self.set_state('DONE')
                return
            self.intercept_point = point
            self.motion_dir = self.cube_vel / np.linalg.norm(self.cube_vel)
            self.get_logger().info(
                f'Intercept point {np.round(point, 3).tolist()}, '
                f'cube arrives in {t_arrival:.2f} s')
            self.send_pose([point[0], point[1], self.pregrasp_z])
            self.set_state('PREGRASP')

        elif self.state == 'PREGRASP':
            if self.elapsed() < self.t_pregrasp:
                return
            p = self.intercept_point
            self.send_pose([p[0], p[1], p[2] + self.grasp_z_offset])
            self.set_state('DESCEND')

        elif self.state == 'DESCEND':
            if self.elapsed() < self.t_descend:
                return
            self.set_state('WAIT_ARRIVAL')

        elif self.state == 'WAIT_ARRIVAL':
            # signed distance of the cube to the intercept point along its motion
            if self.cube_pos is None:
                return
            ahead = np.dot(self.cube_pos - self.intercept_point, self.motion_dir)
            lead = self.close_lead + np.linalg.norm(self.cube_vel) * self.close_lead_time
            if ahead >= -lead:
                self.get_logger().info(
                    f'Cube at the intercept point ({np.round(self.cube_pos, 3).tolist()}): closing gripper.')
                self.send_gripper(GRIPPER_CLOSED)
                self.set_state('CLOSE')

        elif self.state == 'CLOSE':
            if self.elapsed() < self.t_close:
                return
            if not self.grasped:
                self.get_logger().warn('Grasp not confirmed by mujoco_node; lifting anyway.')
            p = self.intercept_point
            self.send_pose([p[0], p[1], self.lift_z])
            self.set_state('LIFT')

        elif self.state == 'LIFT':
            if self.elapsed() < 2.0:
                return
            self.get_logger().info(
                f'Done. Cube at {np.round(self.cube_pos, 3).tolist()} '
                f'(grasped: {self.grasped}).')
            self.set_state('DONE')


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
