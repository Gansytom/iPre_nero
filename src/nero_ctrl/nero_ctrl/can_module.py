#!/usr/bin/env python3
import can
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool, Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, PoseArray

"""This node has to communicate with the CAN bus of agx_arm_ws """

class CanModule(Node):
    def __init__(self):
        super().__init__('can_module')
        self.get_logger().info("CAN module initialized.")

        # parameters
        self._joint_state = JointState()
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.gripper_name = "gripper"
        self.init_pose = [0.0, -0.40, 0.0, 2.0, 0.0, 0.0, 0.0, 0.05]
        self.init_pose_ready = False
        self.start_requested = False
        self.feedback_ready = False
        self.initial_pose_attempts = 0
        self.initial_pose_first_sent_at = None


        # QoS settings
        self.init_node_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        ik_qos = QoSProfile(depth=10, durability=DurabilityPolicy.VOLATILE)
        # timers

        # publishers
        # publishers of agx_arm_ctrl_single_node.py
        self.init_state_joint_pub = self.create_publisher(JointState, 'control/move_j', self.init_node_qos)
        self.move_j_pub = self.create_publisher(JointState, "control/move_j", 10)
        self.move_p_pub = self.create_publisher(PoseStamped, "control/move_p", 10)
        self.move_l_pub = self.create_publisher(PoseStamped, "control/move_l", 10)
        self.move_c_pub = self.create_publisher(PoseArray, "control/move_c", 10)
        self.move_js_pub = self.create_publisher(JointState, "control/move_js", 10)
        # agx_arm_ctrl ignora 'gripper' dentro de move_j: la pinza solo se controla por
        # control/joint_states (el puente de MuJoCo acepta ambos).
        self.direct_joint_pub = self.create_publisher(JointState, "control/joint_states", 10)
        
        # others publishers (can modify and use this)
        self.start_node_publisher = self.create_publisher(Bool, '/start_running_nodes', self.init_node_qos)
        
        self.joint_state_publisher_for_ik = self.create_publisher(
            JointState,
            '/ik/joint_states',
            ik_qos,
        )
        
        # subscribers
        self.create_subscription(Float64MultiArray, 'test_pos', self.test_pos_callback, 10)
        self.create_subscription(Bool, 'start_running_nodes', self.start_node_callback, 10)
        self.create_subscription(JointState, 'feedback/joint_states', self.joint_states_callback, ik_qos)
        self.create_subscription(JointState, 'ik/ik_result', self.ik_result_callback, ik_qos)
        # objetivo articular directo (sin IK), p. ej. la pose "ready" de intercept_node
        self.create_subscription(JointState, 'joint_goal', self.joint_goal_callback, 10)
        # apertura de la pinza: ancho total en metros (0.0 cerrada .. 0.1 abierta)
        self.create_subscription(Float64, 'gripper_command', self.gripper_command_callback, 10)
        self.gripper_force = 1.0
        self.initial_pose_timer = self.create_timer(0.25, self._publish_initial_pose_when_ready)

        # send_initial_pose=False: another node (e.g. intercept_node) moves the arm to its
        # own start pose; can_module only forwards commands and feedback.
        # forward_ik_gripper=True: the IK result (incl. the gripper value of
        # input_obj_position) goes to move_j as is. False: arm joints only, the gripper is
        # driven exclusively by gripper_command (used by intercept_node, so an IK result
        # does not reopen the gripper while holding the cube).
        self.declare_parameter('forward_ik_gripper', True)
        self.forward_ik_gripper = bool(self.get_parameter('forward_ik_gripper').value)
        self.declare_parameter('send_initial_pose', True)
        if not self.get_parameter('send_initial_pose').value:
            self.initial_pose_timer.cancel()
            self.init_pose_ready = True
            self.get_logger().info('Initial pose disabled (send_initial_pose=false).')


    def test_pos_callback(self, msg):
        data = msg.data
        if data and data[0] == 1.0:
            self.get_logger().info("Test position 1 received.")
            self._joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]
            self._joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        else:
            self.get_logger().info("Test position 2 received.")
            self._joint_state.name = ["joint1", "joint2", "joint3","joint4", "joint5", "joint6", "joint7", "gripper"]
            self._joint_state.position = [0.5, 0.5, 0.5, 0.5, 10.0, 0.5, 0.5, 0.05]
            effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        msg = JointState()
        msg.name = self._joint_state.name
        msg.position = self._joint_state.position
        msg.effort = effort
        self.move_j_pub.publish(msg)
        self.get_logger().info(f"Published joint states: {msg.position}")

    def start_node_callback(self, msg):
        if msg.data and not self.start_requested:
            self.start_requested = True
            self.get_logger().info('Starting CAN module.')
            self.start_node_publisher.publish(Bool(data=True))

    def _publish_initial_pose_when_ready(self):
        if not self.start_requested or not self.feedback_ready:
            return

        now = time.monotonic()
        if self.initial_pose_first_sent_at is None:
            self.initial_pose_first_sent_at = now
            return

        if now - self.initial_pose_first_sent_at < 1.0:
            return
        if self.initial_pose_attempts >= 3:
            self.initial_pose_timer.cancel()
            return
        if self.initial_pose_attempts and now - self.initial_pose_first_sent_at < 0.5 * (self.initial_pose_attempts + 1):
            return

        msg = JointState()
        msg.name = self.joint_names + [self.gripper_name]
        msg.position = self.init_pose
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.init_state_joint_pub.publish(msg)
        self.initial_pose_attempts += 1
        self.init_pose_ready = True
        self.get_logger().info(
            f"Published initial joint states (attempt {self.initial_pose_attempts}/3): {msg.position}"
        )

    
    def joint_states_callback(self, msg):
        if len(msg.position) >= len(self.joint_names):
            self.feedback_ready = True
        if self.init_pose_ready:
            self._joint_state.name = msg.name
            self._joint_state.position = msg.position
            self.joint_state_publisher_for_ik.publish(self._joint_state)

    def _arm_only(self, msg):
        """Copia de msg solo con joint1..joint7: la pinza va por su propio canal, asi un
        resultado de IK no le cambia la apertura (p. ej. soltando el cubo agarrado)."""
        out = JointState()
        for i, name in enumerate(msg.name):
            if name in self.joint_names and i < len(msg.position):
                out.name.append(name)
                out.position.append(msg.position[i])
        return out

    def ik_result_callback(self, msg):
        self.get_logger().info(f"Received IK result: {msg.position}")
        # initially only with movej
        if self.forward_ik_gripper:
            self.move_j_pub.publish(msg)
            return
        arm = self._arm_only(msg)
        if len(arm.name) == len(self.joint_names):
            self.move_j_pub.publish(arm)

    def joint_goal_callback(self, msg):
        arm = self._arm_only(msg)
        if len(arm.name) != len(self.joint_names):
            self.get_logger().error(f'joint_goal needs {self.joint_names}, got {list(msg.name)}')
            return
        self.get_logger().info(f"Joint goal: {list(arm.position)}")
        self.move_j_pub.publish(arm)

    def gripper_command_callback(self, msg):
        width = float(min(max(msg.data, 0.0), 0.1))
        out = JointState()
        out.name = [self.gripper_name]
        out.position = [width]
        out.effort = [self.gripper_force]
        self.direct_joint_pub.publish(out)
        self.get_logger().info(f"Gripper command: width {width:.3f} m")

def main(args=None):
    rclpy.init(args=args)
    can_module = CanModule()
    rclpy.spin(can_module)
    can_module.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
