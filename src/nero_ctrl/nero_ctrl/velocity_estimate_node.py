#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float64
from rclpy.time import Time

class VelocityEstimateNode(Node):

    def __init__(self):
        super().__init__('velocity_estimate')
        self.declare_parameter('distance_topic', 'distance')
        self.declare_parameter('output_topic', 'vel_est')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('acceleration_bound', 0.5)
        distance_topic = self.get_parameter('distance_topic').value
        output_topic = self.get_parameter('output_topic').value
        publish_rate = float(self.get_parameter('publish_rate').value)

        self.acc_bound = self.get_parameter('acceleration_bound').value
        self.publisher_ = self.create_publisher(Float64, output_topic, 10)
        self.subscription_ = self.create_subscription(
            Range, distance_topic, self.distance_callback, 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        self.latest_distance = 0.0
        self.latest_stamp = 0.0
        self.prev_distance = 0.0
        self.prev_stamp = 0.0
        self.velocity_estimate = 0.0
        self.new_sample = False

        self.get_logger().info(
            f'Subscribed to "{distance_topic}", publishing on "{output_topic}" at {publish_rate} Hz')

    def distance_callback(self, msg: Range):
        self.latest_distance = msg.range
        self.latest_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.new_sample = True

    def estimate_velocity(self):
        if not self.new_sample:
            return self.velocity_estimate
        self.new_sample = False

        dt = self.latest_stamp - self.prev_stamp
        raw_vel = (self.latest_distance - self.prev_distance) / (dt + 1e-8)

        # Limit how much the estimate can change in this step: at most acc_bound * dt.
        max_step = self.acc_bound * dt
        vel_est = min(max(raw_vel, self.velocity_estimate - max_step),
                      self.velocity_estimate + max_step)

        self.prev_stamp = self.latest_stamp
        self.prev_distance = self.latest_distance
        self.velocity_estimate = vel_est
        return self.velocity_estimate

    def timer_callback(self):
        msg = Float64()
        msg.data = float(self.estimate_velocity())
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityEstimateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
