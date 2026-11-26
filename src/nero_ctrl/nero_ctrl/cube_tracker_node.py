#!/usr/bin/env python3
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float64MultiArray

"""Estimates the cube position and velocity from the ultrasonic sensor.

The sensor sits at one end of the conveyor looking along it, so each reading is
the distance to the face of the cube that faces the sensor:

    cube_center = sensor_position + sensor_direction * (range + cube_half_size)

The velocity is the slope of a least-squares line fitted to the positions along
the conveyor in a sliding time window (robust to the sensor noise, unlike the
difference of two consecutive readings).

Works the same with the real Arduino (ultrasonic_sensor_node) and with the
simulated sensor of the MuJoCo bridge (mujoco_viewer.py --sensor-pose ...): both
publish sensor_msgs/Range on 'distance'.

Publishes /cube_state = [x, y, z, vx, vy, vz] in the robot base frame, only while
a cube is detected and the estimate is reliable.
"""


class CubeTrackerNode(Node):
    def __init__(self):
        super().__init__('cube_tracker_node')

        self.declare_parameter('distance_topic', 'distance')
        # Sensor pose in the robot base frame (base_link)
        self.declare_parameter('sensor_position', [-0.38, -0.40, 0.084])
        self.declare_parameter('sensor_direction', [0.0, 1.0, 0.0])
        self.declare_parameter('cube_half_size', 0.02)
        # Readings beyond this distance mean "nothing on the conveyor" (conveyor end)
        self.declare_parameter('max_valid_range', 0.75)
        self.declare_parameter('min_valid_range', 0.03)
        self.declare_parameter('window', 0.6)        # s, velocity fitting window
        self.declare_parameter('min_samples', 6)
        # Max jump between consecutive readings (m); larger means a glitch or a new cube
        self.declare_parameter('max_jump', 0.08)

        p = lambda name: self.get_parameter(name).value
        self.sensor_position = np.asarray(p('sensor_position'), dtype=float)
        direction = np.asarray(p('sensor_direction'), dtype=float)
        self.sensor_direction = direction / np.linalg.norm(direction)
        self.cube_half_size = float(p('cube_half_size'))
        self.max_valid = float(p('max_valid_range'))
        self.min_valid = float(p('min_valid_range'))
        self.window = float(p('window'))
        self.min_samples = int(p('min_samples'))
        self.max_jump = float(p('max_jump'))

        self.samples = deque()   # (t, s) with s = distance of the cube center to the sensor
        self.detected = False

        self.publisher = self.create_publisher(Float64MultiArray, '/cube_state', 10)
        self.create_subscription(Range, p('distance_topic'), self.range_callback, 10)
        self.get_logger().info(
            f'Cube tracker: sensor at {self.sensor_position.tolist()} looking '
            f'{self.sensor_direction.tolist()}, listening on "{p("distance_topic")}"')

    def range_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        r = float(msg.range)

        if not (self.min_valid <= r <= self.max_valid):
            if self.detected:
                self.get_logger().info('Cube no longer detected.')
            self.samples.clear()
            self.detected = False
            return

        s = r + self.cube_half_size
        if self.samples and abs(s - self.samples[-1][1]) > self.max_jump:
            # discontinuity: glitch or a different object -> start over
            self.samples.clear()
        self.samples.append((t, s))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()
        if len(self.samples) < self.min_samples:
            return

        ts = np.array([x[0] for x in self.samples])
        ss = np.array([x[1] for x in self.samples])
        ts = ts - ts[-1]
        # least squares s(t) = s_now + speed * t  (t = 0 at the newest sample)
        A = np.vstack((np.ones_like(ts), ts)).T
        (s_now, speed), *_ = np.linalg.lstsq(A, ss, rcond=None)

        if not self.detected:
            self.detected = True
            self.get_logger().info(f'Cube detected at {s_now:.3f} m, speed {speed:.3f} m/s')

        pos = self.sensor_position + self.sensor_direction * s_now
        vel = self.sensor_direction * speed
        out = Float64MultiArray()
        out.data = np.concatenate((pos, vel)).tolist()
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CubeTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
