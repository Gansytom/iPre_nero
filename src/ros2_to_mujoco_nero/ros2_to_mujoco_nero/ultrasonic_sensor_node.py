import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class Ultrasonic(Node):
    """Read distance (cm, one value per line) from an Arduino over serial
    and publish it as sensor_msgs/Range (meters)."""

    def __init__(self):
        super().__init__('ultrasonic')
        self.declare_parameter('topic', 'distance')
        self.declare_parameter('usb_port', '/dev/ttyUSB1')
        self.declare_parameter('baud_rate', 57600)
        self.declare_parameter('frame_id', 'ultrasonic_link')
        self.declare_parameter('min_range', 0.02)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('field_of_view', 0.26)

        topic = self.get_parameter('topic').value
        port = self.get_parameter('usb_port').value
        baud = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.fov = float(self.get_parameter('field_of_view').value)

        self.publisher_ = self.create_publisher(Range, topic, 10)

        try:
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.05)
        except serial.SerialException as e:
            self.get_logger().fatal(f'Cannot open serial port {port}: {e}')
            raise

        self.ser.reset_input_buffer()
        self.get_logger().info(f'Reading {port} @ {baud} baud, publishing on "{topic}"')

        self.timer = self.create_timer(0.02, self.ultrasonic_callback)

    def serial_rx(self):
        """Read all pending lines and return the latest valid value in cm, or None."""
        latest = None
        while self.ser.in_waiting:
            line = self.ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            try:
                latest = float(line)
            except ValueError:
                self.get_logger().debug(f'Ignoring malformed line: {line!r}')
        return latest

    def ultrasonic_callback(self):
        try:
            distance_cm = self.serial_rx()
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}', throttle_duration_sec=1.0)
            return

        if distance_cm is None:
            return

        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = self.fov
        msg.min_range = self.min_range
        msg.max_range = self.max_range
        msg.range = distance_cm / 100.0

        self.publisher_.publish(msg)
        self.get_logger().info(f'Distance: {distance_cm:.1f} cm', throttle_duration_sec=1.0)

    def destroy_node(self):
        ser = getattr(self, 'ser', None)
        if ser is not None and ser.is_open:
            ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Ultrasonic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
