from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

"""Cube on the conveyor whose position follows the ultrasonic sensor (/distance).

    ros2 launch ros2_to_mujoco_nero cube_sensor.launch.py
    ros2 launch ros2_to_mujoco_nero cube_sensor.launch.py usb_port:=/dev/ttyACM0 baud_rate:=9600

Close any other serial monitor (e.g. PlatformIO) before launching.
"""


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0',
                              description='Serial port of the Arduino'),
        DeclareLaunchArgument('baud_rate', default_value='9600',
                              description='Must match Serial.begin() in the Arduino sketch'),

        Node(
            package='ros2_to_mujoco_nero',
            executable='ultrasonic_sensor_node',
            name='ultrasonic',
            output='screen',
            parameters=[{
                'usb_port': LaunchConfiguration('usb_port'),
                'baud_rate': ParameterValue(LaunchConfiguration('baud_rate'), value_type=int),
                'topic': 'distance',
            }],
        ),
        Node(
            package='ros2_to_mujoco_nero',
            executable='main_node',
            name='main_node',
            output='screen',
        ),
        Node(
            package='ros2_to_mujoco_nero',
            executable='mujoco_node',
            name='mujoco_node',
            output='screen',
            parameters=[{
                'cube_source': 'sensor',
                'distance_topic': '/distance',
                # Sensor at the start end of the conveyor, looking along +Y
                'sensor_position': [-0.22, -1.07, 0.084],
                'sensor_direction': [0.0, 1.0, 0.0],
            }],
        ),
    ])
