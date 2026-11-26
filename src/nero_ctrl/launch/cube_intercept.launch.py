"""
nero_ctrl — cube_intercept.launch.py
====================================
Cube moving on the conveyor: estimated with the ultrasonic sensor, intercepted and
picked up. Same nodes for the real arm and for the MuJoCo simulation:

  /distance -> cube_tracker_node -> /cube_state -> intercept_node
  intercept_node -> input_obj_position -> main_node -> ik_node -> can_module -> control/move_j
  intercept_node -> joint_goal / gripper_command -> can_module -> control/move_j / control/joint_states

  mode:=real  agx_arm_ctrl over CAN + ultrasonic_sensor_node (Arduino)
  mode:=sim   MuJoCo bridge (mujoco_viewer.py) with the cube and a SIMULATED ultrasonic
              sensor: no hardware needed.
  mode:=sim sensor:=real
              MuJoCo arm, but the REAL ultrasonic sensor (Arduino): the simulated cube
              follows the real readings, so you can move a real cube along the real
              conveyor and watch the simulated arm intercept it. cube_vy / cube_delay /
              sensor_noise are ignored.

Parameters: config/cube_intercept.yaml

Examples:
  ros2 launch nero_ctrl cube_intercept.launch.py mode:=sim
  ros2 launch nero_ctrl cube_intercept.launch.py mode:=sim cube_vy:=0.03 rviz:=true
  ros2 launch nero_ctrl cube_intercept.launch.py mode:=sim sensor:=real usb_port:=/dev/ttyACM0
  ros2 launch nero_ctrl cube_intercept.launch.py mode:=real usb_port:=/dev/ttyACM0
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_model_dir():
    """src/models_for_simulation/agilex_nero (scene with conveyor + cube), searched
    upwards from this file like main_node.find_scene_xml(); NERO_SCENE_XML overrides."""
    env = os.environ.get('NERO_SCENE_XML')
    if env and Path(env).is_file():
        return str(Path(env).parent)
    rel = Path('src/models_for_simulation/agilex_nero')
    for parent in Path(__file__).resolve().parents:
        if (parent / rel / 'scene.xml').is_file():
            return str(parent / rel)
    raise FileNotFoundError('models_for_simulation/agilex_nero/scene.xml not found. '
                            'Set NERO_SCENE_XML to its full path.')


def real_sensor_node(context, tracker):
    """Arduino ultrasonic sensor -> sensor_msgs/Range on the tracker's distance topic."""
    return Node(
        package='nero_ctrl', executable='ultrasonic_sensor_node',
        name='ultrasonic', output='screen',
        parameters=[{
            'usb_port': LaunchConfiguration('usb_port').perform(context),
            'baud_rate': int(LaunchConfiguration('baud_rate').perform(context)),
            'topic': tracker['distance_topic'],
        }],
    )


def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)
    config = os.path.join(get_package_share_directory('nero_ctrl'), 'config', 'cube_intercept.yaml')
    with open(config) as f:
        tracker = yaml.safe_load(f)['cube_tracker_node']['ros__parameters']

    nodes = [
        Node(package='nero_ctrl', executable='main_node', name='main_node', output='screen'),
        Node(package='nero_ctrl', executable='can_module', name='can_module', output='screen',
             parameters=[{'send_initial_pose': False, 'forward_ik_gripper': False}]),
        Node(package='nero_ctrl', executable='ik_node', name='inverse_kinematics_node',
             output='screen'),
        Node(package='nero_ctrl', executable='cube_tracker_node', name='cube_tracker_node',
             output='screen', parameters=[config]),
        Node(package='nero_ctrl', executable='intercept_node', name='intercept_node',
             output='screen', parameters=[config]),
    ]

    if mode == 'real':
        backend = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('agx_arm_ctrl'), 'launch',
                'start_single_agx_arm.launch.py')),
            launch_arguments={
                'can_port': LaunchConfiguration('can_port').perform(context),
                'arm_type': 'nero',
                'effector_type': 'agx_gripper',
                'tcp_offset': '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
                'control_enabled': 'true',
                'speed_percent': LaunchConfiguration('speed_percent').perform(context),
            }.items(),
        )
        return [backend, real_sensor_node(context, tracker), *nodes]

    # mode == 'sim': the bridge simulates the cube and, with sensor:=sim, the ultrasonic
    # sensor too, placed where cube_tracker_node expects it (same YAML).
    real_sensor = LaunchConfiguration('sensor').perform(context) == 'real'
    pose = list(tracker['sensor_position']) + list(tracker['sensor_direction'])
    viewer_args = f'--sensor-pose {" ".join(str(v) for v in pose)} '
    if real_sensor:
        # The cube follows the Arduino readings; beyond max_valid_range = no cube.
        viewer_args += (f'--cube-source sensor '
                        f'--sensor-topic {tracker["distance_topic"]} '
                        f'--sensor-max {tracker["max_valid_range"]}')
    else:
        vy = LaunchConfiguration('cube_vy').perform(context)
        viewer_args += (f'--cube-velocity 0 {vy} 0 '
                        f'--cube-delay {LaunchConfiguration("cube_delay").perform(context)} '
                        f'--sensor-noise {LaunchConfiguration("sensor_noise").perform(context)}')
    backend = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('agx_arm_description'), 'launch',
            'bringup_nero.launch.py')),
        launch_arguments={
            'scene': 'scene.xml',
            'model_dir': find_model_dir(),
            'viewer_args': viewer_args,
            'rviz': LaunchConfiguration('rviz').perform(context),
        }.items(),
    )
    # MuJoCo takes a while to load: start the control nodes once the bridge listens.
    delay = float(LaunchConfiguration('start_delay').perform(context))
    extra = [real_sensor_node(context, tracker)] if real_sensor else []
    return [backend, *extra, TimerAction(period=delay, actions=nodes)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', choices=['real', 'sim']),
        # real
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument('speed_percent', default_value='10',
                              description='Real arm speed (%). Start slow.'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0',
                              description='Arduino serial port (mode:=real or sensor:=real)'),
        DeclareLaunchArgument('baud_rate', default_value='9600',
                              description='Must match Serial.begin() in the sketch'),
        # sim
        DeclareLaunchArgument('sensor', default_value='sim', choices=['sim', 'real'],
                              description='mode:=sim only: simulated ultrasonic sensor, or '
                                          'the real Arduino driving the simulated cube'),
        DeclareLaunchArgument('cube_vy', default_value='0.05',
                              description='Cube speed along the conveyor, m/s (mode:=sim)'),
        DeclareLaunchArgument('cube_delay', default_value='10.0',
                              description='Seconds the cube waits before moving (mode:=sim)'),
        DeclareLaunchArgument('sensor_noise', default_value='0.003',
                              description='Simulated sensor noise, m (mode:=sim)'),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('start_delay', default_value='2.0'),
        OpaqueFunction(function=launch_setup),
    ])
