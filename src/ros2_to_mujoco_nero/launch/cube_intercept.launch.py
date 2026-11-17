from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""Parts 2 and 3: cube moving at constant velocity, intercepted and picked up.

    ros2 launch ros2_to_mujoco_nero cube_intercept.launch.py
    ros2 launch ros2_to_mujoco_nero cube_intercept.launch.py cube_vy:=0.04 rqt_graph:=true
"""


def generate_launch_description():
    cube_vy = LaunchConfiguration('cube_vy')
    return LaunchDescription([
        DeclareLaunchArgument('cube_vy', default_value='0.05',
                              description='Cube speed along the conveyor (Y axis), m/s'),
        DeclareLaunchArgument('rqt_graph', default_value='false',
                              description='Open rqt_graph to show the node/topic graph'),

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
            parameters=[{'cube_velocity': [0.0, cube_vy, 0.0]}],
        ),
        Node(
            package='ros2_to_mujoco_nero',
            executable='inverse_kinematics_node',
            name='inverse_kinematics_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='ros2_to_mujoco_nero',
            executable='intercept_node',
            name='intercept_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='rqt_graph',
            executable='rqt_graph',
            name='rqt_graph',
            condition=IfCondition(LaunchConfiguration('rqt_graph')),
        ),
    ])
