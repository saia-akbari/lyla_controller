from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # PX4 + Gazebo
        ExecuteProcess(
            cmd=['bash', '-c',
                'cd ~/LyLA_ws/src/PX4-Autopilot && '
                'export PX4_GZ_MODEL_POSE="0,0,0,0,0,0" && '
                'make px4_sitl gz_x500'],
            output='screen',
            name='px4_gazebo'
        ),

        # MAVROS - wait 10s for PX4 to start
        TimerAction(period=10.0, actions=[
            ExecuteProcess(
                cmd=['bash', '-c',
                    'source /opt/ros/jazzy/setup.bash && '
                    'ros2 launch mavros px4.launch '
                    'fcu_url:=udp://:14540@127.0.0.1:14557'],
                output='screen',
                name='mavros'
            )
        ]),

        # RViz - wait 12s
        TimerAction(period=12.0, actions=[
            ExecuteProcess(
                cmd=['bash', '-c',
                    'source /opt/ros/jazzy/setup.bash && '
                    'source ~/LyLA_ws/install/setup.bash && '
                    'ros2 run rviz2 rviz2 '
                    '-d ~/LyLA_ws/src/lyla_controller/rviz/lyla.rviz'],
                output='screen',
                name='rviz'
            )
        ]),

        # LyLA Viz - wait 13s
        TimerAction(period=13.0, actions=[
            Node(
                package='lyla_controller',
                executable='lyla_viz',
                name='lyla_viz',
                output='screen'
            )
        ]),

        # LyLA Controller - wait 15s
        TimerAction(period=15.0, actions=[
            Node(
                package='lyla_controller',
                executable='lyla_node',
                name='lyla_node',
                output='screen'
            )
        ]),
    ])
