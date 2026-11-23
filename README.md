# iPre_nero


## Quick Start

### 1. Install Python SDK

```bash
git clone https://github.com/agilexrobotics/pyAgxArm.git
cd pyAgxArm
```

Choose the installation command based on your ROS version:

**Jazzy** installation command:

```bash
pip3 install . --break-system-packages
```

**Humble** installation command:

```bash
pip3 install .
```

### 2. Install Dependencies

Run the script to install all dependencies at once:

```bash
cd ~/iPre_nero/src/agx_arm_ros/scripts/
chmod +x agx_arm_install_deps.sh
bash ./agx_arm_install_deps.sh
```

Or install manually by executing the following commands in order:

1. Python dependencies

    Choose the installation command based on your ROS version:

    **Jazzy** installation command:

    ```bash
    pip3 install python-can scipy numpy --break-system-packages
    ```

    **Humble** installation command:

    ```bash
    pip3 install python-can scipy numpy
    ```

    > If using `effector_type:=revo2_pro` or `revo2_touch`, you also need [bc-stark-sdk](https://pypi.org/project/bc-stark-sdk/).

2. CAN tools

    ```bash
    sudo apt update && sudo apt install can-utils ethtool
    ```

3. ROS2 dependencies

    ```bash
    sudo apt install -y \
        ros-$ROS_DISTRO-ros2-control \
        ros-$ROS_DISTRO-ros2-controllers \
        ros-$ROS_DISTRO-controller-manager \
        ros-$ROS_DISTRO-topic-tools \
        ros-$ROS_DISTRO-joint-state-publisher-gui \
        ros-$ROS_DISTRO-robot-state-publisher \
        ros-$ROS_DISTRO-xacro \
        python3-colcon-common-extensions
    ```

4. MoveIt

    Before using MoveIt, you need to configure the related dependencies. For detailed steps, please refer to: [agx_arm_moveit](./src/agx_arm_moveit/README_EN.md)
    
    Or execute the following commands in order:

    ```bash
    sudo apt install ros-$ROS_DISTRO-moveit*
    ```

    ```bash
    sudo apt-get install -y \
        ros-$ROS_DISTRO-control* \
        ros-$ROS_DISTRO-joint-trajectory-controller \
        ros-$ROS_DISTRO-joint-state-* \
        ros-$ROS_DISTRO-gripper-controllers \
        ros-$ROS_DISTRO-trajectory-msgs
    ```

    If the system locale is not set to English, it must be set to English locale:

    ```bash
    echo "export LC_NUMERIC=en_US.UTF-8" >> ~/.bashrc
    source ~/.bashrc
    ```

### 3. Build and Source Workspace

Check if you are in a virtual environment. If so, it is recommended to exit the virtual environment first.

```bash
which pip3
```

Build and Source the workspace:

```bash
cd ~/iPre_nero
colcon build
source install/setup.bash
```

## Simulator
### 1. Source workspace
```bash
source /opt/ros/jazzy/setup.bash
source /opt/ros/humble/setup.bash
source ~/iPre_nero/install/setup.bash
```
### 2. Launch nero_can_ctrl.launch.py

```bash
ros2 launch nero_ctrl nero_can_ctrl.launch.py mode:=sim
```
Without rviz
```bash
ros2 launch nero_ctrl nero_can_ctrl.launch.py mode:=sim rviz:=false
```

## Hardware

This section describes how to build and use the `nero_ctrl` package with the physical NERO robot. It assumes that the CAN adapter is connected and that the robot is mechanically secure and clear of people or obstacles.

### 1. Build and source the package

From the workspace directory:

```bash
cd /home/ubuntu/Projects/iPre_nero
colcon build --packages-select nero_ctrl
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/Projects/iPre_nero/install/setup.bash
```

If the workspace was built by sourcing Humble instead, use `/opt/ros/humble/setup.bash` and then source the workspace. Build the complete workspace with `colcon build` if other packages were changed.

### 2. Activate the CAN interface

Before starting a hardware controller, activate the CAN interface:

```bash
cd /home/ubuntu/Projects/agx_arm_ws/src/agx_arm_ros/scripts
bash can_activate.sh
```

The default setup uses `can0` at `1 Mbit/s`. Verify that the interface is active and receiving traffic:

```bash
ip link show can0
candump -tz can0
```

Do not run calibration, control, and another CAN-consuming node concurrently on the same interface.

### 3. Available `nero_ctrl` launch files

Open a new terminal for each process and source ROS 2 and the workspace before launching anything.

#### `nero_can_ctrl.launch.py`

This is the standard hardware or simulator launch file. It starts:

- `main_node`, which receives high-level target positions and starts the control flow.
- `can_module`, which sends joint commands to the robot/simulator and forwards feedback to the IK node.
- The physical `agx_arm_ctrl` backend when `mode:=real`.
- The NERO MuJoCo backend when `mode:=sim`.

Hardware mode is the default:

```bash
ros2 launch nero_ctrl nero_can_ctrl.launch.py
```

To explicitly select the interface:

```bash
ros2 launch nero_ctrl nero_can_ctrl.launch.py mode:=real can_port:=can0
```

Simulator mode is also available from this launch file:

```bash
ros2 launch nero_ctrl nero_can_ctrl.launch.py mode:=sim
ros2 launch nero_ctrl nero_can_ctrl.launch.py mode:=sim rviz:=false
```

Do not start hardware and simulator mode at the same time: both consume `control/move_j` and publish `feedback/joint_states`.

The simulator arguments are:

| Argument | Default | Description |
|---|---|---|
| `mode` | `real` | Select `real` hardware or `sim` MuJoCo simulation. |
| `can_port` | `can0` | CAN interface used in hardware mode. |
| `scene` | `scene_nero_arm.xml` | MuJoCo scene used in simulator mode. |
| `rviz` | `true` | Start RViz2 in simulator mode. |
| `start_delay` | `2.0` | Delay before starting `main_node` and `can_module` in simulator mode. |

#### `nero_can_ctrl_ik.launch.py`

This launch file is intended for hardware with inverse kinematics. It activates `can0`, waits for CAN traffic, starts the AgileX arm controller, `main_node`, `can_module`, and `ik_node`, and sets the arm controller speed to `10%`.

```bash
ros2 launch nero_ctrl nero_can_ctrl_ik.launch.py
```

The CAN USB adapter address can be overridden when required:

```bash
ros2 launch nero_ctrl nero_can_ctrl_ik.launch.py can_usb_address:=<USB-BUS-ADDRESS>
```

The default adapter address is `3-3.4:1.0`. This launch file currently uses fixed hardware settings: `can0`, `1 Mbit/s`, `NERO`, and `agx_gripper`. Use `nero_can_ctrl.launch.py` when you need configurable launch arguments.

#### `nero_calibration.launch.py`

This launch file starts only the calibration node. It disables the selected joint or joints, allowing them to be moved to the mechanical zero position. It requires Nero firmware V111 or newer and an active CAN interface.

Stop any robot-control launch before calibration. The node uses `1000000` baud and `can0` by default.

### 4. Calibrate the robot joints

Build and source the package, then run the calibration node:

```bash
cd /home/ubuntu/Projects/iPre_nero
colcon build --packages-select nero_ctrl
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch nero_ctrl nero_calibration.launch.py \
  channel:=can0 \
  bitrate:=1000000 \
  firmware:=v111 \
  joint_index:=1
```

Supported arguments are:

| Argument | Default | Description |
|---|---|---|
| `channel` | `can0` | SocketCAN interface. |
| `bitrate` | `1000000` | CAN bitrate. |
| `firmware` | `v111` | Nero firmware: `v111`, `v112`, `v120`, or `v121`. |
| `joint_index` | `1` | Joint number `1` through `7`, or `255` for all joints. |

Move the selected joint to its mechanical zero position, then, in a second sourced terminal, call:

```bash
ros2 service call /calibrate_joint std_srvs/srv/Trigger "{}"
```

A successful response is similar to:

```text
success: true
message: Joint 1 calibrated and enabled.
```

Keep the calibration launch running until the service response arrives. Firmware V111 does not return an acknowledgement for the calibration command through every SDK path, so an enabled response should not be interpreted as the only verification: also inspect CAN traffic with `candump -tz can0`.

To calibrate all joints, use:

```bash
ros2 launch nero_ctrl nero_calibration.launch.py \
  channel:=can0 \
  bitrate:=1000000 \
  firmware:=v111 \
  joint_index:=255
```

Before calling the service for `joint_index:=255`, place all seven joints at their mechanical zero positions.

### 5. Enable or disable the robot

When using a launch that starts `agx_arm_ctrl`, enable the arm with:

```bash
ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}"
```

Disable it before ending an experiment or if motion must stop:

```bash
ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: false}"
```

### 6. Send joint targets

The standard joint command topic is `/control/move_j`. It accepts `sensor_msgs/msg/JointState` messages. The first seven names are arm joints and the optional eighth name is the gripper.

Move the gripper manually:

```bash
ros2 topic pub --once /control/move_j sensor_msgs/msg/JointState \
  "{name: [gripper], position: [0.05], effort: [1.0]}"
```

Typical gripper values are:

- Open: `0.10`
- Half open: `0.05`
- Closed: `0.00`
- Width range: `0.0` to `0.1` meters
- Force range: `0.5` to `3.0` N

Send a position to all joints and the gripper:

```bash
ros2 topic pub --once /control/move_j sensor_msgs/msg/JointState \
  "{name: [joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper], position: [0.0, -0.40, 0.0, 2.0, 0.0, 0.0, 0.0, 0.05], effort: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}"
```

Joint positions are expected to use the units and conventions implemented by `agx_arm_ctrl`. Start with low speeds and verify each motion in a safe workspace.

### 7. Send a Cartesian target with inverse kinematics

Start the hardware IK launch:

```bash
ros2 launch nero_ctrl nero_can_ctrl_ik.launch.py
```

Publish a seven-element `std_msgs/msg/Float64MultiArray` target to `/input_obj_position`:

```bash
ros2 topic pub --once /input_obj_position std_msgs/msg/Float64MultiArray \
  "{data: [-0.40, 0.0, 0.3, -95.0, 74.0, -179.0, 0.0]}"
```

The array format is:

```text
[x, y, z, Rx, Ry, Rz, gripper_position]
```

- `x`, `y`, `z`: target position in meters.
- `Rx`, `Ry`, `Rz`: target orientation in degrees. `main_node` converts them to radians before publishing `/obj_point`.
- `gripper_position`: gripper width in meters.

The target flow is:

```text
/input_obj_position
  -> main_node
  -> /obj_point
  -> ik_node
  -> /ik/ik_result
  -> can_module
  -> /control/move_j
  -> agx_arm_ctrl
```

The IK node uses the MuJoCo model loaded from `/model_path`. It publishes computed joint states to `/ik/ik_result`; `can_module` forwards these states to the real arm or to the selected simulator backend.

Useful runtime checks:

```bash
ros2 topic echo /feedback/joint_states
ros2 topic echo /ik/ik_result
ros2 topic list
ros2 node list
```

### 8. Safety and troubleshooting

- Activate CAN only when the robot area is clear and the E-stop is accessible.
- Keep a human operator ready to stop motion at all times.
- Never launch two nodes that control the same arm or CAN interface.
- If a CAN launch does not detect traffic, verify `can0`, adapter power, wiring, bitrate, and USB address.
- If `nero_can_ctrl_ik.launch.py` waits indefinitely, check that `candump -n 1 can0` receives frames and that no other process already owns the interface.
- If a `Float64MultiArray` target is ignored, verify that it contains exactly seven elements.
- For calibration failures, keep the calibration node running, confirm CAN traffic, and verify that the configured firmware matches the installed firmware.
