#!/usr/bin/env python3
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import traceback
import json
import torch
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import asyncio
from ament_index_python.packages import get_package_share_directory
from tf2_ros import TransformBroadcaster

from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from sensor_msgs.msg import NavSatFix

from lyla_controller import LyLA_forROS as LyAT

mavros_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10)

class LyapunovAdaptiveTransformer(Node):

    def __init__(self):
        super().__init__('lyapunov_adaptive_transformer')
        self.initialize_lyat()
        self.init_states()
        self.init_topics()
        self.init_clients()
        self.get_logger().info("LyLA Node Initialized")

    def init_states(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.armed = False
        self.offboard_mode = False
        self.actual_traj = []
        self.desired_traj = []
        self.tracking_errors = []
        self.time_log = []
        self.u_log = []
        self.theta_log = []
        self.HOME_X = 0.0
        self.HOME_Y = 0.0

    def init_topics(self):
        self.tf_broadcaster = TransformBroadcaster(self)
        self.setpoint_pub = self.create_publisher(
            PositionTarget, 'setpoint_raw/local', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, 'local_position/pose',
            self.pose_callback, mavros_qos)
        self.vel_sub = self.create_subscription(
            TwistStamped, 'local_position/velocity_local',
            self.velocity_callback, mavros_qos)
        self.state_sub = self.create_subscription(
            State, 'state',
            self.state_callback, mavros_qos)

    def init_clients(self):
        self.arming_client = self.create_client(CommandBool, 'cmd/arming')
        while not self.arming_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for arming service...')
        self.set_mode_client = self.create_client(SetMode, 'set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for set_mode service...')

    def initialize_lyat(self):
        config_path = os.path.join(
            get_package_share_directory('lyla_controller'),
            'config', 'config_LyLA.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.tf_end = self.config['T_final']
        self.dt = self.config['dt']
        self.VEL_MAX = self.config.get('VEL_MAX', 2.0)

    def pose_callback(self, msg):
        self.position[0] = msg.pose.position.x
        self.position[1] = msg.pose.position.y
        self.position[2] = msg.pose.position.z

    def velocity_callback(self, msg):
        self.velocity[0] = msg.twist.linear.x
        self.velocity[1] = msg.twist.linear.y
        self.velocity[2] = msg.twist.linear.z

    def state_callback(self, msg):
        self.armed = msg.armed
        self.offboard_mode = (msg.mode == "OFFBOARD")

    def send_velocity(self, vx, vy, vz):
        self.get_logger().info(f"Sending velocity command: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")

        vx, vy, vz = saturate_vector(vx, vy, vz, self.VEL_MAX)
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_PX |
            PositionTarget.IGNORE_PY |
            PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW)
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        self.setpoint_pub.publish(msg)

    async def set_offboard_and_arm(self):
        self.get_logger().info("Streaming setpoints to enable OFFBOARD mode...")
        for i in range(200):
            self.send_velocity(0.0, 0.0, 0.0)
            await self.sleep(0.02)
        self.get_logger().info("Setting OFFBOARD mode...")
        last_request = self.get_clock().now()
        while not self.offboard_mode and rclpy.ok():
            now = self.get_clock().now()
            if (now - last_request).nanoseconds > 2e9:
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                future = self.set_mode_client.call_async(req)
                await self.spin_until_future_complete(future)
                if future.result() and future.result().mode_sent:
                    self.get_logger().info("OFFBOARD mode set!")
                last_request = self.get_clock().now()
            self.send_velocity(0.0, 0.0, 0.0)
            await self.sleep(0.05)
        self.get_logger().info("Arming...")
        last_request = self.get_clock().now()
        while not self.armed and rclpy.ok():
            now = self.get_clock().now()
            if (now - last_request).nanoseconds > 2e9:
                req = CommandBool.Request()
                req.value = True
                future = self.arming_client.call_async(req)
                await self.spin_until_future_complete(future)
                if future.result() and future.result().success:
                    self.get_logger().info("Armed!")
                last_request = self.get_clock().now()
            self.send_velocity(0.0, 0.0, 0.0)
            await self.sleep(0.05)

    async def takeoff(self, target_height=2.5):
        self.get_logger().info(f"Taking off to {target_height}m...")
        while self.position[2] < target_height * 0.95 and rclpy.ok():
            self.send_velocity(0.0, 0.0, 0.8)
            await self.sleep(0.05)
        for i in range(20):
            self.send_velocity(0.0, 0.0, 0.0)
            await self.sleep(0.05)
        self.get_logger().info(f"At height {self.position[2]:.2f}m - starting trajectory")

    async def run_trajectory(self):
        self.get_logger().info("Starting LyLA trajectory tracking...")
        controller = LyAT.LbDNN_Controller(self.config)
        traj_start_time = self.get_clock().now()

        while rclpy.ok():
            t = (self.get_clock().now() - traj_start_time).nanoseconds / 1e9
            if t > self.tf_end:
                self.get_logger().info("Trajectory complete!")
                break

            x = torch.tensor([
                self.position[0], self.position[1], self.position[2],
                self.velocity[0], self.velocity[1], self.velocity[2]
            ], dtype=torch.float32)

            t_tensor = torch.tensor(t, dtype=torch.float32)
            xd, xd_dot = LyAT.Dynamics.desired_trajectory(t_tensor)
            u, Phi = controller.parameter_adaptation(x, t_tensor)

            e = x - xd
            err = torch.norm(e).item()
            self.actual_traj.append(self.position.copy())
            self.desired_traj.append(xd[:3].detach().numpy().copy())
            self.tracking_errors.append(err)
            self.time_log.append(t)
            self.u_log.append(u.detach().numpy().copy())
            theta = torch.cat([p.view(-1) for p in controller.nn.parameters()])
            self.theta_log.append(theta.detach().numpy().copy())

            self.get_logger().info(
                f"t={t:.2f}s | "
                f"Pos=[{self.position[0]:.2f},{self.position[1]:.2f},{self.position[2]:.2f}] | "
                f"Des=[{xd[0].item():.2f},{xd[1].item():.2f},{xd[2].item():.2f}] | "
                f"Err={err:.3f}m")

            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "autonomy_park"
            tf.child_frame_id = "target_position"
            tf.transform.translation.x = xd[0].item()
            tf.transform.translation.y = xd[1].item()
            tf.transform.translation.z = xd[2].item()
            tf.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(tf)

            self.send_velocity(float(u[0]), float(u[1]), float(u[2]))
            await self.sleep(self.dt)

    async def return_home(self):
        self.get_logger().info("Returning home...")
        for _ in range(200):
            ex = self.HOME_X - self.position[0]
            ey = self.HOME_Y - self.position[1]
            if math.sqrt(ex**2 + ey**2) < 0.3:
                break
            k = 0.4
            self.send_velocity(k*ex, k*ey, 0.0)
            await self.sleep(0.05)

    async def land(self):
        self.get_logger().info("Landing...")
        req = SetMode.Request()
        req.custom_mode = "AUTO.LAND"
        future = self.set_mode_client.call_async(req)
        await self.spin_until_future_complete(future)
        while self.armed and rclpy.ok():
            await self.sleep(0.5)
        self.get_logger().info("Landed!")

    def plot_results(self):
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt

            actual = np.array(self.actual_traj)
            desired = np.array(self.desired_traj)
            errors = np.array(self.tracking_errors)
            times = np.array(self.time_log)
            u_data = np.array(self.u_log)
            theta_data = np.array(self.theta_log)

            fig = plt.figure(figsize=(20, 12))

            # 3D trajectory with equal axis scaling
            ax1 = fig.add_subplot(231, projection='3d')
            ax1.plot(desired[:,0], desired[:,1], desired[:,2],
                    'b--', linewidth=2, label='Desired')
            ax1.plot(actual[:,0], actual[:,1], actual[:,2],
                    'r-', linewidth=2, label='Actual')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory Tracking')
            ax1.legend()
            # Force equal scaling on all 3 axes
            all_x = np.concatenate([desired[:,0], actual[:,0]])
            all_y = np.concatenate([desired[:,1], actual[:,1]])
            all_z = np.concatenate([desired[:,2], actual[:,2]])
            max_range = max(
                all_x.max() - all_x.min(),
                all_y.max() - all_y.min(),
                all_z.max() - all_z.min()) / 2
            x_mid = (all_x.max() + all_x.min()) / 2
            y_mid = (all_y.max() + all_y.min()) / 2
            z_mid = (all_z.max() + all_z.min()) / 2
            ax1.set_xlim(x_mid - max_range, x_mid + max_range)
            ax1.set_ylim(y_mid - max_range, y_mid + max_range)
            ax1.set_zlim(z_mid - max_range, z_mid + max_range)

            # Tracking error
            ax2 = fig.add_subplot(232)
            ax2.plot(times, errors, 'g-', linewidth=2)
            rms = np.sqrt(np.mean(errors**2))
            ax2.axhline(y=rms, color='r', linestyle='--',
                       label=f'RMS = {rms:.4f} m')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('||e|| (m)')
            ax2.set_title('Tracking Error vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Control inputs
            ax3 = fig.add_subplot(233)
            if u_data.shape[0] > 0:
                ax3.plot(times, u_data[:,0], 'r-', linewidth=1.5, label='u_x')
                ax3.plot(times, u_data[:,1], 'g-', linewidth=1.5, label='u_y')
                ax3.plot(times, u_data[:,2], 'b-', linewidth=1.5, label='u_z')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('u (m/s)')
            ax3.set_title('Control Inputs vs Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Control input norm
            ax4 = fig.add_subplot(234)
            if u_data.shape[0] > 0:
                u_norm = np.linalg.norm(u_data, axis=1)
                ax4.plot(times, u_norm, 'm-', linewidth=2)
                rms_u = np.sqrt(np.mean(u_norm**2))
                ax4.axhline(y=rms_u, color='r', linestyle='--',
                           label=f'RMS = {rms_u:.4f}')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('||u||')
            ax4.set_title('Control Input Norm vs Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # All 906 weights
            ax5 = fig.add_subplot(235)
            for i in range(theta_data.shape[1]):
                ax5.plot(times, theta_data[:, i],
                        linewidth=0.3, alpha=0.3)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Weight value')
            ax5.set_title('All 906 Network Weights vs Time')
            ax5.grid(True, alpha=0.2)

            # XY top view
            ax6 = fig.add_subplot(236)
            ax6.plot(desired[:,0], desired[:,1],
                    'b--', linewidth=2, label='Desired')
            ax6.plot(actual[:,0], actual[:,1],
                    'r-', linewidth=2, label='Actual')
            ax6.set_xlabel('X (m)')
            ax6.set_ylabel('Y (m)')
            ax6.set_title('XY Top View')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_aspect('equal')

            plt.tight_layout()
            plt.savefig('/tmp/lyla_results.png', dpi=150)
            self.get_logger().info("Plot saved to /tmp/lyla_results.png")
            plt.show()

        except Exception as e:
            self.get_logger().error(f"Plotting error: {e}")

    async def sleep(self, seconds):
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.005)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > seconds:
                break

    async def spin_until_future_complete(self, future):
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        return future.result()

    async def run_mission(self):
        try:
            await self.set_offboard_and_arm()
            await self.takeoff(target_height=2.5)
            await self.return_home()
            await self.run_trajectory()
            await self.return_home()
        except Exception as e:
            self.get_logger().error(f"Mission error: {e}")
            self.get_logger().error(traceback.format_exc())
        finally:
            await self.land()
            if len(self.actual_traj) > 10:
                self.plot_results()


def saturate_vector(vx, vy, vz, max_mag):
    mag = math.sqrt(vx**2 + vy**2 + vz**2)
    if mag > max_mag and mag > 0:
        s = max_mag / mag
        return vx*s, vy*s, vz*s
    return vx, vy, vz


def main(args=None):
    rclpy.init(args=args)
    node = LyapunovAdaptiveTransformer()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(node.run_mission())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        loop.close()


if __name__ == '__main__':
    main()
