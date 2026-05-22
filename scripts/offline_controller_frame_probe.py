#!/usr/bin/env python3
"""
Offline controller-frame probe for bag replay.

This node does not command the drone. It subscribes to /ekf/pose and
/mavros/local_position/pose and publishes diagnostic candidate velocity mappings
so we can determine whether the old controller x/y mapping and cmd_vel frame
assumption are correct.

It intentionally bypasses iris_mng/state_machine/RC so bag replay does not need
operator radio states.
"""
import csv
import json
import math
import os

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
import tf.transformations as tft


def yaw_from_quat(qmsg):
    return tft.euler_from_quaternion([qmsg.x, qmsg.y, qmsg.z, qmsg.w])[2]


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate2(yaw, x, y):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def norm2(x, y):
    return math.hypot(x, y)


def unit_or_zero(x, y):
    n = norm2(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n


def dot_unit(ax, ay, bx, by):
    au = unit_or_zero(ax, ay)
    bu = unit_or_zero(bx, by)
    if au == (0.0, 0.0) or bu == (0.0, 0.0):
        return float('nan')
    return au[0] * bu[0] + au[1] * bu[1]


class OfflineControllerFrameProbe:
    def __init__(self):
        rospy.init_node('offline_controller_frame_probe', anonymous=True)

        self.pose_topic = rospy.get_param('~pose_topic', '/ekf/pose')
        self.local_pose_topic = rospy.get_param('~local_pose_topic', '/mavros/local_position/pose')
        self.yaw_target = float(rospy.get_param('~yaw_target_rad', math.pi / 2.0))
        self.setpoint_z = float(rospy.get_param('~setpoint_z', 1.0))
        self.csv_path = rospy.get_param('~csv_path', '')
        self.rate_hz = float(rospy.get_param('~rate_hz', 20.0))

        self.ekf_pose = None
        self.local_pose = None

        self.diag_pub = rospy.Publisher('/controller/offline_frame_probe', String, queue_size=10)
        self.current_raw_pub = rospy.Publisher('/controller/offline_cmd/current_mapping_raw', TwistStamped, queue_size=10)
        self.current_world_pub = rospy.Publisher('/controller/offline_cmd/current_mapping_after_cmd_vel', TwistStamped, queue_size=10)
        self.direct_world_pub = rospy.Publisher('/controller/offline_cmd/direct_landpad_world', TwistStamped, queue_size=10)
        self.body_raw_pub = rospy.Publisher('/controller/offline_cmd/body_from_landpad_raw', TwistStamped, queue_size=10)
        self.body_world_pub = rospy.Publisher('/controller/offline_cmd/body_from_landpad_after_cmd_vel', TwistStamped, queue_size=10)

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=10)
        rospy.Subscriber(self.local_pose_topic, PoseStamped, self.local_pose_cb, queue_size=10)

        self.csv_file = None
        self.csv_writer = None
        self.csv_fields = [
            'stamp', 'pose_x', 'pose_y', 'pose_z', 'pose_yaw_deg',
            'local_yaw_deg', 'target_yaw_deg', 'yaw_error_deg',
            'ideal_world_vx', 'ideal_world_vy',
            'current_raw_vx', 'current_raw_vy',
            'current_after_cmd_vel_vx', 'current_after_cmd_vel_vy',
            'body_from_landpad_raw_vx', 'body_from_landpad_raw_vy',
            'body_from_landpad_after_cmd_vel_vx', 'body_from_landpad_after_cmd_vel_vy',
            'cos_current_after_vs_ideal', 'cos_body_after_vs_ideal',
            'cos_current_raw_vs_ideal', 'cos_body_raw_vs_ideal'
        ]
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
            self.csv_writer.writeheader()

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.timer_cb)
        rospy.loginfo('[offline_controller_frame_probe] pose_topic=%s local_pose_topic=%s yaw_target=%.3f rad',
                      self.pose_topic, self.local_pose_topic, self.yaw_target)

    def pose_cb(self, msg):
        self.ekf_pose = msg

    def local_pose_cb(self, msg):
        self.local_pose = msg

    def make_twist(self, stamp, frame, vx, vy, vz, wz):
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = wz
        return msg

    def timer_cb(self, _event):
        if self.ekf_pose is None:
            return

        stamp = self.ekf_pose.header.stamp if not self.ekf_pose.header.stamp.is_zero() else rospy.Time.now()
        x = self.ekf_pose.pose.position.x
        y = self.ekf_pose.pose.position.y
        z = self.ekf_pose.pose.position.z
        yaw = yaw_from_quat(self.ekf_pose.pose.orientation)
        local_yaw = yaw
        if self.local_pose is not None:
            local_yaw = yaw_from_quat(self.local_pose.pose.orientation)

        # Controller setpoint is x=0, y=0, z=setpoint_z, yaw=yaw_target.
        # Positive Kp direction only; gains are irrelevant for frame/sign diagnosis.
        err_x = -x
        err_y = -y
        err_z = self.setpoint_z - z
        err_yaw = wrap_pi(self.yaw_target - yaw)

        # Ideal correction if landpad frame == MAVROS world/local frame.
        ideal_world_vx = err_x
        ideal_world_vy = err_y

        # Current C++ mapping after PID signs, approximated for proportional control:
        # vel.vx = err_x, vel.vy = err_y; output.x = -vel.vy, output.y = -vel.vx.
        current_raw_vx = -err_y
        current_raw_vy = -err_x

        # What DroneControl::cmd_vel does to raw body-frame x/y: rotate by local yaw to world.
        current_world_vx, current_world_vy = rotate2(local_yaw, current_raw_vx, current_raw_vy)

        # Correct raw body-frame command if we want DroneControl::cmd_vel to produce ideal world/landpad correction.
        body_raw_vx, body_raw_vy = rotate2(-local_yaw, ideal_world_vx, ideal_world_vy)
        body_world_vx, body_world_vy = rotate2(local_yaw, body_raw_vx, body_raw_vy)

        self.current_raw_pub.publish(self.make_twist(stamp, 'drone/body_assumed', current_raw_vx, current_raw_vy, err_z, err_yaw))
        self.current_world_pub.publish(self.make_twist(stamp, 'world_after_cmd_vel_model', current_world_vx, current_world_vy, err_z, err_yaw))
        self.direct_world_pub.publish(self.make_twist(stamp, 'landpad_or_world_direct', ideal_world_vx, ideal_world_vy, err_z, err_yaw))
        self.body_raw_pub.publish(self.make_twist(stamp, 'drone/body_assumed', body_raw_vx, body_raw_vy, err_z, err_yaw))
        self.body_world_pub.publish(self.make_twist(stamp, 'world_after_cmd_vel_model', body_world_vx, body_world_vy, err_z, err_yaw))

        row = {
            'stamp': stamp.to_sec(),
            'pose_x': x, 'pose_y': y, 'pose_z': z,
            'pose_yaw_deg': math.degrees(yaw),
            'local_yaw_deg': math.degrees(local_yaw),
            'target_yaw_deg': math.degrees(self.yaw_target),
            'yaw_error_deg': math.degrees(err_yaw),
            'ideal_world_vx': ideal_world_vx, 'ideal_world_vy': ideal_world_vy,
            'current_raw_vx': current_raw_vx, 'current_raw_vy': current_raw_vy,
            'current_after_cmd_vel_vx': current_world_vx, 'current_after_cmd_vel_vy': current_world_vy,
            'body_from_landpad_raw_vx': body_raw_vx, 'body_from_landpad_raw_vy': body_raw_vy,
            'body_from_landpad_after_cmd_vel_vx': body_world_vx, 'body_from_landpad_after_cmd_vel_vy': body_world_vy,
            'cos_current_after_vs_ideal': dot_unit(current_world_vx, current_world_vy, ideal_world_vx, ideal_world_vy),
            'cos_body_after_vs_ideal': dot_unit(body_world_vx, body_world_vy, ideal_world_vx, ideal_world_vy),
            'cos_current_raw_vs_ideal': dot_unit(current_raw_vx, current_raw_vy, ideal_world_vx, ideal_world_vy),
            'cos_body_raw_vs_ideal': dot_unit(body_raw_vx, body_raw_vy, ideal_world_vx, ideal_world_vy),
        }
        msg = String()
        msg.data = json.dumps(row, sort_keys=True)
        self.diag_pub.publish(msg)
        if self.csv_writer:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def spin(self):
        rospy.spin()
        if self.csv_file:
            self.csv_file.close()


if __name__ == '__main__':
    OfflineControllerFrameProbe().spin()
