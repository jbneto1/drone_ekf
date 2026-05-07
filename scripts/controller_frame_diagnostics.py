#!/usr/bin/env python3
"""
Controller/frame diagnostics for EKF precision landing flight tests.

This node is intentionally passive: it does not command the drone and does not
change controller behavior. It republishes the key pose/command/frame quantities
as JSON on /controller/frame_diagnostics and can optionally write a CSV file.

Use it while rosbagging to diagnose whether x/y errors are being interpreted as
landpad-frame, world-frame, or drone/body-frame commands.
"""

import csv
import json
import math
import os
from collections import OrderedDict

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import tf.transformations as tft


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    quat = [q.x, q.y, q.z, q.w]
    try:
        return tft.euler_from_quaternion(quat)[2]
    except Exception:
        return float('nan')


def pose_to_dict(msg):
    yaw = yaw_from_quat(msg.pose.orientation)
    return OrderedDict([
        ('stamp', msg.header.stamp.to_sec()),
        ('frame_id', msg.header.frame_id),
        ('x', msg.pose.position.x),
        ('y', msg.pose.position.y),
        ('z', msg.pose.position.z),
        ('yaw_rad', yaw),
        ('yaw_deg', math.degrees(yaw) if math.isfinite(yaw) else float('nan')),
    ])


def odom_to_pose_dict(msg):
    ps = PoseStamped()
    ps.header = msg.header
    ps.pose = msg.pose.pose
    return pose_to_dict(ps)


def twist_to_dict(msg):
    return OrderedDict([
        ('stamp', msg.header.stamp.to_sec()),
        ('frame_id', msg.header.frame_id),
        ('vx', msg.twist.linear.x),
        ('vy', msg.twist.linear.y),
        ('vz', msg.twist.linear.z),
        ('wz', msg.twist.angular.z),
    ])


def nearest_axis(yaw):
    """Return nearest landpad axis direction for drone +X/front."""
    if not math.isfinite(yaw):
        return 'unknown'
    axes = [
        ('landpad +X', 0.0),
        ('landpad +Y', math.pi / 2.0),
        ('landpad -X', math.pi),
        ('landpad -Y', -math.pi / 2.0),
    ]
    name, _ = min(axes, key=lambda item: abs(wrap_pi(yaw - item[1])))
    return name


def flatten(prefix, value, out):
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(prefix + '_' + k if prefix else k, v, out)
    else:
        out[prefix] = value


class LastMsg:
    def __init__(self):
        self.msg = None
        self.rx_time = None

    def update(self, msg):
        self.msg = msg
        self.rx_time = rospy.Time.now()

    def age(self, now):
        if self.rx_time is None:
            return float('inf')
        return (now - self.rx_time).to_sec()


class ControllerFrameDiagnostics:
    def __init__(self):
        rospy.init_node('controller_frame_diagnostics', anonymous=False)

        self.rate_hz = rospy.get_param('~rate_hz', 10.0)
        self.yaw_target = rospy.get_param('~yaw_target_rad', math.pi / 2.0)
        self.csv_path = rospy.get_param('~csv_path', '')
        self.max_age_warn = rospy.get_param('~max_age_warn', 0.5)

        self.topics = {
            'ekf_pose': rospy.get_param('~ekf_pose_topic', '/ekf/pose'),
            'ekf_odom': rospy.get_param('~ekf_odom_topic', '/ekf/odom'),
            'controller_pose': rospy.get_param('~controller_pose_topic', '/ekf/pose'),
            'raw_aruco_pose': rospy.get_param('~raw_aruco_pose_topic', '/aruco/pose'),
            'thermal_pose': rospy.get_param('~thermal_pose_topic', '/thermal/pose'),
            'local_pose': rospy.get_param('~local_pose_topic', '/mavros/local_position/pose'),
            'raw_controller_cmd': rospy.get_param('~raw_controller_cmd_topic', '/controller/raw_cmd_vel'),
            'mavros_cmd': rospy.get_param('~mavros_cmd_topic', '/mavros/setpoint_velocity/cmd_vel'),
        }

        self.last = {name: LastMsg() for name in self.topics}

        rospy.Subscriber(self.topics['ekf_pose'], PoseStamped,
                         lambda msg: self.last['ekf_pose'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['ekf_odom'], Odometry,
                         lambda msg: self.last['ekf_odom'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['controller_pose'], PoseStamped,
                         lambda msg: self.last['controller_pose'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['raw_aruco_pose'], PoseStamped,
                         lambda msg: self.last['raw_aruco_pose'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['thermal_pose'], PoseStamped,
                         lambda msg: self.last['thermal_pose'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['local_pose'], PoseStamped,
                         lambda msg: self.last['local_pose'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['raw_controller_cmd'], TwistStamped,
                         lambda msg: self.last['raw_controller_cmd'].update(msg), queue_size=10)
        rospy.Subscriber(self.topics['mavros_cmd'], TwistStamped,
                         lambda msg: self.last['mavros_cmd'].update(msg), queue_size=10)

        self.pub = rospy.Publisher('/controller/frame_diagnostics', String, queue_size=10)

        self.csv_file = None
        self.csv_writer = None
        self.csv_keys = None
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.csv_file = open(self.csv_path, 'w', newline='')

        rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1e-3)), self.publish)
        rospy.on_shutdown(self.close)

        rospy.loginfo('[FRAME_DIAG] Running. Publishing /controller/frame_diagnostics')
        rospy.loginfo('[FRAME_DIAG] Yaw target: %.3f rad / %.1f deg',
                      self.yaw_target, math.degrees(self.yaw_target))

    def close(self):
        if self.csv_file:
            self.csv_file.close()

    def publish(self, _event):
        now = rospy.Time.now()

        data = OrderedDict()
        data['stamp'] = now.to_sec()
        data['yaw_target_rad'] = self.yaw_target
        data['yaw_target_deg'] = math.degrees(self.yaw_target)
        data['topics'] = self.topics
        data['ages'] = OrderedDict((name, self.last[name].age(now)) for name in self.last)

        if self.last['ekf_pose'].msg is not None:
            ekf = pose_to_dict(self.last['ekf_pose'].msg)
            yaw_err = wrap_pi(self.yaw_target - ekf['yaw_rad'])
            ekf['yaw_error_to_target_rad'] = yaw_err
            ekf['yaw_error_to_target_deg'] = math.degrees(yaw_err)
            ekf['drone_front_nearest_axis'] = nearest_axis(ekf['yaw_rad'])
            data['ekf_pose'] = ekf

        if self.last['ekf_odom'].msg is not None:
            data['ekf_odom_pose'] = odom_to_pose_dict(self.last['ekf_odom'].msg)

        for name in ['controller_pose', 'raw_aruco_pose', 'thermal_pose', 'local_pose']:
            if self.last[name].msg is not None:
                data[name] = pose_to_dict(self.last[name].msg)

        for name in ['raw_controller_cmd', 'mavros_cmd']:
            if self.last[name].msg is not None:
                data[name] = twist_to_dict(self.last[name].msg)

        # If MAVROS command is world/local velocity and local pose yaw is known,
        # reconstruct the approximate body-frame linear command that produced it.
        if self.last['local_pose'].msg is not None and self.last['mavros_cmd'].msg is not None:
            yaw = yaw_from_quat(self.last['local_pose'].msg.pose.orientation)
            vx_w = self.last['mavros_cmd'].msg.twist.linear.x
            vy_w = self.last['mavros_cmd'].msg.twist.linear.y
            c = math.cos(yaw)
            s = math.sin(yaw)
            data['mavros_cmd_reconstructed_body_xy'] = OrderedDict([
                ('vx_body_est', c * vx_w + s * vy_w),
                ('vy_body_est', -s * vx_w + c * vy_w),
            ])

        stale = [name for name, age in data['ages'].items()
                 if math.isfinite(age) and age > self.max_age_warn]
        data['stale_topics_over_threshold'] = stale

        msg = String()
        msg.data = json.dumps(data, allow_nan=True)
        self.pub.publish(msg)

        if self.csv_file:
            row = OrderedDict()
            flatten('', data, row)
            if self.csv_writer is None:
                self.csv_keys = list(row.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_keys)
                self.csv_writer.writeheader()
            self.csv_writer.writerow({k: row.get(k, '') for k in self.csv_keys})
            self.csv_file.flush()

        if 'ekf_pose' in data:
            rospy.loginfo_throttle(
                2.0,
                '[FRAME_DIAG] pose x=%.2f y=%.2f yaw=%.1fdeg target=%.1fdeg err=%.1fdeg front=%s',
                data['ekf_pose']['x'], data['ekf_pose']['y'], data['ekf_pose']['yaw_deg'],
                data['yaw_target_deg'], data['ekf_pose']['yaw_error_to_target_deg'],
                data['ekf_pose']['drone_front_nearest_axis'])


def main():
    ControllerFrameDiagnostics()
    rospy.spin()


if __name__ == '__main__':
    main()
