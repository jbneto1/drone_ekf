#!/usr/bin/env python3
"""
Live EKF tuning plotter for rosbag replay.

This node subscribes while a bag is played with --clock and saves focused plots
on shutdown. It is intentionally narrower than plotter.py:

1. Measurements vs EKF
2. Innovations by observation source
3. Control response
4. XY landing view
5. Detrended dead reckoning

Thermal is intentionally omitted for now. ArUco markers, UWB, and laser are
supported when their topics are present.
"""

import json
import math
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tf.transformations as tft
import yaml
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry


COMPONENTS = [
    ('X', 'x', 'm'),
    ('Y', 'y', 'm'),
    ('Z', 'z', 'm'),
    ('Yaw', 'yaw', 'deg'),
]


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    try:
        return tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
    except Exception:
        return float('nan')


def finite_mask(*arrays):
    if not arrays:
        return np.array([], dtype=bool)
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def rms(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('nan')
    return float(np.sqrt(np.mean(values * values)))


class Series:
    """Small timestamped numeric container."""

    def __init__(self, label):
        self.label = label
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.wz = []

    def __len__(self):
        return len(self.t)

    def add_pose(self, t, pose):
        self.t.append(t)
        self.x.append(pose.position.x)
        self.y.append(pose.position.y)
        self.z.append(pose.position.z)
        self.yaw.append(yaw_from_quat(pose.orientation))

    def add_point(self, t, point):
        self.t.append(t)
        self.x.append(point.x)
        self.y.append(point.y)
        self.z.append(point.z)
        self.yaw.append(float('nan'))

    def add_twist(self, t, twist):
        self.t.append(t)
        self.vx.append(twist.linear.x)
        self.vy.append(twist.linear.y)
        self.vz.append(twist.linear.z)
        self.wz.append(twist.angular.z)

    def pose_arrays(self, unwrap_yaw=True):
        return self._arrays(['x', 'y', 'z', 'yaw'], unwrap_yaw=unwrap_yaw)

    def twist_arrays(self):
        return self._arrays(['vx', 'vy', 'vz', 'wz'], unwrap_yaw=False)

    def _arrays(self, keys, unwrap_yaw):
        if not self.t:
            data = {'t': np.array([])}
            for key in keys:
                data[key] = np.array([])
            return data

        order = np.argsort(np.asarray(self.t))
        t = np.asarray(self.t, dtype=float)[order]

        # Keep the first value for duplicate stamps. Duplicate stamps can happen
        # in replayed bags and make interpolation ambiguous.
        keep = np.ones(len(t), dtype=bool)
        if len(t) > 1:
            keep[1:] = np.diff(t) > 1e-9
        t = t[keep]

        data = {'t': t}
        for key in keys:
            values = np.asarray(getattr(self, key), dtype=float)[order][keep]
            if key == 'yaw' and unwrap_yaw:
                valid = np.isfinite(values)
                if np.count_nonzero(valid) > 1:
                    values = values.copy()
                    values[valid] = np.unwrap(values[valid])
            data[key] = values
        return data


class EKFTuningPlotterLive:
    def __init__(self):
        rospy.init_node('ekf_tuning_plotter_live', anonymous=True)
        self.load_config()

        self.start_time = None
        self.shutdown_complete = False

        self.ekf = Series('EKF')
        self.dead_reckoning = Series('Dead reckoning')
        self.measurements = OrderedDict()
        self.commands = OrderedDict([
            ('raw', Series('Controller raw command (landpad frame)')),
            ('body', Series('Controller body command (body frame)')),
            ('mavros', Series('MAVROS setpoint command')),
            ('velocity_body', Series('Measured velocity body')),
        ])

        self.setup_measurement_series()
        self.setup_subscribers()

        rospy.on_shutdown(self.save_and_exit)
        rospy.loginfo('[EKF_TUNING] Writing focused plots to: %s', self.output_dir)
        rospy.loginfo('[EKF_TUNING] Play a bag with --clock, then Ctrl-C this node to save plots.')

    def load_config(self):
        config_path = rospy.get_param('~config_file', '')
        if not config_path:
            pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_path, 'config', 'ekf_params.yaml')

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            rospy.loginfo('[EKF_TUNING] Loaded config: %s', config_path)
        else:
            rospy.logwarn('[EKF_TUNING] Config not found, using default topics: %s', config_path)

        default_root = self.config.get('plotter', {}).get('save_dir', '/tmp/ekf_plots')
        default_dir = os.path.join(
            default_root,
            'ekf_tuning_live_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        self.output_dir = rospy.get_param('~output_dir', default_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_quiver_arrows = int(rospy.get_param('~max_quiver_arrows', 45))
        self.include_laser = bool(rospy.get_param('~include_laser', True))

    def setup_measurement_series(self):
        sensors = self.config.get('sensors', {})
        marker_cfg = sensors.get('aruco', {}).get('markers', {})
        marker_ids = sorted(int(mid) for mid in marker_cfg.keys()) if marker_cfg else [363, 417, 682]

        for marker_id in marker_ids:
            self.measurements[f'aruco_{marker_id}'] = Series(f'ArUco ID {marker_id}')

        self.measurements['uwb'] = Series('UWB')
        if self.include_laser:
            self.measurements['laser'] = Series('Laser')

    def setup_subscribers(self):
        topics = self.config.get('output_topics', {})

        rospy.Subscriber(
            topics.get('ekf_pose', '/ekf/pose'),
            PoseStamped, self.ekf_pose_cb, queue_size=100
        )
        rospy.Subscriber(
            topics.get('dead_reckoning', '/ekf/dead_reckoning'),
            PoseStamped, self.dead_reckoning_cb, queue_size=100
        )

        marker_prefix = topics.get(
            'aruco_marker_measurement_prefix',
            '/ekf/measurements/aruco/marker_'
        )
        for key in list(self.measurements.keys()):
            if not key.startswith('aruco_'):
                continue
            marker_id = key.split('_', 1)[1]
            rospy.Subscriber(
                f'{marker_prefix}{marker_id}',
                PoseStamped,
                lambda msg, source=key: self.pose_measurement_cb(source, msg),
                queue_size=100
            )

        rospy.Subscriber(
            topics.get('uwb_measurement', '/ekf/measurements/uwb'),
            PointStamped,
            lambda msg: self.point_measurement_cb('uwb', msg),
            queue_size=100
        )

        if 'laser' in self.measurements:
            rospy.Subscriber(
                topics.get('laser_measurement', '/ekf/measurements/laser'),
                PointStamped,
                lambda msg: self.point_measurement_cb('laser', msg),
                queue_size=100
            )

        rospy.Subscriber('/controller/raw_cmd_vel', TwistStamped,
                         lambda msg: self.twist_cb('raw', msg), queue_size=100)
        rospy.Subscriber('/controller/body_cmd_vel', TwistStamped,
                         lambda msg: self.twist_cb('body', msg), queue_size=100)
        rospy.Subscriber('/mavros/setpoint_velocity/cmd_vel', TwistStamped,
                         lambda msg: self.twist_cb('mavros', msg), queue_size=100)
        rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped,
                         lambda msg: self.twist_cb('velocity_body', msg), queue_size=100)

    def rel_time(self, stamp):
        if stamp.is_zero():
            t = rospy.Time.now().to_sec()
        else:
            t = stamp.to_sec()
        if self.start_time is None:
            self.start_time = t
        return t - self.start_time

    def ekf_pose_cb(self, msg):
        if not self.shutdown_complete:
            self.ekf.add_pose(self.rel_time(msg.header.stamp), msg.pose)

    def dead_reckoning_cb(self, msg):
        if not self.shutdown_complete:
            self.dead_reckoning.add_pose(self.rel_time(msg.header.stamp), msg.pose)

    def pose_measurement_cb(self, source, msg):
        if not self.shutdown_complete and source in self.measurements:
            self.measurements[source].add_pose(self.rel_time(msg.header.stamp), msg.pose)

    def point_measurement_cb(self, source, msg):
        if not self.shutdown_complete and source in self.measurements:
            self.measurements[source].add_point(self.rel_time(msg.header.stamp), msg.point)

    def twist_cb(self, source, msg):
        if not self.shutdown_complete and source in self.commands:
            self.commands[source].add_twist(self.rel_time(msg.header.stamp), msg.twist)

    def plot_measurements_vs_ekf(self, ekf, measurements):
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Measurements vs EKF', fontsize=15, fontweight='bold')
        styles = self.measurement_styles()

        for ax, (title, key, unit) in zip(axes, COMPONENTS):
            if len(ekf['t']) > 0:
                values = np.degrees(ekf[key]) if key == 'yaw' else ekf[key]
                ax.plot(ekf['t'], values, color='blue', linewidth=2.0, label='EKF')

            for source, data in measurements.items():
                arr = data.pose_arrays()
                if len(arr['t']) == 0 or key not in arr:
                    continue
                values = np.degrees(arr[key]) if key == 'yaw' else arr[key]
                mask = finite_mask(arr['t'], values)
                if not np.any(mask):
                    continue
                style = styles.get(source, {})
                ax.scatter(arr['t'][mask], values[mask],
                           s=style.get('s', 24),
                           marker=style.get('marker', 'o'),
                           color=style.get('color', None),
                           alpha=style.get('alpha', 0.75),
                           label=data.label)

            ax.set_ylabel(f'{title} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8, ncol=2)

        axes[-1].set_xlabel('Time (s)')
        self.save_fig(fig, '01_measurements_vs_ekf.png')

    def plot_innovations(self, ekf, measurements):
        residuals = self.compute_innovations(ekf, measurements)
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Measurement Innovations: Observation - EKF', fontsize=15, fontweight='bold')
        styles = self.measurement_styles()

        for ax, (title, key, unit) in zip(axes, COMPONENTS):
            plotted = False
            for source, res in residuals.items():
                if key not in res:
                    continue
                t = res['t']
                values = res[key]
                mask = finite_mask(t, values)
                if not np.any(mask):
                    continue
                style = styles.get(source, {})
                ax.scatter(t[mask], values[mask],
                           s=style.get('s', 22),
                           marker=style.get('marker', 'o'),
                           color=style.get('color', None),
                           alpha=style.get('alpha', 0.75),
                           label=measurements[source].label)
                plotted = True

            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(f'{title} residual ({unit})')
            ax.grid(True, alpha=0.3)
            if plotted:
                ax.legend(loc='upper right', fontsize=8, ncol=2)
            else:
                ax.text(0.5, 0.5, 'No observations for this component',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, alpha=0.55)

        axes[-1].set_xlabel('Time (s)')
        self.save_fig(fig, '02_innovations.png')
        return residuals

    def compute_innovations(self, ekf, measurements):
        residuals = OrderedDict()
        if len(ekf['t']) < 2:
            return residuals

        for source, data in measurements.items():
            arr = data.pose_arrays()
            if len(arr['t']) == 0:
                continue
            res = {'t': arr['t']}
            for key in ['x', 'y', 'z']:
                values = arr[key]
                mask = (finite_mask(arr['t'], values) &
                        (arr['t'] >= ekf['t'][0]) &
                        (arr['t'] <= ekf['t'][-1]))
                out = np.full(len(arr['t']), np.nan)
                if np.any(mask):
                    ekf_interp = np.interp(arr['t'][mask], ekf['t'], ekf[key])
                    out[mask] = values[mask] - ekf_interp
                res[key] = out

            yaw_values = arr['yaw']
            yaw_res = np.full(len(arr['t']), np.nan)
            mask = (finite_mask(arr['t'], yaw_values) &
                    (arr['t'] >= ekf['t'][0]) &
                    (arr['t'] <= ekf['t'][-1]))
            if np.any(mask):
                ekf_yaw = np.interp(arr['t'][mask], ekf['t'], ekf['yaw'])
                yaw_res[mask] = [math.degrees(wrap_pi(m - e))
                                 for m, e in zip(yaw_values[mask], ekf_yaw)]
            res['yaw'] = yaw_res
            residuals[source] = res
        return residuals

    def plot_control_response(self, ekf, commands):
        fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
        fig.suptitle('Control Response', fontsize=15, fontweight='bold')

        raw = commands['raw'].twist_arrays()
        body = commands['body'].twist_arrays()
        mavros = commands['mavros'].twist_arrays()
        vel_body = commands['velocity_body'].twist_arrays()

        ax = axes[0]
        if len(ekf['t']) > 0:
            ax.plot(ekf['t'], -ekf['x'], color='tab:red', linewidth=1.8, label='-EKF x error (m)')
            ax.plot(ekf['t'], -ekf['y'], color='tab:blue', linewidth=1.8, label='-EKF y error (m)')
        ax.set_ylabel('Position error (m)')
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        if len(raw['t']) > 0:
            ax2.plot(raw['t'], raw['vx'], color='tab:red', linestyle='--', label='raw vx landpad (m/s)')
            ax2.plot(raw['t'], raw['vy'], color='tab:blue', linestyle='--', label='raw vy landpad (m/s)')
        ax2.set_ylabel('Raw command (m/s)')
        self.merge_legends(ax, ax2, loc='upper right')
        ax.set_title('Controller sign check: landpad correction vs EKF error')

        ax = axes[1]
        if len(body['t']) > 0:
            ax.plot(body['t'], body['vx'], color='tab:red', linestyle='--', label='body cmd vx')
            ax.plot(body['t'], body['vy'], color='tab:blue', linestyle='--', label='body cmd vy')
        if len(vel_body['t']) > 0:
            ax.plot(vel_body['t'], vel_body['vx'], color='tab:red', linewidth=1.7, alpha=0.75, label='measured body vx')
            ax.plot(vel_body['t'], vel_body['vy'], color='tab:blue', linewidth=1.7, alpha=0.75, label='measured body vy')
        ax.set_ylabel('Body XY velocity (m/s)')
        ax.set_title('Body-frame command vs measured body velocity')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        ax = axes[2]
        if len(mavros['t']) > 0:
            ax.plot(mavros['t'], mavros['vx'], color='tab:red', label='MAVROS vx')
            ax.plot(mavros['t'], mavros['vy'], color='tab:blue', label='MAVROS vy')
        if len(raw['t']) > 0:
            ax.plot(raw['t'], raw['vx'], color='tab:red', linestyle=':', alpha=0.65, label='raw vx landpad')
            ax.plot(raw['t'], raw['vy'], color='tab:blue', linestyle=':', alpha=0.65, label='raw vy landpad')
        ax.set_ylabel('XY command (m/s)')
        ax.set_title('Final MAVROS command and raw landpad command')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        ax = axes[3]
        if len(body['t']) > 0:
            ax.plot(body['t'], body['vz'], color='tab:green', linestyle='--', label='body cmd vz')
            ax.plot(body['t'], body['wz'], color='tab:purple', linestyle='--', label='body cmd yaw rate')
        if len(vel_body['t']) > 0:
            ax.plot(vel_body['t'], vel_body['vz'], color='tab:green', alpha=0.75, label='measured body vz')
            ax.plot(vel_body['t'], vel_body['wz'], color='tab:purple', alpha=0.75, label='measured yaw rate')
        ax.set_ylabel('Z / yaw-rate')
        ax.set_title('Vertical and yaw command response')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlabel('Time (s)')

        self.save_fig(fig, '03_control_response.png')

    def plot_xy_landing_view(self, ekf, measurements, commands):
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.suptitle('XY Landing View in Landpad Frame', fontsize=15, fontweight='bold')
        styles = self.measurement_styles()

        if len(ekf['t']) > 0:
            ax.plot(ekf['x'], ekf['y'], color='blue', linewidth=2.0, label='EKF trajectory')
            ax.scatter([ekf['x'][0]], [ekf['y'][0]], color='blue', marker='o', s=60, label='EKF start')
            ax.scatter([ekf['x'][-1]], [ekf['y'][-1]], color='blue', marker='x', s=80, label='EKF end')

        for source, data in measurements.items():
            arr = data.pose_arrays()
            if len(arr['t']) == 0:
                continue
            mask = finite_mask(arr['x'], arr['y'])
            if not np.any(mask):
                continue
            style = styles.get(source, {})
            ax.scatter(arr['x'][mask], arr['y'][mask],
                       s=style.get('s', 24),
                       marker=style.get('marker', 'o'),
                       color=style.get('color', None),
                       alpha=style.get('alpha', 0.55),
                       label=data.label)

        raw = commands['raw'].twist_arrays()
        if len(ekf['t']) > 2 and len(raw['t']) > 2:
            sample_count = min(self.max_quiver_arrows, len(ekf['t']))
            sample_idx = np.linspace(0, len(ekf['t']) - 1, sample_count).astype(int)
            vx = np.interp(ekf['t'][sample_idx], raw['t'], raw['vx'])
            vy = np.interp(ekf['t'][sample_idx], raw['t'], raw['vy'])
            ax.quiver(ekf['x'][sample_idx], ekf['y'][sample_idx], vx, vy,
                      angles='xy', scale_units='xy', scale=4.0,
                      color='black', alpha=0.35, width=0.003,
                      label='raw command arrows')

        ax.scatter([0.0], [0.0], color='black', marker='+', s=160, label='Setpoint')
        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.35)
        ax.axvline(0.0, color='black', linewidth=0.8, alpha=0.35)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best', fontsize=8)
        self.save_fig(fig, '04_xy_landing_view.png')

    def plot_detrended_dead_reckoning(self, ekf, dr):
        if len(ekf['t']) == 0 or len(dr['t']) == 0:
            rospy.logwarn('[EKF_TUNING] Skipping detrended dead reckoning plot; missing data.')
            return

        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Detrended Dead Reckoning vs EKF\n'
                     'Relative change only; absolute DR is not a moving-pad reference',
                     fontsize=15, fontweight='bold')

        for ax, (title, key, unit) in zip(axes, COMPONENTS):
            ekf_values = np.degrees(ekf[key]) if key == 'yaw' else ekf[key]
            dr_values = np.degrees(dr[key]) if key == 'yaw' else dr[key]

            ekf_mask = finite_mask(ekf['t'], ekf_values)
            dr_mask = finite_mask(dr['t'], dr_values)
            if np.any(ekf_mask):
                ekf_delta = ekf_values[ekf_mask] - ekf_values[ekf_mask][0]
                ax.plot(ekf['t'][ekf_mask], ekf_delta, color='blue', linewidth=2.0, label='EKF delta')
            if np.any(dr_mask):
                dr_delta = dr_values[dr_mask] - dr_values[dr_mask][0]
                ax.plot(dr['t'][dr_mask], dr_delta, color='green', linestyle='--',
                        linewidth=1.8, label='Dead reckoning delta')

            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.35)
            ax.set_ylabel(f'Delta {title} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time (s)')
        self.save_fig(fig, '05_dead_reckoning_detrended.png')

    def measurement_styles(self):
        colors = ['tab:red', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:gray']
        styles = {}
        marker_idx = 0
        for source in self.measurements:
            if source.startswith('aruco_'):
                styles[source] = {
                    'color': colors[marker_idx % len(colors)],
                    'marker': 'o',
                    'alpha': 0.78,
                    's': 28
                }
                marker_idx += 1
            elif source == 'uwb':
                styles[source] = {'color': 'tab:purple', 'marker': '^', 'alpha': 0.78, 's': 36}
            elif source == 'laser':
                styles[source] = {'color': 'tab:orange', 'marker': 's', 'alpha': 0.80, 's': 32}
        return styles

    def build_summary(self, residuals):
        summary = OrderedDict()
        summary['generated_at'] = datetime.now().isoformat()
        summary['sample_counts'] = OrderedDict()
        summary['sample_counts']['ekf'] = len(self.ekf)
        summary['sample_counts']['dead_reckoning'] = len(self.dead_reckoning)
        summary['sample_counts']['measurements'] = OrderedDict(
            (source, len(series)) for source, series in self.measurements.items()
        )
        summary['sample_counts']['commands'] = OrderedDict(
            (source, len(series)) for source, series in self.commands.items()
        )

        summary['innovation_stats'] = OrderedDict()
        for source, res in residuals.items():
            source_stats = OrderedDict()
            for key in ['x', 'y', 'z', 'yaw']:
                values = np.asarray(res[key])
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                source_stats[key] = OrderedDict([
                    ('count', int(len(values))),
                    ('mean', float(np.mean(values))),
                    ('std', float(np.std(values))),
                    ('rms', rms(values)),
                    ('max_abs', float(np.max(np.abs(values)))),
                ])
            summary['innovation_stats'][self.measurements[source].label] = source_stats
        return summary

    def merge_legends(self, ax1, ax2, loc='best'):
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc=loc, fontsize=8, ncol=2)

    def save_fig(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        rospy.loginfo('[EKF_TUNING] Saved %s', path)

    def save_and_exit(self):
        if self.shutdown_complete:
            return
        self.shutdown_complete = True

        if len(self.ekf) == 0:
            rospy.logwarn('[EKF_TUNING] No EKF samples collected; no plots written.')
            return

        ekf = self.ekf.pose_arrays()
        dr = self.dead_reckoning.pose_arrays()

        try:
            self.plot_measurements_vs_ekf(ekf, self.measurements)
            residuals = self.plot_innovations(ekf, self.measurements)
            self.plot_control_response(ekf, self.commands)
            self.plot_xy_landing_view(ekf, self.measurements, self.commands)
            self.plot_detrended_dead_reckoning(ekf, dr)

            summary = self.build_summary(residuals)
            summary_path = os.path.join(self.output_dir, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, allow_nan=True)
            rospy.loginfo('[EKF_TUNING] Saved %s', summary_path)
        except Exception as exc:
            rospy.logerr('[EKF_TUNING] Failed to save plots: %s', exc)
            import traceback
            traceback.print_exc()

    def run(self):
        rospy.spin()


def main():
    node = EKFTuningPlotterLive()
    node.run()


if __name__ == '__main__':
    main()
