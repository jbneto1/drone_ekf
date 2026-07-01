#!/usr/bin/env python3
"""
EKF Plotter V5 - Combined estimator diagnostics.

This plotter focuses on the plots that help tune and debug the EKF:
1. Combined measurement overview (EKF vs dead reckoning vs all sensors)
2. Pre-fit innovations
3. Covariance diagonal P
4. Selected Kalman gain terms

It intentionally avoids generating one figure per sensor.
"""

import json
import os
import signal
import sys
import time
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import rospy
import tf.transformations as tft
import yaml
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class SensorData:
    """Container for pose/point time series data."""

    def __init__(self, max_points=10000):
        self.times = deque(maxlen=max_points)
        self.x = deque(maxlen=max_points)
        self.y = deque(maxlen=max_points)
        self.z = deque(maxlen=max_points)
        self.yaw = deque(maxlen=max_points)
        self.has_data = False

    def add_pose(self, t, pose):
        self.times.append(t)
        self.x.append(pose.position.x)
        self.y.append(pose.position.y)
        self.z.append(pose.position.z)

        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = tft.euler_from_quaternion(q)[2]
        self.yaw.append(np.degrees(yaw))
        self.has_data = True

    def add_point(self, t, point):
        self.times.append(t)
        self.x.append(point.x if not np.isnan(point.x) else None)
        self.y.append(point.y if not np.isnan(point.y) else None)
        self.z.append(point.z if not np.isnan(point.z) else None)
        self.yaw.append(None)
        self.has_data = True

    def get_arrays(self):
        arrays = {
            'times': np.array(list(self.times)),
            'x': np.array([v if v is not None else np.nan for v in self.x]),
            'y': np.array([v if v is not None else np.nan for v in self.y]),
            'z': np.array([v if v is not None else np.nan for v in self.z]),
            'yaw': np.array([v if v is not None else np.nan for v in self.yaw]),
        }
        if len(arrays['times']) > 1:
            order = np.argsort(arrays['times'], kind='mergesort')
            for key in arrays:
                arrays[key] = arrays[key][order]
        return arrays


class EventBuffer:
    """Stores JSON diagnostic events for later plotting."""

    def __init__(self, max_points=10000):
        self.events = deque(maxlen=max_points)

    def add(self, event):
        self.events.append(event)

    def __len__(self):
        return len(self.events)

    def list(self):
        return sorted(self.events, key=lambda event: event.get('t', 0.0))


class EKFPlotter:
    """Plot combined EKF measurement and diagnostic topics."""

    def __init__(self):
        rospy.init_node('ekf_plotter', anonymous=True)

        self.load_config()
        max_pts = self.config.get('plotter', {}).get('max_points', 10000)

        self.ekf_data = SensorData(max_pts)
        self.dr_data = SensorData(max_pts)
        self.thermal_data = SensorData(max_pts)
        self.laser_data = SensorData(max_pts)
        self.uwb_data = SensorData(max_pts)

        marker_cfg = self.config.get('sensors', {}).get('aruco', {}).get('markers', {})
        marker_ids = sorted(int(mid) for mid in marker_cfg.keys()) if marker_cfg else [363, 417, 682]
        self.aruco_marker_data = {mid: SensorData(max_pts) for mid in marker_ids}

        self.innovation_events = EventBuffer(max_pts)
        self.covariance_events = EventBuffer(max_pts)
        self.kalman_gain_events = EventBuffer(max_pts)
        self.marker_quality_events = EventBuffer(max_pts)
        self.timing_events = EventBuffer(max_pts)
        self.camera_timing_events = EventBuffer(max_pts)
        self.aruco_detector_timing_events = EventBuffer(max_pts)
        self.aruco_ekf_timing_events = EventBuffer(max_pts)

        self.sensor_config = {
            'aruco': {'enabled': False, 'active': False},
            'laser': {'enabled': False, 'active': False},
            'uwb': {'enabled': False, 'active': False},
            'thermal': {'enabled': False, 'active': False},
            'process_model': 'unknown'
        }

        self.marker_colors = {
            363: 'crimson',
            417: 'darkcyan',
            682: 'darkgoldenrod'
        }
        self.sensor_colors = {
            'aruco': 'tab:red',
            'laser': 'orange',
            'uwb': 'purple',
            'thermal': 'saddlebrown'
        }
        self.component_markers = {
            'position': 'o',
            'yaw': '^',
            'z': 's',
            'xy': 'D'
        }
        self.component_gate_colors = {
            'position': 'tab:blue',
            'yaw': 'tab:purple',
            'z': 'tab:orange',
            'xy': 'tab:green'
        }
        self.nis_gate_levels = [
            ('68%', ':'),
            ('95%', '-.'),
            ('99%', '--')
        ]
        plotter_cfg = self.config.get('plotter', {})
        self.plot_dpi = plotter_cfg.get('dpi', 90)
        self.max_plot_points = plotter_cfg.get('max_plot_points', 2500)
        self.max_event_points = plotter_cfg.get('max_event_points', 3000)
        self.tight_bbox = plotter_cfg.get('tight_bbox', False)
        self.shutdown_timeout_sec = float(plotter_cfg.get('shutdown_timeout_sec', 0.0))
        self.detached_shutdown_save = bool(plotter_cfg.get('detached_shutdown_save', True))
        configured_shutdown_log = plotter_cfg.get('shutdown_save_log', '')
        self.shutdown_save_log = (
            self.resolve_config_path(configured_shutdown_log)
            if configured_shutdown_log
            else os.path.join(self.save_dir, 'plotter_shutdown_save.log')
        )
        os.makedirs(os.path.dirname(self.shutdown_save_log), exist_ok=True)
        self.shutdown_status_file = os.environ.get(
            'DRONE_EKF_PLOTTER_STATUS_FILE', ''
        )
        self.write_shutdown_status('initialized')

        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000

        self.start_time = None
        self.shutdown_complete = False

        # Make roslaunch / rospy shutdown call the same save path as Ctrl+C.
        rospy.on_shutdown(self.save_and_exit)

        self.setup_subscribers()

        rospy.loginfo(f"[PLOTTER] Initialized. Saving to: {self.save_dir}")
        rospy.loginfo(f"[PLOTTER] Detached saver log: {self.shutdown_save_log}")
        rospy.loginfo("[PLOTTER] Waiting for data...")

    def resolve_config_path(self, configured_path):
        """Resolve ~, environment variables, and config-relative output paths."""
        expanded = os.path.expandvars(os.path.expanduser(str(configured_path)))
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.config_path_base, expanded)
        return os.path.abspath(expanded)

    def write_shutdown_status(self, state, pid=None, error=None):
        """Atomically report saver state to the optional shell wrapper."""
        if not self.shutdown_status_file:
            return

        status_path = os.path.abspath(
            os.path.expandvars(os.path.expanduser(self.shutdown_status_file))
        )
        status_dir = os.path.dirname(status_path)
        temporary_path = f"{status_path}.tmp.{os.getpid()}"
        try:
            if status_dir:
                os.makedirs(status_dir, exist_ok=True)
            with open(temporary_path, 'w') as status_file:
                status_file.write(f"{state}\n")
                status_file.write(f"{self.shutdown_save_log}\n")
                status_file.write(f"{pid if pid is not None else ''}\n")
                status_file.write(f"{error if error is not None else ''}\n")
            os.replace(temporary_path, status_path)
        except OSError as exc:
            rospy.logwarn("[PLOTTER] Could not write shutdown status: %s", exc)

    def load_config(self):
        """Load configuration from YAML."""
        config_path = rospy.get_param('~config_file', '')
        if not config_path:
            pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_path, 'config', 'ekf_params.yaml')
        config_path = os.path.abspath(
            os.path.expandvars(os.path.expanduser(config_path))
        )
        self.config_path = config_path

        # A relative output path in config/ekf_params.yaml belongs to the
        # package directory, not roslaunch's process cwd (usually ~/.ros).
        config_dir = os.path.dirname(config_path)
        self.config_path_base = (
            os.path.dirname(config_dir)
            if os.path.basename(config_dir) == 'config'
            else config_dir
        )

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            rospy.loginfo(f"[PLOTTER] Loaded config from: {config_path}")

        configured_save_dir = self.config.get(
            'plotter', {}
        ).get('save_dir', '/tmp/ekf_plots')
        self.save_dir = self.resolve_config_path(configured_save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def setup_subscribers(self):
        """Setup ROS subscribers."""
        topics = self.config.get('output_topics', {})

        rospy.Subscriber(
            topics.get('ekf_odom', '/ekf/odom'),
            Odometry, self.ekf_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get('dead_reckoning', '/ekf/dead_reckoning'),
            PoseStamped, self.dr_callback, queue_size=1
        )

        rospy.Subscriber(
            topics.get('thermal_measurement', '/ekf/measurements/thermal'),
            PoseStamped, self.thermal_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get('laser_measurement', '/ekf/measurements/laser'),
            PointStamped, self.laser_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get('uwb_measurement', '/ekf/measurements/uwb'),
            PointStamped, self.uwb_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get('sensor_status', '/ekf/sensor_status'),
            String, self.status_callback, queue_size=1
        )

        aruco_prefix = topics.get(
            'aruco_marker_measurement_prefix',
            '/ekf/measurements/aruco/marker_'
        )
        for marker_id in sorted(self.aruco_marker_data.keys()):
            rospy.Subscriber(
                f'{aruco_prefix}{marker_id}',
                PoseStamped,
                lambda msg, mid=marker_id: self.aruco_marker_callback(msg, mid),
                queue_size=1
            )

        rospy.Subscriber(
            topics.get('innovation_debug', '/ekf/debug/innovation'),
            String, self.innovation_callback, queue_size=100
        )
        rospy.Subscriber(
            topics.get('covariance_debug', '/ekf/debug/covariance'),
            String, self.covariance_callback, queue_size=100
        )
        rospy.Subscriber(
            topics.get('kalman_gain_debug', '/ekf/debug/kalman_gain'),
            String, self.kalman_gain_callback, queue_size=100
        )
        rospy.Subscriber(
            topics.get('aruco_marker_quality', '/aruco/debug/marker_quality'),
            String, self.marker_quality_callback, queue_size=100
        )
        rospy.Subscriber(
            topics.get('timing_debug', '/ekf/debug/timing'),
            String, self.timing_callback, queue_size=200
        )
        rospy.Subscriber(
            topics.get('stereo_camera_timing', '/stereo/debug/timing'),
            String, self.camera_timing_callback, queue_size=200
        )
        rospy.Subscriber(
            topics.get('aruco_detector_timing', '/aruco/debug/timing'),
            String, self.aruco_detector_timing_callback, queue_size=200
        )

    def relative_time_from_sec(self, stamp_sec):
        if self.start_time is None:
            self.start_time = stamp_sec
        return stamp_sec - self.start_time

    def get_time(self, stamp):
        if stamp.is_zero():
            return self.relative_time_from_sec(rospy.Time.now().to_sec())
        return self.relative_time_from_sec(stamp.to_sec())

    def parse_debug_event(self, msg):
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return None
        receive_stamp_sec = rospy.Time.now().to_sec()
        try:
            stamp_sec = float(event.get('stamp', receive_stamp_sec))
        except (TypeError, ValueError):
            stamp_sec = receive_stamp_sec
        if not np.isfinite(stamp_sec) or stamp_sec <= 0.0:
            stamp_sec = receive_stamp_sec
        event['t'] = self.relative_time_from_sec(stamp_sec)
        event['receive_t'] = self.relative_time_from_sec(receive_stamp_sec)
        event['plotter_latency_sec'] = receive_stamp_sec - stamp_sec
        return event

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------

    def ekf_callback(self, msg):
        if self.shutdown_complete:
            return
        self.ekf_data.add_pose(self.get_time(msg.header.stamp), msg.pose.pose)

    def dr_callback(self, msg):
        if self.shutdown_complete:
            return
        self.dr_data.add_pose(self.get_time(msg.header.stamp), msg.pose)

    def thermal_callback(self, msg):
        if self.shutdown_complete:
            return
        self.thermal_data.add_pose(self.get_time(msg.header.stamp), msg.pose)

    def laser_callback(self, msg):
        if self.shutdown_complete:
            return
        self.laser_data.add_point(self.get_time(msg.header.stamp), msg.point)

    def uwb_callback(self, msg):
        if self.shutdown_complete:
            return
        self.uwb_data.add_point(self.get_time(msg.header.stamp), msg.point)

    def status_callback(self, msg):
        try:
            self.sensor_config = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def aruco_marker_callback(self, msg, marker_id):
        if self.shutdown_complete:
            return
        if marker_id not in self.aruco_marker_data:
            self.aruco_marker_data[marker_id] = SensorData(
                self.config.get('plotter', {}).get('max_points', 10000)
            )
        self.aruco_marker_data[marker_id].add_pose(self.get_time(msg.header.stamp), msg.pose)

    def innovation_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.innovation_events.add(event)

    def covariance_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.covariance_events.add(event)

    def kalman_gain_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.kalman_gain_events.add(event)

    def marker_quality_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.marker_quality_events.add(event)

    def timing_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.timing_events.add(event)
            if event.get('stage') == 'aruco_callback_profile':
                self.aruco_ekf_timing_events.add(event)

    def camera_timing_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.camera_timing_events.add(event)

    def aruco_detector_timing_callback(self, msg):
        if self.shutdown_complete:
            return
        event = self.parse_debug_event(msg)
        if event is not None:
            self.aruco_detector_timing_events.add(event)

    # ---------------------------------------------------------------------
    # Plot helpers
    # ---------------------------------------------------------------------

    def unwrap_yaw(self, yaw_array):
        valid_mask = ~np.isnan(yaw_array)
        if np.sum(valid_mask) < 2:
            return yaw_array
        result = yaw_array.copy()
        unwrapped = np.degrees(np.unwrap(np.radians(yaw_array[valid_mask])))
        result[valid_mask] = unwrapped
        return result

    def event_color(self, sensor_name, marker_id=None):
        if sensor_name == 'aruco' and marker_id in self.marker_colors:
            return self.marker_colors[marker_id]
        return self.sensor_colors.get(sensor_name, 'black')

    def event_label(self, sensor_name, marker_id=None):
        if sensor_name == 'aruco' and marker_id is not None:
            return f'ArUco {marker_id}'
        return sensor_name.upper()

    def component_label(self, component):
        return {
            'position': 'position NIS',
            'yaw': 'yaw NIS',
            'z': 'z NIS',
            'xy': 'xy NIS'
        }.get(component, component)

    def component_gate_threshold(self, component, level):
        """Chi-square thresholds used to visually interpret NIS."""
        thresholds = {
            'position': {'68%': 3.53, '95%': 7.815, '99%': 11.345},  # 3 DOF
            'xy': {'68%': 2.30, '95%': 5.991, '99%': 9.210},         # 2 DOF
            'z': {'68%': 1.00, '95%': 3.841, '99%': 6.635},          # 1 DOF
            'yaw': {'68%': 1.00, '95%': 3.841, '99%': 6.635},        # 1 DOF
        }
        return thresholds.get(component, {}).get(level)

    def legend_marker(self, color, label, marker='o', markersize=7):
        return Line2D(
            [0], [0],
            marker=marker,
            linestyle='None',
            color=color,
            markerfacecolor=color if marker not in ('x', '_') else 'none',
            markeredgecolor=color,
            markersize=markersize,
            label=label
        )

    def status_legend_handles(self, accepted_label='Accepted', rejected_label='Rejected'):
        return [
            self.legend_marker('black', accepted_label, marker='o', markersize=6),
            self.legend_marker('black', rejected_label, marker='x', markersize=7),
        ]

    def time_key(self, t):
        return round(float(t), 6)

    def aruco_update_status_lookup(self):
        component_axes = {
            'position': ('x', 'y', 'z'),
            'xy': ('x', 'y'),
            'z': ('z',),
            'yaw': ('yaw',),
        }
        lookup = {}
        for event in self.innovation_events.list():
            if event.get('sensor') != 'aruco':
                continue
            marker_id = event.get('marker_id')
            if marker_id is None:
                continue
            axes = component_axes.get(event.get('component'))
            if axes is None:
                continue

            accepted = bool(event.get('accepted', True))
            key = self.time_key(event.get('t', 0.0))
            for axis_key in axes:
                lookup[(int(marker_id), axis_key, key)] = accepted
        return lookup

    def aruco_update_acceptance(self, lookup, marker_id, axis_key, times):
        accepted = np.ones(len(times), dtype=bool)
        for idx, t in enumerate(times):
            status = lookup.get((int(marker_id), axis_key, self.time_key(t)))
            if status is not None:
                accepted[idx] = status
        return accepted

    def save_figure(self, fig, filepath):
        if self.tight_bbox:
            fig.savefig(filepath, dpi=self.plot_dpi, bbox_inches='tight')
        else:
            fig.savefig(filepath, dpi=self.plot_dpi)
        plt.close(fig)
        rospy.loginfo(f"[PLOTTER] Saved: {filepath}")

    def thin_indices(self, length, max_points):
        if length <= max_points:
            return np.arange(length)
        return np.linspace(0, length - 1, max_points).astype(int)

    def thin_series(self, times, values, max_points=None):
        if max_points is None:
            max_points = self.max_plot_points
        if len(times) <= max_points:
            return times, values
        idx = self.thin_indices(len(times), max_points)
        return times[idx], values[idx]

    def thin_events(self, events, max_points=None):
        if max_points is None:
            max_points = self.max_event_points
        if len(events) <= max_points:
            return events
        idx = self.thin_indices(len(events), max_points)
        return [events[i] for i in idx]

    def innovation_value(self, event, axis_key, field='innovation'):
        values = event.get(field)
        if values is None:
            return None

        component = event.get('component')
        if component == 'position':
            index_map = {'x': 0, 'y': 1, 'z': 2}
        elif component == 'xy':
            index_map = {'x': 0, 'y': 1}
        elif component == 'z':
            index_map = {'z': 0}
        elif component == 'yaw':
            index_map = {'yaw': 0}
        else:
            return None

        if axis_key not in index_map or index_map[axis_key] >= len(values):
            return None

        value = float(values[index_map[axis_key]])
        if axis_key == 'yaw':
            return np.degrees(value)
        return value

    def selected_gain_value(self, event, axis_key):
        gain = event.get('kalman_gain')
        if gain is None:
            return None

        K = np.asarray(gain, dtype=float)
        component = event.get('component')
        rows, cols = K.shape

        if component == 'position':
            if axis_key == 'x' and rows > 0 and cols > 0:
                return K[0, 0]
            if axis_key == 'y' and rows > 1 and cols > 1:
                return K[1, 1]
            if axis_key == 'z' and rows > 2 and cols > 2:
                return K[2, 2]
        elif component == 'xy':
            if axis_key == 'x' and rows > 0 and cols > 0:
                return K[0, 0]
            if axis_key == 'y' and rows > 1 and cols > 1:
                return K[1, 1]
        elif component == 'z':
            if axis_key == 'z' and rows > 2 and cols > 0:
                return K[2, 0]
        elif component == 'yaw':
            if axis_key == 'yaw':
                if event.get('measurement_gain') is not None:
                    return float(event['measurement_gain'])
                if rows > 8 and cols > 0:
                    return K[8, 0]

        return None

    # ---------------------------------------------------------------------
    # Plot generation
    # ---------------------------------------------------------------------

    def save_all_plots(self, prefix='final'):
        rospy.loginfo(f"[PLOTTER] Generating plots with prefix: {prefix}")
        start_wall = time.time()
        yaw_cfg = self.config.get('plotter', {}).get('yaw', {})
        should_unwrap = yaw_cfg.get('unwrap', True)

        ekf = self.ekf_data.get_arrays()
        dr = self.dr_data.get_arrays()
        thermal = self.thermal_data.get_arrays()
        laser = self.laser_data.get_arrays()
        uwb = self.uwb_data.get_arrays()
        aruco_markers = {
            mid: data.get_arrays()
            for mid, data in self.aruco_marker_data.items()
        }

        if should_unwrap:
            ekf['yaw'] = self.unwrap_yaw(ekf['yaw'])
            dr['yaw'] = self.unwrap_yaw(dr['yaw'])
            thermal['yaw'] = self.unwrap_yaw(thermal['yaw'])
            for marker in aruco_markers.values():
                marker['yaw'] = self.unwrap_yaw(marker['yaw'])

        plot_count = 0
        timeout_hit = False
        skipped_timeout = []
        skipped_no_data = []
        failed_plots = []

        deadline = None
        # 0.0 means no internal plotter timeout; finish all available plots.
        if prefix == 'final' and self.shutdown_timeout_sec > 0.0:
            deadline = time.time() + self.shutdown_timeout_sec

        def run_plot(label, should_plot, plot_fn):
            nonlocal plot_count, timeout_hit

            if not should_plot:
                skipped_no_data.append(label)
                return

            if timeout_hit:
                skipped_timeout.append(label)
                return

            if deadline is not None and time.time() >= deadline:
                timeout_hit = True
                skipped_timeout.append(label)
                rospy.logwarn(
                    "[PLOTTER] Shutdown save budget reached before %s; "
                    "skipping remaining plots",
                    label
                )
                return

            try:
                plot_fn()
                plot_count += 1
            except Exception as exc:
                failed_plots.append((label, exc))
                rospy.logerr("[PLOTTER] Failed while generating %s: %s", label, exc)

        run_plot(
            'combined comparison',
            self.ekf_data.has_data and len(ekf['times']) > 10,
            lambda: self.plot_combined_measurements(
                ekf, dr, thermal, laser, uwb, aruco_markers, yaw_cfg, prefix
            )
        )
        run_plot('innovations', len(self.innovation_events) > 0, lambda: self.plot_innovations(prefix))
        run_plot('NIS', len(self.innovation_events) > 0, lambda: self.plot_nis(prefix))
        run_plot('covariance', len(self.covariance_events) > 0, lambda: self.plot_covariance(prefix))
        run_plot('kalman gain', len(self.kalman_gain_events) > 0, lambda: self.plot_selected_gains(prefix))
        run_plot('marker quality', len(self.marker_quality_events) > 0, lambda: self.plot_marker_quality(prefix))
        run_plot(
            'timing diagnostics',
            len(self.timing_events) > 0 or len(self.innovation_events) > 0 or len(self.covariance_events) > 0,
            lambda: self.plot_timing_diagnostics(prefix)
        )
        run_plot(
            'ArUco latency profile',
            (
                len(self.camera_timing_events) > 0 or
                len(self.aruco_detector_timing_events) > 0 or
                len(self.aruco_ekf_timing_events) > 0
            ),
            lambda: self.plot_aruco_latency_profile(prefix)
        )

        elapsed = time.time() - start_wall
        if plot_count == 0:
            rospy.logwarn("[PLOTTER] No plots were generated")
        else:
            rospy.loginfo(
                f"[PLOTTER] Generated {plot_count} plot(s) in {self.save_dir} "
                f"({elapsed:.1f} s)"
            )
        if skipped_timeout:
            rospy.logwarn("[PLOTTER] Skipped due to shutdown timeout: %s", ", ".join(skipped_timeout))
        if failed_plots:
            rospy.logerr("[PLOTTER] Failed plots: %s", ", ".join(label for label, _exc in failed_plots))

        return {
            'plot_count': plot_count,
            'timeout_hit': timeout_hit,
            'skipped_timeout': skipped_timeout,
            'skipped_no_data': skipped_no_data,
            'failed_plots': failed_plots,
            'elapsed_sec': elapsed,
        }

    def plot_combined_measurements(self, ekf, dr, thermal, laser, uwb, aruco_markers,
                                   yaw_cfg, prefix):
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle(
            'Combined Sensor Fusion Overview\n'
            'EKF vs Dead Reckoning vs All Measurements',
            fontsize=14, fontweight='bold'
        )

        components = [
            ('X (m)', 'x'),
            ('Y (m)', 'y'),
            ('Z (m)', 'z'),
            ('Yaw (deg)', 'yaw')
        ]

        styles = {
            'ekf': {'color': 'blue', 'linewidth': 2.5, 'label': 'EKF (Fused)', 'zorder': 5},
            'dr': {'color': 'green', 'linewidth': 2.0, 'linestyle': '--', 'alpha': 0.7,
                   'label': 'Dead Reckoning', 'zorder': 4},
            'thermal': {'color': 'saddlebrown', 'marker': 'D', 's': 34, 'alpha': 0.7,
                        'label': 'Thermal', 'zorder': 3},
            'laser': {'color': 'orange', 'marker': 's', 's': 30, 'alpha': 0.7,
                      'label': 'Laser', 'zorder': 3},
            'uwb': {'color': 'purple', 'marker': '^', 's': 30, 'alpha': 0.7,
                    'label': 'UWB', 'zorder': 3},
        }

        all_handles = []
        all_labels = []
        aruco_update_lookup = self.aruco_update_status_lookup()
        plotted_rejected_aruco = False

        for ax_idx, (label, key) in enumerate(components):
            ax = axes[ax_idx]
            is_yaw = key == 'yaw'

            if len(ekf['times']) > 0:
                t_plot, v_plot = self.thin_series(ekf['times'], ekf[key])
                line, = ax.plot(
                    t_plot, v_plot,
                    color=styles['ekf']['color'],
                    linewidth=styles['ekf']['linewidth'],
                    label=styles['ekf']['label'],
                    zorder=styles['ekf']['zorder']
                )
                if ax_idx == 0:
                    all_handles.append(line)
                    all_labels.append(styles['ekf']['label'])

            if len(dr['times']) > 0:
                skip_dr = is_yaw and not yaw_cfg.get('show_dead_reckoning', True)
                if not skip_dr:
                    t_plot, v_plot = self.thin_series(dr['times'], dr[key])
                    line, = ax.plot(
                        t_plot, v_plot,
                        color=styles['dr']['color'],
                        linewidth=styles['dr']['linewidth'],
                        linestyle=styles['dr']['linestyle'],
                        alpha=styles['dr']['alpha'],
                        label=styles['dr']['label'],
                        zorder=styles['dr']['zorder']
                    )
                    if ax_idx == 0:
                        all_handles.append(line)
                        all_labels.append(styles['dr']['label'])

            for marker_id, marker_data in sorted(aruco_markers.items()):
                if marker_id not in self.aruco_marker_data or not self.aruco_marker_data[marker_id].has_data:
                    continue
                valid = ~np.isnan(marker_data[key])
                if np.any(valid):
                    label_text = f'ArUco {marker_id}'
                    color = self.marker_colors.get(marker_id, 'black')
                    times = marker_data['times'][valid]
                    values = marker_data[key][valid]
                    accepted_mask = self.aruco_update_acceptance(
                        aruco_update_lookup, marker_id, key, times
                    )
                    for accepted, marker_style, size, alpha, linewidth in (
                        (True, 'o', 30, 0.75, 0.4),
                        (False, 'x', 36, 0.95, 0.9),
                    ):
                        status_mask = accepted_mask if accepted else ~accepted_mask
                        if not np.any(status_mask):
                            continue
                        t_plot, v_plot = self.thin_series(times[status_mask], values[status_mask])
                        ax.scatter(
                            t_plot, v_plot,
                            c=color, s=size, marker=marker_style, alpha=alpha,
                            edgecolors=color, linewidths=linewidth, zorder=4
                        )
                        if not accepted:
                            plotted_rejected_aruco = True
                    if label_text not in all_labels:
                        all_handles.append(self.legend_marker(color, label_text, marker='o'))
                        all_labels.append(label_text)

            for sensor_name, sensor_data in (
                ('thermal', thermal),
                ('laser', laser),
                ('uwb', uwb),
            ):
                sensor_store = getattr(self, f'{sensor_name}_data')
                if sensor_store.has_data and len(sensor_data['times']) > 0:
                    valid = ~np.isnan(sensor_data[key])
                    if np.any(valid):
                        style = styles[sensor_name]
                        t_plot, v_plot = self.thin_series(sensor_data['times'][valid], sensor_data[key][valid])
                        scatter = ax.scatter(
                            t_plot, v_plot,
                            c=style['color'],
                            s=style['s'],
                            marker=style['marker'],
                            alpha=style['alpha'],
                            label=style['label'],
                            edgecolors='none',
                            zorder=style['zorder']
                        )
                        if style['label'] not in all_labels:
                            all_handles.append(scatter)
                            all_labels.append(style['label'])

            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            if is_yaw and yaw_cfg.get('ylim') is not None:
                ax.set_ylim(yaw_cfg['ylim'][0], yaw_cfg['ylim'][1])

        if all_handles:
            legend_title = None
            if plotted_rejected_aruco:
                status_handles = self.status_legend_handles(
                    'Accepted ArUco update',
                    'Rejected ArUco update'
                )
                all_handles.extend(status_handles)
                all_labels.extend([handle.get_label() for handle in status_handles])
                legend_title = 'Color: source/marker | Symbol: ArUco update status'
            axes[0].legend(
                all_handles, all_labels,
                loc='upper right', fontsize=9, ncol=2,
                title=legend_title
            )

        process_model = self.sensor_config.get('process_model', 'Unknown')
        axes[0].annotate(
            f'Process Model: {process_model}',
            xy=(0.01, 0.98), xycoords='axes fraction',
            fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'combined_comparison_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_innovations(self, prefix):
        events = self.thin_events(self.innovation_events.list())
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle('EKF Pre-fit Innovations', fontsize=14, fontweight='bold')

        axis_specs = [
            ('X innovation (m)', 'x'),
            ('Y innovation (m)', 'y'),
            ('Z innovation (m)', 'z'),
            ('Yaw innovation (deg)', 'yaw')
        ]

        legend_entries = {}
        for ax, (label, axis_key) in zip(axes, axis_specs):
            grouped_points = {}
            for event in events:
                value = self.innovation_value(event, axis_key, field='innovation')
                if value is None:
                    continue

                sensor_name = event.get('sensor', 'unknown')
                marker_id = event.get('marker_id')
                accepted = bool(event.get('accepted', True))
                color = self.event_color(sensor_name, marker_id)
                legend_label = self.event_label(sensor_name, marker_id)
                key = (legend_label, color, accepted)
                grouped_points.setdefault(key, [[], []])
                grouped_points[key][0].append(event['t'])
                grouped_points[key][1].append(value)
                if legend_label not in legend_entries:
                    legend_entries[legend_label] = self.legend_marker(color, legend_label)

            for (legend_label, color, accepted), (times, values) in grouped_points.items():
                ax.scatter(
                    times, values,
                    c=color,
                    s=24,
                    marker='o' if accepted else 'x',
                    alpha=0.7 if accepted else 0.95,
                    linewidths=0.8 if not accepted else 0.0
                )

            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)

        if legend_entries:
            handles = list(legend_entries.values())
            labels = list(legend_entries.keys())
            if any(not bool(event.get('accepted', True)) for event in events):
                status_handles = self.status_legend_handles('Accepted update', 'Rejected update')
                handles.extend(status_handles)
                labels.extend([handle.get_label() for handle in status_handles])
            axes[0].legend(
                handles, labels,
                loc='upper right', fontsize=9, ncol=2,
                title='Color: sensor/marker | Symbol: status'
            )
        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'innovations_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_nis(self, prefix):
        events = self.thin_events(self.innovation_events.list())
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle('Normalized Innovation Squared (NIS)', fontsize=14, fontweight='bold')

        sensor_legend = {}
        component_set = set()
        grouped_points = {}
        all_nis_values = []
        all_sqrt_nis_values = []
        any_rejected = False
        for event in events:
            value = event.get('mahalanobis_sq')
            if value is None:
                continue

            sensor_name = event.get('sensor', 'unknown')
            marker_id = event.get('marker_id')
            component = event.get('component', 'unknown')
            accepted = bool(event.get('accepted', True))
            color = self.event_color(sensor_name, marker_id)
            sensor_label = self.event_label(sensor_name, marker_id)
            marker_style = self.component_markers.get(component, 'o') if accepted else 'x'
            sqrt_value = np.sqrt(max(float(value), 0.0))
            all_nis_values.append(float(value))
            all_sqrt_nis_values.append(float(sqrt_value))
            key = (sensor_label, color, component, marker_style, accepted)
            grouped_points.setdefault(key, [[], [], []])
            grouped_points[key][0].append(event['t'])
            grouped_points[key][1].append(value)
            grouped_points[key][2].append(sqrt_value)
            any_rejected = any_rejected or not accepted
            if sensor_label not in sensor_legend:
                sensor_legend[sensor_label] = self.legend_marker(color, sensor_label)
            component_set.add(component)

        for (_sensor_label, color, _component, marker_style, accepted), (times, values, sqrt_values) in grouped_points.items():
            axes[0].scatter(
                times, values,
                c=color, s=24, marker=marker_style, alpha=0.75,
                linewidths=0.8 if not accepted else 0.0
            )
            axes[1].scatter(
                times, sqrt_values,
                c=color, s=24, marker=marker_style, alpha=0.75,
                linewidths=0.8 if not accepted else 0.0
            )

        gate_handles = []
        for component in sorted(component_set):
            gate_color = self.component_gate_colors.get(component, 'black')
            for level, linestyle in self.nis_gate_levels:
                gate_value = self.component_gate_threshold(component, level)
                if gate_value is None:
                    continue
                gate_label = f'{self.component_label(component)} {level} gate'
                line = axes[0].axhline(
                    gate_value,
                    linestyle=linestyle,
                    linewidth=1.0,
                    color=gate_color,
                    alpha=0.55,
                    label=gate_label
                )
                axes[1].axhline(
                    np.sqrt(gate_value),
                    linestyle=linestyle,
                    linewidth=1.0,
                    color=gate_color,
                    alpha=0.55
                )
                gate_handles.append(line)

        axes[0].set_ylabel('NIS = innovation^T S^-1 innovation', fontsize=10)
        axes[1].set_ylabel('sqrt(NIS)', fontsize=10)
        for ax in axes:
            ax.grid(True, alpha=0.3)

        nis_cfg = self.config.get('plotter', {}).get('nis', {})
        nis_ylim = nis_cfg.get('ylim')
        sqrt_ylim = nis_cfg.get('sqrt_ylim')
        if nis_ylim is not None:
            axes[0].set_ylim(nis_ylim[0], nis_ylim[1])
            clipped = sum(v > nis_ylim[1] for v in all_nis_values)
            if clipped > 0:
                axes[0].annotate(
                    f'{clipped} point(s) above axis limit',
                    xy=(0.01, 0.95), xycoords='axes fraction',
                    fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
                )
        if sqrt_ylim is not None:
            axes[1].set_ylim(sqrt_ylim[0], sqrt_ylim[1])
            clipped = sum(v > sqrt_ylim[1] for v in all_sqrt_nis_values)
            if clipped > 0:
                axes[1].annotate(
                    f'{clipped} point(s) above axis limit',
                    xy=(0.01, 0.95), xycoords='axes fraction',
                    fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
                )

        component_handles = [
            self.legend_marker('black', self.component_label(component),
                               marker=self.component_markers.get(component, 'o'), markersize=7)
            for component in sorted(component_set)
        ]
        handles = list(sensor_legend.values()) + component_handles + gate_handles
        labels = [handle.get_label() for handle in handles]
        if any_rejected:
            reject_handle = self.legend_marker('black', 'Rejected update', marker='x', markersize=7)
            handles.append(reject_handle)
            labels.append(reject_handle.get_label())
        if handles:
            axes[0].legend(handles, labels, loc='upper right', fontsize=8, ncol=2,
                           title='Color: sensor/marker | Shape: update type | Dashed: gate')
        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'nis_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_timing_diagnostics(self, prefix):
        timing_events = self.thin_events(self.timing_events.list())
        if not timing_events:
            timing_events = []
            for event in self.innovation_events.list():
                if event.get('age_sec') is None:
                    continue
                timing_events.append({
                    't': event['t'],
                    'stage': 'measurement_update',
                    'sensor': event.get('sensor', 'unknown'),
                    'component': event.get('component', 'unknown'),
                    'accepted': event.get('accepted', True),
                    'age_sec': event.get('age_sec'),
                })
            for event in self.covariance_events.list():
                timing_events.append({
                    't': event['t'],
                    'stage': event.get('source', 'covariance'),
                    'sensor': 'ekf',
                    'component': 'covariance',
                    'age_sec': event.get('plotter_latency_sec'),
                })
            timing_events = sorted(timing_events, key=lambda event: event.get('t', 0.0))
            timing_events = self.thin_events(timing_events)
        if not timing_events:
            return

        fig, axes = plt.subplots(5, 1, figsize=(16, 17))
        fig.suptitle('EKF Timing Diagnostics', fontsize=14, fontweight='bold')

        def event_name(event):
            stage = event.get('stage', 'unknown')
            sensor = event.get('sensor')
            component = event.get('component')
            if stage == 'prediction':
                return 'prediction'
            if sensor and component:
                return f'{sensor}:{component}'
            return sensor or stage

        names = sorted({event_name(event) for event in timing_events})
        name_to_y = {name: idx for idx, name in enumerate(names)}
        cmap = plt.get_cmap('tab10')
        color_map = {name: cmap(idx % 10) for idx, name in enumerate(names)}

        grouped_times = {}
        for event in timing_events:
            name = event_name(event)
            grouped_times.setdefault(name, []).append(event['t'])

            age = event.get('age_sec')
            if age is not None:
                try:
                    age = float(age)
                except (TypeError, ValueError):
                    age = None
            if age is not None and np.isfinite(age):
                axes[0].scatter(
                    event['t'], age,
                    c=[color_map[name]], s=22,
                    marker='o' if event.get('accepted', True) else 'x',
                    alpha=0.8,
                    label=name if name not in axes[0].get_legend_handles_labels()[1] else None
                )

            axes[3].scatter(
                event['t'], name_to_y[name],
                c=[color_map[name]], s=20,
                marker='o' if event.get('accepted', True) else 'x',
                alpha=0.75
            )

        axes[0].set_ylabel('Processing age (s)', fontsize=11)
        axes[0].set_title('Message stamp age when EKF processed the event', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc='upper right', fontsize=8, ncol=3)

        for name, times in grouped_times.items():
            times = np.array(sorted(t for t in times if np.isfinite(t)), dtype=float)
            if len(times) <= 1:
                continue
            dt = np.diff(times)
            axes[1].scatter(
                times[1:], dt,
                c=[color_map[name]], s=18, alpha=0.7,
                label=name
            )
        axes[1].set_ylabel('Event dt (s)', fontsize=11)
        axes[1].set_title('Spacing between consecutive events of each source', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, loc='upper right', fontsize=8, ncol=3)

        prediction_events = [
            event for event in timing_events
            if event.get('stage') == 'prediction' and event.get('dt') is not None
        ]
        if prediction_events:
            times = []
            dts = []
            observed_times = []
            observed_dts = []
            for event in prediction_events:
                try:
                    dt = float(event.get('dt'))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(dt):
                    times.append(event['t'])
                    dts.append(dt)
                try:
                    observed_dt = float(event.get('observed_dt'))
                except (TypeError, ValueError):
                    observed_dt = None
                if observed_dt is not None and np.isfinite(observed_dt):
                    observed_times.append(event['t'])
                    observed_dts.append(observed_dt)
            axes[2].plot(
                times, dts, color='tab:blue', linewidth=1.4,
                label='fixed integration dt'
            )
            if observed_dts:
                axes[2].plot(
                    observed_times, observed_dts,
                    color='tab:orange', linewidth=1.0, alpha=0.8,
                    label='observed source-message spacing'
                )
            axes[2].legend(loc='upper right', fontsize=8)
        else:
            axes[2].annotate(
                'No prediction dt samples on /ekf/debug/timing',
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        axes[2].set_ylabel('Prediction dt (s)', fontsize=11)
        axes[2].set_title(
            'Fixed prediction dt versus observed message spacing',
            fontsize=11
        )
        axes[2].grid(True, alpha=0.3)

        axes[3].set_yticks(list(name_to_y.values()))
        axes[3].set_yticklabels(list(name_to_y.keys()), fontsize=8)
        axes[3].set_ylabel('Event source', fontsize=11)
        axes[3].set_title('EKF event raster by stamp time', fontsize=11)
        axes[3].grid(True, axis='x', alpha=0.3)

        ages_by_name = {}
        for event in timing_events:
            age = event.get('age_sec')
            if age is None:
                continue
            try:
                age = float(age)
            except (TypeError, ValueError):
                continue
            if np.isfinite(age):
                ages_by_name.setdefault(event_name(event), []).append(age)
        for name, ages in ages_by_name.items():
            axes[4].hist(
                ages, bins=40, histtype='step',
                linewidth=1.2, color=color_map[name],
                label=name
            )
        axes[4].set_xlabel('Processing age (s)', fontsize=11)
        axes[4].set_ylabel('Count', fontsize=11)
        axes[4].set_title('Distribution of processing age', fontsize=11)
        axes[4].grid(True, alpha=0.3)
        handles, labels = axes[4].get_legend_handles_labels()
        if handles:
            axes[4].legend(handles, labels, loc='upper right', fontsize=8, ncol=3)

        for ax in axes[:4]:
            ax.set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'timing_diagnostics_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_aruco_latency_profile(self, prefix):
        """Correlate camera, detector, ROS-boundary, and EKF timing records."""
        camera_events = self.thin_events(self.camera_timing_events.list())
        detector_events = self.thin_events(self.aruco_detector_timing_events.list())
        ekf_events = self.thin_events(self.aruco_ekf_timing_events.list())

        def stamp_key(event):
            try:
                return round(float(event.get('stamp')), 6)
            except (TypeError, ValueError):
                return None

        def finite_value(event, key):
            try:
                value = float(event.get(key))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def plot_metric(ax, events, key, label, **kwargs):
            points = [
                (event['t'], finite_value(event, key))
                for event in events
            ]
            points = [(t, value) for t, value in points if value is not None]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    label=label, linewidth=1.1, **kwargs
                )

        camera_by_stamp = {
            stamp_key(event): event
            for event in camera_events
            if stamp_key(event) is not None
        }
        detector_by_stamp = {
            stamp_key(event): event
            for event in detector_events
            if stamp_key(event) is not None
        }

        image_ros_points = []
        left_image_ros_points = []
        right_image_ros_points = []
        for detector_event in detector_events:
            camera_event = camera_by_stamp.get(stamp_key(detector_event))
            if camera_event is None:
                continue
            callback_stamp = finite_value(detector_event, 'callback_start_stamp')
            publish_stamp = finite_value(camera_event, 'publish_complete_stamp')
            detector_mode = detector_event.get('pose_estimation_mode', 'stereo')
            if (
                detector_mode == 'stereo' and
                callback_stamp is not None and publish_stamp is not None
            ):
                image_ros_points.append(
                    (detector_event['t'], (callback_stamp - publish_stamp) * 1000.0)
                )
            for points, receipt_key, publish_key in (
                (
                    left_image_ros_points,
                    'left_subscriber_receipt_stamp',
                    'left_publish_complete_stamp',
                ),
                (
                    right_image_ros_points,
                    'right_subscriber_receipt_stamp',
                    'right_publish_complete_stamp',
                ),
            ):
                receipt_stamp = finite_value(detector_event, receipt_key)
                side_publish_stamp = finite_value(camera_event, publish_key)
                if (
                    receipt_stamp is not None and receipt_stamp > 0.0 and
                    side_publish_stamp is not None and side_publish_stamp > 0.0
                ):
                    points.append(
                        (
                            detector_event['t'],
                            (receipt_stamp - side_publish_stamp) * 1000.0,
                        )
                    )

        pose_ros_points = []
        for ekf_event in ekf_events:
            detector_event = detector_by_stamp.get(stamp_key(ekf_event))
            if detector_event is None:
                continue
            marker_id = ekf_event.get('marker_id')
            publish_stamps = detector_event.get(
                'marker_pose_publish_complete_stamps', {}
            )
            pose_publish_stamp = publish_stamps.get(str(marker_id))
            callback_stamp = finite_value(ekf_event, 'callback_start_stamp')
            try:
                pose_publish_stamp = float(pose_publish_stamp)
            except (TypeError, ValueError):
                pose_publish_stamp = None
            if (
                callback_stamp is not None and
                pose_publish_stamp is not None and
                np.isfinite(pose_publish_stamp)
            ):
                pose_ros_points.append(
                    (ekf_event['t'], (callback_stamp - pose_publish_stamp) * 1000.0)
                )

        fig, axes = plt.subplots(5, 1, figsize=(17, 22))
        fig.suptitle(
            'ArUco End-to-End Latency and Bottleneck Profile',
            fontsize=14, fontweight='bold'
        )
        stereo_detector_events = [
            event for event in detector_events
            if event.get('pose_estimation_mode', 'stereo') == 'stereo'
        ]
        monocular_detector_events = [
            event for event in detector_events
            if event.get('pose_estimation_mode') == 'monocular'
        ]

        # Cross-process boundaries. The two ROS-overhead lines subtract the
        # producer's publish-complete time from the consumer callback start.
        plot_metric(axes[0], camera_events, 'capture_read_ms', 'V4L2 read/decode')
        plot_metric(axes[0], camera_events, 'camera_pipeline_ms', 'Camera frame→publish')
        plot_metric(
            axes[0], detector_events, 'source_to_callback_ms',
            'Frame→detector callback'
        )
        plot_metric(
            axes[0], ekf_events, 'source_to_callback_ms',
            'Frame→EKF callback'
        )
        if image_ros_points:
            axes[0].plot(
                [point[0] for point in image_ros_points],
                [point[1] for point in image_ros_points],
                label='ROS images→synchronized callback', linewidth=1.1
            )
        for points, label in (
            (left_image_ros_points, 'ROS left image transport/queue'),
            (right_image_ros_points, 'ROS right image transport/queue'),
        ):
            if points:
                axes[0].plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    label=label, linewidth=0.9, alpha=0.8
                )
        plot_metric(
            axes[0], stereo_detector_events, 'sync_dispatch_ms',
            'Stereo sync dispatch'
        )
        if pose_ros_points:
            axes[0].plot(
                [point[0] for point in pose_ros_points],
                [point[1] for point in pose_ros_points],
                label='ROS pose transport/queue', linewidth=1.1
            )
        axes[0].axhline(0.0, color='black', linewidth=0.7, alpha=0.5)
        axes[0].set_title('Cross-node latency boundaries')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=8, ncol=3)

        detector_metrics = [
            ('cv_bridge_ms', 'cv_bridge conversion/share'),
            ('grayscale_ms', 'Grayscale'),
            ('rectify_ms', 'Undistort/rectify'),
            ('visualization_prepare_ms', 'Visualization prep'),
            ('detect_left_ms', 'Detect left'),
            ('pixel_gate_ms', 'Pixel gate'),
            ('result_and_pose_publish_ms', 'Transform/result/publish'),
        ]
        for key, label in detector_metrics:
            plot_metric(axes[1], detector_events, key, label)
        for key, label in (
            ('detect_right_ms', 'Detect right'),
            ('stereo_match_ms', 'Stereo match'),
            ('raw_pose_ms', 'Raw monocular diagnostics'),
            ('stereo_pnp_ms', 'Sequential stereo PnP'),
        ):
            plot_metric(axes[1], stereo_detector_events, key, label)
        plot_metric(
            axes[1], monocular_detector_events, 'stereo_pnp_ms',
            'ArUco single-marker pose'
        )
        axes[1].set_title('ArUco/OpenCV procedure stage durations')
        axes[1].set_ylabel('Duration (ms)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=8, ncol=4)

        ekf_metrics = [
            ('stale_check_ms', 'Stale check'),
            ('transform_ms', 'Camera→landpad transform'),
            ('validation_ms', 'Validation'),
            ('measurement_publish_ms', 'Measurement publish'),
            ('initialization_ms', 'Initialization'),
            ('xy_update_ms', 'EKF x/y update'),
            ('z_update_ms', 'EKF z update'),
            ('yaw_update_ms', 'EKF yaw update'),
            ('estimate_publish_ms', 'Estimate/TF publish'),
            ('callback_total_ms', 'Whole EKF callback'),
        ]
        for key, label in ekf_metrics:
            plot_metric(axes[2], ekf_events, key, label)
        axes[2].set_title('EKF ArUco callback stage durations')
        axes[2].set_ylabel('Duration (ms)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right', fontsize=8, ncol=4)

        distribution_specs = [
            (detector_events, 'cv_bridge_ms', 'cv_bridge'),
            (detector_events, 'rectify_ms', 'rectify'),
            (detector_events, 'detect_left_ms', 'detect L'),
            (stereo_detector_events, 'detect_right_ms', 'detect R'),
            (stereo_detector_events, 'raw_pose_ms', 'raw pose diag'),
            (stereo_detector_events, 'stereo_pnp_ms', 'stereo PnP'),
            (monocular_detector_events, 'stereo_pnp_ms', 'mono ArUco pose'),
            (detector_events, 'detector_total_ms', 'detector total'),
            (ekf_events, 'callback_total_ms', 'EKF callback'),
        ]
        distributions = []
        distribution_labels = []
        for events, key, label in distribution_specs:
            values = [
                finite_value(event, key)
                for event in events
            ]
            values = [value for value in values if value is not None]
            if values:
                distributions.append(values)
                distribution_labels.append(label)
        if distributions:
            # Matplotlib first coerces its input with np.asarray().  Recent
            # NumPy versions reject a plain list of differently sized series
            # as a ragged array, so preserve each latency series as one object.
            boxplot_data = np.empty(len(distributions), dtype=object)
            for index, values in enumerate(distributions):
                boxplot_data[index] = np.asarray(values, dtype=float)
            axes[3].boxplot(
                boxplot_data, labels=distribution_labels,
                showfliers=True, whis=(5, 95)
            )
        axes[3].set_title('Stage latency distributions (boxes; whiskers P5–P95)')
        axes[3].set_ylabel('Duration (ms)')
        axes[3].tick_params(axis='x', labelrotation=25)
        axes[3].grid(True, axis='y', alpha=0.3)

        ordered_detector = sorted(
            detector_events,
            key=lambda event: float(event.get('stamp', 0.0))
        )
        plot_metric(
            axes[4], ordered_detector, 'callback_total_ms',
            'Detector callback load'
        )
        plot_metric(
            axes[4], ordered_detector, 'source_to_callback_ms',
            'Frame age at detector callback'
        )
        if len(ordered_detector) > 1:
            stamps = np.array([
                float(event.get('stamp', 0.0))
                for event in ordered_detector
            ])
            times = np.array([event['t'] for event in ordered_detector])
            axes[4].plot(
                times[1:], np.diff(stamps) * 1000.0,
                label='Processed-frame spacing', linewidth=1.0
            )

            sequences = np.array([
                int(event.get('sequence', 0))
                for event in ordered_detector
            ])
            dropped = np.maximum(np.diff(sequences) - 1, 0)
            if np.any(dropped > 0):
                drop_axis = axes[4].twinx()
                drop_axis.scatter(
                    times[1:][dropped > 0], dropped[dropped > 0],
                    color='tab:red', marker='x', s=30,
                    label='Skipped camera sequences'
                )
                drop_axis.set_ylabel('Skipped frames')
                handles, labels = drop_axis.get_legend_handles_labels()
                if handles:
                    drop_axis.legend(handles, labels, loc='upper left', fontsize=8)
        axes[4].set_title(
            'Load, input spacing, and queue-drop evidence '
            '(callback load > frame spacing cannot keep up)'
        )
        axes[4].set_ylabel('Time (ms)')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(loc='upper right', fontsize=8, ncol=3)

        for ax in (axes[0], axes[1], axes[2], axes[4]):
            ax.set_xlabel('Time (s)')

        plt.tight_layout()
        filepath = os.path.join(
            self.save_dir, f'aruco_latency_profile_{prefix}.png'
        )
        self.save_figure(fig, filepath)

    def plot_marker_quality(self, prefix):
        configured_markers = set(self.aruco_marker_data.keys())
        events = [
            event for event in self.thin_events(self.marker_quality_events.list())
            if event.get('marker_id') in configured_markers
        ]
        if not events:
            return

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('ArUco Detector Quality', fontsize=14, fontweight='bold')

        marker_legend = {}
        grouped_span = {}
        grouped_reproj = {}
        grouped_range = {}
        min_spans = {}
        any_rejected = False
        for event in events:
            marker_id = event.get('marker_id')
            color = self.marker_colors.get(marker_id, 'black')
            label = f'ArUco {marker_id}'
            accepted = bool(event.get('accepted', False))
            key = (label, color, accepted)
            any_rejected = any_rejected or not accepted

            pixel_span = event.get('pixel_span_diag')
            min_span = event.get('min_pixel_span_diag')
            reproj = event.get('reprojection_error_px')
            range_m = event.get('range_m')

            if pixel_span is not None:
                grouped_span.setdefault(key, [[], []])
                grouped_span[key][0].append(event['t'])
                grouped_span[key][1].append(pixel_span)
                if label not in marker_legend:
                    marker_legend[label] = self.legend_marker(color, label, marker='o')
            if min_span is not None and float(min_span) > 0.0:
                min_spans[(label, color)] = float(min_span)
            if reproj is not None:
                grouped_reproj.setdefault(key, [[], []])
                grouped_reproj[key][0].append(event['t'])
                grouped_reproj[key][1].append(reproj)
            if range_m is not None:
                grouped_range.setdefault(key, [[], []])
                grouped_range[key][0].append(event['t'])
                grouped_range[key][1].append(range_m)

        for grouped, ax in (
            (grouped_span, axes[0]),
            (grouped_reproj, axes[1]),
            (grouped_range, axes[2]),
        ):
            for (label, color, accepted), (times, values) in grouped.items():
                ax.scatter(
                    times, values,
                    c=color,
                    s=24,
                    marker='o' if accepted else 'x',
                    alpha=0.75 if accepted else 0.95,
                    linewidths=0.8 if not accepted else 0.0
                )

        min_span_handles = []
        for (label, color), min_span in sorted(min_spans.items()):
            line = axes[0].axhline(
                min_span,
                color=color,
                linestyle='--',
                linewidth=1.0,
                alpha=0.55,
                label=f'{label} min span'
            )
            min_span_handles.append(line)

        axes[0].set_ylabel('Diagonal span (px)', fontsize=11)
        axes[1].set_ylabel('Reprojection error (px)', fontsize=11)
        axes[2].set_ylabel('Estimated range (m)', fontsize=11)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        if marker_legend:
            handles = list(marker_legend.values())
            labels = [handle.get_label() for handle in handles]
            status_handles = [self.legend_marker('black', 'Accepted detection', marker='o', markersize=6)]
            if any_rejected:
                status_handles.append(
                    self.legend_marker('black', 'Rejected detection', marker='x', markersize=7)
                )
            handles.extend(status_handles + min_span_handles)
            labels.extend([handle.get_label() for handle in status_handles + min_span_handles])
            axes[0].legend(handles, labels, loc='upper right', fontsize=9, ncol=3,
                           title='Color: marker | Circle/X: accepted/rejected | Dashed: min span')
        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'aruco_marker_quality_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_covariance(self, prefix):
        events = self.thin_events(self.covariance_events.list())
        if not events:
            return

        times = np.array([event['t'] for event in events])
        p_diag = np.array([event.get('p_diag', []) for event in events], dtype=float)
        if p_diag.ndim != 2 or p_diag.shape[1] < 10:
            rospy.logwarn("[PLOTTER] Covariance debug topic missing expected 10-state diagonal")
            return

        cov_cfg = self.config.get('plotter', {}).get('covariance', {})
        mode = cov_cfg.get('mode', 'std')
        yscale = cov_cfg.get('yscale', 'log')
        if mode == 'variance':
            plot_values = p_diag.copy()
            value_label = 'Variance'
            title_suffix = 'variance'
        else:
            plot_values = np.sqrt(np.maximum(p_diag, 0.0))
            value_label = 'Std dev'
            title_suffix = 'standard deviation'

        if yscale == 'log':
            plot_values[plot_values <= 0.0] = np.nan

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(
            f'EKF Covariance Diagonal P ({title_suffix})',
            fontsize=14, fontweight='bold'
        )

        groups = [
            ('Position uncertainty', [0, 1, 2], ['x', 'y', 'z']),
            ('Velocity uncertainty', [3, 4, 5], ['vx', 'vy', 'vz']),
            ('Quaternion uncertainty', [6, 7, 8, 9], ['qx', 'qy', 'qz', 'qw']),
        ]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        for ax, (title, indices, labels) in zip(axes, groups):
            for idx, label, color in zip(indices, labels, colors):
                ax.plot(times, plot_values[:, idx], linewidth=1.5, label=label, color=color)
            if yscale == 'log':
                ax.set_yscale('log')
            ax.set_ylabel(value_label, fontsize=11)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9, ncol=2)

        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'covariance_{prefix}.png')
        self.save_figure(fig, filepath)

    def plot_selected_gains(self, prefix):
        events = [event for event in self.thin_events(self.kalman_gain_events.list())
                  if event.get('accepted', True)]
        if not events:
            return

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle('Selected Kalman Gain Terms', fontsize=14, fontweight='bold')

        axis_specs = [
            ('K[x <- meas_x]', 'x'),
            ('K[y <- meas_y]', 'y'),
            ('K[z <- meas_z]', 'z'),
            ('K[yaw meas corrected]', 'yaw')
        ]

        legend_entries = {}
        for ax, (label, axis_key) in zip(axes, axis_specs):
            grouped_points = {}
            for event in events:
                value = self.selected_gain_value(event, axis_key)
                if value is None:
                    continue

                sensor_name = event.get('sensor', 'unknown')
                marker_id = event.get('marker_id')
                color = self.event_color(sensor_name, marker_id)
                legend_label = self.event_label(sensor_name, marker_id)
                key = (legend_label, color)
                grouped_points.setdefault(key, [[], []])
                grouped_points[key][0].append(event['t'])
                grouped_points[key][1].append(value)

                if legend_label not in legend_entries:
                    legend_entries[legend_label] = self.legend_marker(color, legend_label)

            for (legend_label, color), (times, values) in grouped_points.items():
                ax.scatter(times, values, c=color, s=24, marker='o', alpha=0.8)

            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)

        if legend_entries:
            axes[0].legend(
                legend_entries.values(), legend_entries.keys(),
                loc='upper right', fontsize=9, ncol=2
            )
        axes[-1].set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, f'kalman_gain_{prefix}.png')
        self.save_figure(fig, filepath)

    # ---------------------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------------------

    def start_detached_save(self):
        """Fork a detached saver so roslaunch can exit without killing plot generation.

        Older ROS/roslaunch versions do not support per-node sigint/sigterm
        timeout attributes. If final plotting takes longer than roslaunch's
        built-in grace period, roslaunch sends SIGTERM/SIGKILL to this node.
        The detached child is placed in a new session and writes progress to a
        log file, while the parent returns immediately to roslaunch.
        """
        log_path = self.shutdown_save_log or os.path.join(
            self.save_dir, 'plotter_shutdown_save.log'
        )
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

        try:
            pid = os.fork()
        except AttributeError:
            # os.fork is not available on non-POSIX systems. ROS deployments for
            # this project are Linux-based, but keep a safe synchronous fallback.
            print("[PLOTTER] Detached save unavailable on this OS; saving in-process")
            return self.save_and_report_in_process()
        except OSError as exc:
            print(f"[PLOTTER] Could not fork detached saver: {exc}")
            print("[PLOTTER] Falling back to in-process save")
            return self.save_and_report_in_process()

        if pid > 0:
            self.write_shutdown_status('detached_started', pid=pid)
            print(f"[PLOTTER] Detached saver started with PID {pid}", flush=True)
            print(f"[PLOTTER] Saver log: {log_path}", flush=True)
            print(
                "[PLOTTER] roslaunch may exit now; plots will continue "
                "saving in the detached process.",
                flush=True
            )
            return None

        # Child process.
        exit_code = 0
        try:
            try:
                os.setsid()
            except Exception:
                pass

            # Do not let terminal Ctrl+C or roslaunch escalation intended for
            # the parent interrupt the saver.
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            except Exception:
                pass

            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.dup2(fd, 1)
                os.dup2(fd, 2)
            finally:
                if fd > 2:
                    os.close(fd)

            print("\n" + "=" * 70, flush=True)
            print(f"[PLOTTER CHILD] Detached save started. PID={os.getpid()}", flush=True)
            print(f"[PLOTTER CHILD] Saving to: {self.save_dir}", flush=True)
            print("=" * 70, flush=True)

            result = self.save_all_plots('final')
            self.print_save_result(result)
            print("[PLOTTER CHILD] Detached save finished", flush=True)
        except Exception:
            exit_code = 1
            import traceback
            traceback.print_exc()
        finally:
            self.write_shutdown_status(
                'complete' if exit_code == 0 else 'failed',
                pid=os.getpid(),
                error=None if exit_code == 0 else 'detached saver failed'
            )
            try:
                plt.close('all')
            except Exception:
                pass
            os._exit(exit_code)

    def print_save_result(self, result):
        if result['failed_plots']:
            print("[PLOTTER] Some plots failed:", flush=True)
            for label, exc in result['failed_plots']:
                print(f"  - {label}: {exc}", flush=True)

        if result['timeout_hit']:
            print("[PLOTTER] Partial plot save: shutdown budget was reached.", flush=True)
            if result['skipped_timeout']:
                print("[PLOTTER] Skipped due to timeout:", flush=True)
                for label in result['skipped_timeout']:
                    print(f"  - {label}", flush=True)
        elif result['failed_plots']:
            print("[PLOTTER] Plot save completed with failures.", flush=True)
        else:
            print("[PLOTTER] All available plots saved successfully!", flush=True)

        print(f"[PLOTTER] Generated: {result['plot_count']} plot(s)", flush=True)
        print(f"[PLOTTER] Location: {self.save_dir}", flush=True)

    def save_and_report_in_process(self):
        self.write_shutdown_status('saving_in_process', pid=os.getpid())
        try:
            result = self.save_all_plots('final')
            self.print_save_result(result)
            self.write_shutdown_status('complete', pid=os.getpid())
            return result
        except Exception as exc:
            self.write_shutdown_status(
                'failed', pid=os.getpid(), error=str(exc)
            )
            raise

    def save_and_exit(self):
        if self.shutdown_complete:
            return

        self.shutdown_complete = True
        print("\n" + "=" * 50)
        print("[PLOTTER] Shutting down, saving plots...")
        print("=" * 50)

        has_any_data = (
            self.ekf_data.has_data or
            self.dr_data.has_data or
            self.thermal_data.has_data or
            self.laser_data.has_data or
            self.uwb_data.has_data or
            any(data.has_data for data in self.aruco_marker_data.values()) or
            len(self.innovation_events) > 0 or
            len(self.covariance_events) > 0 or
            len(self.kalman_gain_events) > 0 or
            len(self.marker_quality_events) > 0 or
            len(self.timing_events) > 0 or
            len(self.camera_timing_events) > 0 or
            len(self.aruco_detector_timing_events) > 0 or
            len(self.aruco_ekf_timing_events) > 0
        )
        if not has_any_data:
            print("[PLOTTER] No data collected, nothing to save")
            self.write_shutdown_status('no_data', pid=os.getpid())
            return

        try:
            if self.detached_shutdown_save:
                self.start_detached_save()
            else:
                self.save_and_report_in_process()
        except Exception as e:
            self.write_shutdown_status(
                'failed', pid=os.getpid(), error=str(e)
            )
            print(f"[PLOTTER] Error saving plots: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        print("[PLOTTER] Running. Press Ctrl+C to stop and save plots.")
        rospy.spin()


def main():
    plotter = None
    try:
        plotter = EKFPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"[PLOTTER] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if plotter:
            plotter.save_and_exit()


if __name__ == '__main__':
    main()
