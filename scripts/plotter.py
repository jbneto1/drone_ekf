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
        return {
            'times': np.array(list(self.times)),
            'x': np.array([v if v is not None else np.nan for v in self.x]),
            'y': np.array([v if v is not None else np.nan for v in self.y]),
            'z': np.array([v if v is not None else np.nan for v in self.z]),
            'yaw': np.array([v if v is not None else np.nan for v in self.yaw]),
        }


class EventBuffer:
    """Stores JSON diagnostic events for later plotting."""

    def __init__(self, max_points=10000):
        self.events = deque(maxlen=max_points)

    def add(self, event):
        self.events.append(event)

    def __len__(self):
        return len(self.events)

    def list(self):
        return list(self.events)


class EKFPlotter:
    """Plot combined EKF measurement and diagnostic topics."""

    def __init__(self):
        rospy.init_node('ekf_plotter', anonymous=True)

        self.load_config()
        max_pts = self.config.get('plotter', {}).get('max_points', 10000)

        self.ekf_data = SensorData(max_pts)
        self.dr_data = SensorData(max_pts)
        self.aruco_data = SensorData(max_pts)
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

        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000

        self.start_time = None
        self.shutdown_complete = False

        self.setup_subscribers()

        rospy.loginfo(f"[PLOTTER] Initialized. Saving to: {self.save_dir}")
        rospy.loginfo("[PLOTTER] Waiting for data...")

    def load_config(self):
        """Load configuration from YAML."""
        config_path = rospy.get_param('~config_file', '')
        if not config_path:
            pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_path, 'config', 'ekf_params.yaml')

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            rospy.loginfo(f"[PLOTTER] Loaded config from: {config_path}")

        self.save_dir = self.config.get('plotter', {}).get('save_dir', '/tmp/ekf_plots')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
            topics.get('aruco_measurement', '/ekf/measurements/aruco'),
            PoseStamped, self.aruco_callback, queue_size=1
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
            topics.get('aruco_measurement', '/ekf/measurements/aruco') + '/marker_'
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

    def relative_time_from_sec(self, stamp_sec):
        if self.start_time is None:
            self.start_time = stamp_sec
        return stamp_sec - self.start_time

    def get_time(self, stamp):
        return self.relative_time_from_sec(stamp.to_sec())

    def parse_debug_event(self, msg):
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return None
        stamp_sec = float(event.get('stamp', rospy.Time.now().to_sec()))
        event['t'] = self.relative_time_from_sec(stamp_sec)
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

    def aruco_callback(self, msg):
        if self.shutdown_complete:
            return
        self.aruco_data.add_pose(self.get_time(msg.header.stamp), msg.pose)

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
        aruco = self.aruco_data.get_arrays()
        thermal = self.thermal_data.get_arrays()
        laser = self.laser_data.get_arrays()
        uwb = self.uwb_data.get_arrays()
        aruco_markers = {mid: data.get_arrays() for mid, data in self.aruco_marker_data.items()}

        if should_unwrap:
            ekf['yaw'] = self.unwrap_yaw(ekf['yaw'])
            dr['yaw'] = self.unwrap_yaw(dr['yaw'])
            aruco['yaw'] = self.unwrap_yaw(aruco['yaw'])
            thermal['yaw'] = self.unwrap_yaw(thermal['yaw'])
            for marker in aruco_markers.values():
                marker['yaw'] = self.unwrap_yaw(marker['yaw'])

        plot_count = 0

        if self.ekf_data.has_data and len(ekf['times']) > 10:
            self.plot_combined_measurements(
                ekf, dr, aruco, thermal, laser, uwb, aruco_markers, yaw_cfg, prefix
            )
            plot_count += 1

        if len(self.innovation_events) > 0:
            self.plot_innovations(prefix)
            plot_count += 1
            self.plot_nis(prefix)
            plot_count += 1

        if len(self.covariance_events) > 0:
            self.plot_covariance(prefix)
            plot_count += 1

        if len(self.kalman_gain_events) > 0:
            self.plot_selected_gains(prefix)
            plot_count += 1

        if len(self.marker_quality_events) > 0:
            self.plot_marker_quality(prefix)
            plot_count += 1

        if plot_count == 0:
            rospy.logwarn("[PLOTTER] No data available for plotting")
        else:
            elapsed = time.time() - start_wall
            rospy.loginfo(
                f"[PLOTTER] Generated {plot_count} plot(s) in {self.save_dir} "
                f"({elapsed:.1f} s)"
            )

    def plot_combined_measurements(self, ekf, dr, aruco, thermal, laser, uwb, aruco_markers,
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
                    t_plot, v_plot = self.thin_series(marker_data['times'][valid], marker_data[key][valid])
                    scatter = ax.scatter(
                        t_plot, v_plot,
                        c=color, s=30, marker='o', alpha=0.75,
                        label=label_text, edgecolors=color, linewidths=0.4, zorder=4
                    )
                    if label_text not in all_labels:
                        all_handles.append(scatter)
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
            axes[0].legend(all_handles, all_labels, loc='upper right', fontsize=9, ncol=2)

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

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('EKF Covariance Diagonal P', fontsize=14, fontweight='bold')

        groups = [
            ('Position covariance', [0, 1, 2], ['Pxx', 'Pyy', 'Pzz']),
            ('Velocity covariance', [3, 4, 5], ['Pvx', 'Pvy', 'Pvz']),
            ('Quaternion covariance', [6, 7, 8, 9], ['Pqx', 'Pqy', 'Pqz', 'Pqw']),
        ]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        for ax, (title, indices, labels) in zip(axes, groups):
            for idx, label, color in zip(indices, labels, colors):
                ax.plot(times, p_diag[:, idx], linewidth=1.5, label=label, color=color)
            ax.set_ylabel('Variance', fontsize=11)
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

    def save_and_exit(self):
        if self.shutdown_complete:
            return

        self.shutdown_complete = True
        print("\n" + "=" * 50)
        print("[PLOTTER] Shutting down, saving plots...")
        print("=" * 50)

        if not self.ekf_data.has_data and len(self.innovation_events) == 0:
            print("[PLOTTER] No data collected, nothing to save")
            return

        try:
            self.save_all_plots('final')
            print("[PLOTTER] All plots saved successfully!")
            print(f"[PLOTTER] Location: {self.save_dir}")
        except Exception as e:
            print(f"[PLOTTER] Error saving plots: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        print("[PLOTTER] Running. Press Ctrl+C to stop and save plots.")
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self.save_and_exit()


def main():
    plotter = None
    try:
        plotter = EKFPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        if plotter:
            plotter.save_and_exit()
    except Exception as e:
        print(f"[PLOTTER] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if plotter:
            plotter.save_and_exit()


if __name__ == '__main__':
    main()
