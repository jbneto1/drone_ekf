#!/usr/bin/env python3
"""
EKF Plotter V4 - Modular Sensor Visualization

Visualization-only plotter that dynamically creates plots based on enabled sensors.

Plots:
1. Per-sensor XYZ+Yaw vs Time (sensor measurement vs EKF vs dead reckoning)
2. Combined comparison plot (all sensors + EKF + dead reckoning)

Removes: XY trajectory, velocity plots (per requirements)
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os
from datetime import datetime
import tf.transformations as tft
import json
import yaml


class SensorData:
    """Container for sensor time series data."""
    def __init__(self, max_points=10000):
        self.times = deque(maxlen=max_points)
        self.x = deque(maxlen=max_points)
        self.y = deque(maxlen=max_points)
        self.z = deque(maxlen=max_points)
        self.yaw = deque(maxlen=max_points)  # degrees
        self.has_data = False
    
    def add_pose(self, t, pose):
        """Add data from PoseStamped message."""
        self.times.append(t)
        self.x.append(pose.position.x)
        self.y.append(pose.position.y)
        self.z.append(pose.position.z)
        
        q = [pose.orientation.x, pose.orientation.y, 
             pose.orientation.z, pose.orientation.w]
        euler = tft.euler_from_quaternion(q)
        self.yaw.append(np.degrees(euler[2]))
        self.has_data = True
    
    def add_point(self, t, point):
        """Add data from PointStamped message (partial data)."""
        self.times.append(t)
        self.x.append(point.x if not np.isnan(point.x) else None)
        self.y.append(point.y if not np.isnan(point.y) else None)
        self.z.append(point.z if not np.isnan(point.z) else None)
        self.yaw.append(None)  # Point messages don't have orientation
        self.has_data = True
    
    def get_arrays(self):
        """Return numpy arrays for plotting."""
        return {
            'times': np.array(list(self.times)),
            'x': np.array([v if v is not None else np.nan for v in self.x]),
            'y': np.array([v if v is not None else np.nan for v in self.y]),
            'z': np.array([v if v is not None else np.nan for v in self.z]),
            'yaw': np.array([v if v is not None else np.nan for v in self.yaw])
        }


class EKFPlotter:
    """Modular plotter for EKF analysis with dynamic sensor support."""
    
    def __init__(self):
        rospy.init_node('ekf_plotter', anonymous=True)
        
        # Load config
        self.load_config()
        
        # Data storage
        max_pts = self.config.get('plotter', {}).get('max_points', 10000)
        
        self.ekf_data = SensorData(max_pts)
        self.dr_data = SensorData(max_pts)  # Dead reckoning
        self.aruco_data = SensorData(max_pts)
        self.thermal_data = SensorData(max_pts)
        self.laser_data = SensorData(max_pts)
        self.uwb_data = SensorData(max_pts)

        marker_cfg = self.config.get('sensors', {}).get('aruco', {}).get('markers', {})
        marker_ids = sorted(int(mid) for mid in marker_cfg.keys()) if marker_cfg else [363, 682, 417]
        self.aruco_marker_data = {mid: SensorData(max_pts) for mid in marker_ids}
        
        # Sensor status tracking
        self.sensor_config = {
            'aruco': {'enabled': False, 'active': False},
            'laser': {'enabled': False, 'active': False},
            'uwb': {'enabled': False, 'active': False},
            'thermal': {'enabled': False, 'active': False},
            'process_model': 'unknown'
        }
        
        # Timing
        self.start_time = None
        self.shutdown_complete = False
        
        # Setup subscribers
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
        
        # EKF outputs
        rospy.Subscriber(
            topics.get('ekf_odom', '/ekf/odom'),
            Odometry, self.ekf_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get('dead_reckoning', '/ekf/dead_reckoning'),
            PoseStamped, self.dr_callback, queue_size=1
        )
        
        # Sensor measurements
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
        
        # Sensor status
        rospy.Subscriber(
            topics.get('sensor_status', '/ekf/sensor_status'),
            String, self.status_callback, queue_size=1
        )

        # EKF-transformed per-marker measurements, not raw camera-frame detections.
        aruco_prefix = topics.get('aruco_marker_measurement_prefix',
                                  topics.get('aruco_measurement', '/ekf/measurements/aruco') + '/marker_')
        for marker_id in sorted(self.aruco_marker_data.keys()):
            rospy.Subscriber(f'{aruco_prefix}{marker_id}', PoseStamped,
                             lambda msg, mid=marker_id: self.aruco_marker_callback(msg, mid),
                             queue_size=1)
    
    def get_time(self, stamp):
        """Get relative time from start."""
        if self.start_time is None:
            self.start_time = stamp.to_sec()
        return stamp.to_sec() - self.start_time
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def ekf_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.ekf_data.add_pose(t, msg.pose.pose)
    
    def dr_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.dr_data.add_pose(t, msg.pose)
    
    def aruco_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.aruco_data.add_pose(t, msg.pose)
    
    def thermal_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.thermal_data.add_pose(t, msg.pose)

    def laser_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.laser_data.add_point(t, msg.point)
    
    def uwb_callback(self, msg):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        self.uwb_data.add_point(t, msg.point)
    
    def status_callback(self, msg):
        """Update sensor configuration from EKF node."""
        try:
            self.sensor_config = json.loads(msg.data)
        except json.JSONDecodeError:
            pass
    
    def aruco_marker_callback(self, msg, marker_id):
        if self.shutdown_complete:
            return
        t = self.get_time(msg.header.stamp)
        if marker_id not in self.aruco_marker_data:
            self.aruco_marker_data[marker_id] = SensorData(self.config.get('plotter', {}).get('max_points', 10000))
        self.aruco_marker_data[marker_id].add_pose(t, msg.pose)
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    def unwrap_yaw(self, yaw_array):
        """Unwrap yaw angles to avoid discontinuities."""
        valid_mask = ~np.isnan(yaw_array)
        if np.sum(valid_mask) < 2:
            return yaw_array
        
        result = yaw_array.copy()
        valid_yaw = np.radians(yaw_array[valid_mask])
        unwrapped = np.degrees(np.unwrap(valid_yaw))
        result[valid_mask] = unwrapped
        return result
    
    def save_all_plots(self, prefix='final'):
        """Generate all plots based on available data."""
        rospy.loginfo(f"[PLOTTER] Generating plots with prefix: {prefix}")
        
        # Load yaw display config
        self.yaw_cfg = self.config.get('plotter', {}).get('yaw', {})
        should_unwrap = self.yaw_cfg.get('unwrap', True)
        
        # Get data arrays
        ekf = self.ekf_data.get_arrays()
        dr = self.dr_data.get_arrays()
        aruco = self.aruco_data.get_arrays()
        thermal = self.thermal_data.get_arrays()
        laser = self.laser_data.get_arrays()
        uwb = self.uwb_data.get_arrays()
        aruco_markers = {mid: data.get_arrays() for mid, data in self.aruco_marker_data.items()}
        
        # Unwrap yaw only if configured
        if should_unwrap:
            ekf['yaw'] = self.unwrap_yaw(ekf['yaw'])
            dr['yaw'] = self.unwrap_yaw(dr['yaw'])
            aruco['yaw'] = self.unwrap_yaw(aruco['yaw'])
            thermal['yaw'] = self.unwrap_yaw(thermal['yaw'])
            for marker in aruco_markers.values():
                marker['yaw'] = self.unwrap_yaw(marker['yaw'])
        
        plot_count = 0
        
        # Per-sensor plots (only if sensor has data)
        if self.aruco_data.has_data and len(aruco['times']) > 5:
            self.plot_sensor_comparison('ArUco Combined', aruco, ekf, dr, prefix)
            plot_count += 1

        for marker_id, marker_data in sorted(aruco_markers.items()):
            if self.aruco_marker_data[marker_id].has_data and len(marker_data['times']) > 5:
                self.plot_sensor_comparison(f'ArUco Marker {marker_id}', marker_data, ekf, dr, prefix)
                plot_count += 1

        if self.thermal_data.has_data and len(thermal['times']) > 5:
            self.plot_sensor_comparison('Thermal', thermal, ekf, dr, prefix)
            plot_count += 1
        
        if self.laser_data.has_data and len(laser['times']) > 5:
            self.plot_sensor_comparison('Laser', laser, ekf, dr, prefix)
            plot_count += 1
        
        if self.uwb_data.has_data and len(uwb['times']) > 5:
            self.plot_sensor_comparison('UWB', uwb, ekf, dr, prefix)
            plot_count += 1
        
        # Combined comparison plot
        if self.ekf_data.has_data and len(ekf['times']) > 10:
            self.plot_combined_comparison(ekf, dr, aruco, laser, uwb, thermal, aruco_markers, prefix)
            plot_count += 1
        
        if plot_count == 0:
            rospy.logwarn("[PLOTTER] No data available for plotting")
        else:
            rospy.loginfo(f"[PLOTTER] Generated {plot_count} plot(s) in {self.save_dir}")
    
    def plot_sensor_comparison(self, sensor_name, sensor_data, ekf_data, dr_data, prefix):
        """
        Plot XYZ+Yaw vs Time for a single sensor.
        Shows: Sensor measurement, EKF fused estimate, Dead reckoning (uncorrected)
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'{sensor_name} Sensor vs EKF vs Dead Reckoning\n(All in Landpad Frame)',
                    fontsize=14, fontweight='bold')
        
        # Determine which components this sensor measures
        measures_x   = not np.all(np.isnan(sensor_data['x']))
        measures_y   = not np.all(np.isnan(sensor_data['y']))
        measures_z   = not np.all(np.isnan(sensor_data['z']))
        measures_yaw = not np.all(np.isnan(sensor_data['yaw']))
        
        components = [
            ('X (m)',     'x',   measures_x),
            ('Y (m)',     'y',   measures_y),
            ('Z (m)',     'z',   measures_z),
            ('Yaw (deg)', 'yaw', measures_yaw)
        ]
        
        colors = {'ekf': 'blue', 'dr': 'green', 'sensor': 'red'}
        
        for ax_idx, (label, key, has_sensor_data) in enumerate(components):
            ax = axes[ax_idx]
            is_yaw = (key == 'yaw')
            
            # Always plot EKF
            if len(ekf_data['times']) > 0:
                ax.plot(ekf_data['times'], ekf_data[key],
                    color=colors['ekf'], linewidth=2, label='EKF (fused)')
            
            # Plot Dead Reckoning — skip for yaw if configured
            if len(dr_data['times']) > 0:
                skip_dr = is_yaw and not self.yaw_cfg.get('show_dead_reckoning', True)
                if not skip_dr:
                    ax.plot(dr_data['times'], dr_data[key],
                        color=colors['dr'], linewidth=1.5, linestyle='--',
                        alpha=0.7, label='Dead Reckoning')
            
            # Plot sensor data if available for this component
            if has_sensor_data and len(sensor_data['times']) > 0:
                valid_mask = ~np.isnan(sensor_data[key])
                if np.any(valid_mask):
                    ax.scatter(
                        sensor_data['times'][valid_mask],
                        sensor_data[key][valid_mask],
                        c=colors['sensor'], s=30, marker='o',
                        alpha=0.6, label=f'{sensor_name}',
                        edgecolors='darkred', linewidths=0.5
                    )
            else:
                ax.annotate(f'{sensor_name} does not measure {label.split()[0]}',
                        xy=(0.5, 0.5), xycoords='axes fraction',
                        fontsize=10, style='italic', alpha=0.5,
                        ha='center', va='center')
            
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            # Apply fixed y-axis limits for yaw
            if is_yaw:
                ylim = self.yaw_cfg.get('ylim', None)
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
        
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        plt.tight_layout()
        filename = f'sensor_{sensor_name.lower()}_{prefix}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        rospy.loginfo(f"[PLOTTER] Saved: {filepath}")
    
    def plot_combined_comparison(self, ekf, dr, aruco, laser, uwb, thermal, aruco_markers, prefix):
        """
        Combined comparison plot showing all sensors, EKF, and dead reckoning.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle('Combined Sensor Fusion Overview\n'
                    'EKF (Fused) vs Dead Reckoning vs All Sensor Measurements',
                    fontsize=14, fontweight='bold')
        
        components = [
            ('X (m)',     'x'),
            ('Y (m)',     'y'),
            ('Z (m)',     'z'),
            ('Yaw (deg)', 'yaw')
        ]
        
        styles = {
            'ekf':   {'color': 'blue',   'linewidth': 2.5, 'label': 'EKF (Fused)',                    'zorder': 5},
            'dr':    {'color': 'green',  'linewidth': 2,   'linestyle': '--', 'alpha': 0.7,
                    'label': 'Dead Reckoning (No Corrections)',                                       'zorder': 4},
            'aruco': {'color': 'red',    'marker': 'o', 's': 40, 'alpha': 0.45, 'label': 'ArUco combined', 'zorder': 3},
            'thermal': {'color': 'brown', 'marker': 'D', 's': 38, 'alpha': 0.75, 'label': 'Thermal', 'zorder': 3},
            'laser': {'color': 'orange', 'marker': 's', 's': 35, 'alpha': 0.7, 'label': 'Laser',      'zorder': 3},
            'uwb':   {'color': 'purple', 'marker': '^', 's': 35, 'alpha': 0.7, 'label': 'UWB',        'zorder': 3}
        }
        marker_colors = ['crimson', 'darkcyan', 'darkgoldenrod', 'magenta', 'black', 'tab:pink']
        
        all_handles = []
        all_labels  = []
        
        for ax_idx, (label, key) in enumerate(components):
            ax = axes[ax_idx]
            is_yaw = (key == 'yaw')
            
            # Plot EKF
            if len(ekf['times']) > 0:
                line, = ax.plot(ekf['times'], ekf[key],
                    color=styles['ekf']['color'],
                    linewidth=styles['ekf']['linewidth'],
                    label=styles['ekf']['label'],
                    zorder=styles['ekf']['zorder'])
                if ax_idx == 0:
                    all_handles.append(line)
                    all_labels.append(styles['ekf']['label'])
            
            # Plot Dead Reckoning — skip for yaw if configured
            if len(dr['times']) > 0:
                skip_dr = is_yaw and not self.yaw_cfg.get('show_dead_reckoning', True)
                if not skip_dr:
                    line, = ax.plot(dr['times'], dr[key],
                        color=styles['dr']['color'],
                        linewidth=styles['dr']['linewidth'],
                        linestyle=styles['dr']['linestyle'],
                        alpha=styles['dr']['alpha'],
                        label=styles['dr']['label'],
                        zorder=styles['dr']['zorder'])
                    if ax_idx == 0:
                        all_handles.append(line)
                        all_labels.append(styles['dr']['label'])
            
            # Plot ArUco
            if self.aruco_data.has_data and len(aruco['times']) > 0:
                valid = ~np.isnan(aruco[key])
                if np.any(valid):
                    scatter = ax.scatter(aruco['times'][valid], aruco[key][valid],
                            c=styles['aruco']['color'],
                            s=styles['aruco']['s'],
                            marker=styles['aruco']['marker'],
                            alpha=styles['aruco']['alpha'],
                            label=styles['aruco']['label'],
                            edgecolors='darkred', linewidths=0.5,
                            zorder=styles['aruco']['zorder'])
                    if ax_idx == 0 and styles['aruco']['label'] not in all_labels:
                        all_handles.append(scatter)
                        all_labels.append(styles['aruco']['label'])
            
            # Plot per-marker ArUco measurements with distinct colors
            for idx, (marker_id, marker_data) in enumerate(sorted(aruco_markers.items())):
                if marker_id not in self.aruco_marker_data or not self.aruco_marker_data[marker_id].has_data:
                    continue
                if len(marker_data['times']) == 0:
                    continue
                valid = ~np.isnan(marker_data[key])
                if np.any(valid):
                    color = marker_colors[idx % len(marker_colors)]
                    label_text = f'ArUco ID {marker_id}'
                    scatter = ax.scatter(marker_data['times'][valid], marker_data[key][valid],
                            c=color, s=34, marker='o', alpha=0.75,
                            label=label_text, edgecolors=color, linewidths=0.5, zorder=4)
                    if label_text not in all_labels:
                        all_handles.append(scatter)
                        all_labels.append(label_text)

            # Plot Thermal
            if self.thermal_data.has_data and len(thermal['times']) > 0:
                valid = ~np.isnan(thermal[key])
                if np.any(valid):
                    scatter = ax.scatter(thermal['times'][valid], thermal[key][valid],
                            c=styles['thermal']['color'],
                            s=styles['thermal']['s'],
                            marker=styles['thermal']['marker'],
                            alpha=styles['thermal']['alpha'],
                            label=styles['thermal']['label'],
                            edgecolors='saddlebrown', linewidths=0.5,
                            zorder=styles['thermal']['zorder'])
                    if styles['thermal']['label'] not in all_labels:
                        all_handles.append(scatter)
                        all_labels.append(styles['thermal']['label'])

            # Plot Laser
            if self.laser_data.has_data and len(laser['times']) > 0:
                valid = ~np.isnan(laser[key])
                if np.any(valid):
                    scatter = ax.scatter(laser['times'][valid], laser[key][valid],
                            c=styles['laser']['color'],
                            s=styles['laser']['s'],
                            marker=styles['laser']['marker'],
                            alpha=styles['laser']['alpha'],
                            label=styles['laser']['label'],
                            edgecolors='darkorange', linewidths=0.5,
                            zorder=styles['laser']['zorder'])
                    if styles['laser']['label'] not in all_labels:
                        all_handles.append(scatter)
                        all_labels.append(styles['laser']['label'])
            
            # Plot UWB
            if self.uwb_data.has_data and len(uwb['times']) > 0:
                valid = ~np.isnan(uwb[key])
                if np.any(valid):
                    scatter = ax.scatter(uwb['times'][valid], uwb[key][valid],
                            c=styles['uwb']['color'],
                            s=styles['uwb']['s'],
                            marker=styles['uwb']['marker'],
                            alpha=styles['uwb']['alpha'],
                            label=styles['uwb']['label'],
                            edgecolors='darkviolet', linewidths=0.5,
                            zorder=styles['uwb']['zorder'])
                    if ax_idx == 0 and styles['uwb']['label'] not in all_labels:
                        all_handles.append(scatter)
                        all_labels.append(styles['uwb']['label'])
            
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Apply fixed y-axis limits for yaw
            if is_yaw:
                ylim = self.yaw_cfg.get('ylim', None)
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
        
        if all_handles:
            axes[0].legend(all_handles, all_labels, loc='upper right', fontsize=9, ncol=2)
        
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        process_model = self.sensor_config.get('process_model', 'Unknown')
        info_text = f'Process Model: {process_model}'
        axes[0].annotate(info_text, xy=(0.01, 0.98), xycoords='axes fraction',
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        filename = f'combined_comparison_{prefix}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        rospy.loginfo(f"[PLOTTER] Saved: {filepath}")
    
    # =========================================================================
    # SHUTDOWN HANDLING
    # =========================================================================
    
    def save_and_exit(self):
        """Save plots on shutdown."""
        if self.shutdown_complete:
            return
        
        self.shutdown_complete = True
        
        print("\n" + "="*50)
        print("[PLOTTER] Shutting down, saving plots...")
        print("="*50)
        
        if not self.ekf_data.has_data:
            print("[PLOTTER] No EKF data collected, nothing to save")
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
        """Main run loop."""
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