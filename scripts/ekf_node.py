#!/usr/bin/env python3
"""
Extended Kalman Filter Node for Drone Sensor Fusion - V4 (Modular)

Configuration-driven EKF supporting multiple process models and sensors:
- Process Models: PX4 Velocity, Optical Flow
- Measurement Updates: ArUco, Laser Altimeter, UWB

All parameters loaded from YAML config at runtime.
"""

import rospy
import numpy as np
import yaml
import os
import sys
import time
from collections import deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, PointStamped
from mavros_msgs.msg import Altitude
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import tf.transformations as tft
import tf2_ros
from enum import Enum
import json


class EKFState(Enum):
    WAITING = 1
    TRACKING = 2


class SensorStatus:
    """Track sensor health and last update times."""
    def __init__(self):
        self.aruco_active = False
        self.laser_active = False
        self.uwb_active = False
        self.thermal_active = False
        self.last_aruco_time = 0.0
        self.last_laser_time = 0.0
        self.last_uwb_time = 0.0
        self.last_thermal_time = 0.0
        self.timeout = 2.0  # seconds


class DroneEKF:
    """Extended Kalman Filter for drone state estimation relative to landing pad."""
    
    def __init__(self):
        rospy.init_node('drone_ekf', anonymous=True)
        
        # Load configuration
        self.load_config()
        
        # state machine
        self.mode = EKFState.WAITING
        
        # Landpad frame tracking (since local frame ≠ landpad frame)
        self.landpad_origin_local = None  # Position of landpad in local frame
        self.last_local_position = None   # For computing landpad frame rate
        
        # Initialize EKF state vector: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        self.state = np.zeros(10)
        self.state[9] = 1.0  # qw = 1 (identity quaternion)
        
        # Dead reckoning state (for comparison - no corrections applied)
        self.dead_reckoning_pos = np.zeros(3)
        self.dead_reckoning_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Initialize covariance matrix
        self.init_covariance()
        
        # Process-model timing is defined by the PX4/MAVROS source frequency,
        # not callback arrival jitter. Keep observed source spacing separately
        # so communication delays remain visible in diagnostics.
        process_sensor_cfg = self.config['sensors']['px4_velocity']
        self.process_sample_rate_hz = float(
            process_sensor_cfg.get('sample_rate_hz', 30.0)
        )
        if not np.isfinite(self.process_sample_rate_hz) or self.process_sample_rate_hz <= 0.0:
            raise ValueError("sensors.px4_velocity.sample_rate_hz must be positive")
        self.dt = 1.0 / self.process_sample_rate_hz
        self.last_process_stamp = None
        self.observed_process_dt = None
        
        # Velocity storage
        self.last_body_velocity = np.zeros(3)
        self.last_angular_velocity = np.zeros(3)
        
        # Sensor status tracking
        self.sensor_status = SensorStatus()
        
        self.shutting_down = False

        # Human-facing debug dashboard state. The JSON debug topics remain the
        # authoritative stream for plots/bags; this only keeps the terminal tidy.
        self.setup_debug_dashboard()
        self.dashboard_timer = None
        if self.dashboard_enabled:
            self.dashboard_timer = rospy.Timer(
                rospy.Duration(max(float(self.dashboard_period), 0.1)),
                self.dashboard_timer_cb
            )
        
        rospy.on_shutdown(self.on_shutdown)
        
        # TF broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Setup subscribers and publishers
        self.setup_subscribers()
        self.setup_publishers()
        
        # Publish static landpad frame
        self.publish_static_landpad_frame()
        
        # Status JSON exists for dashboards/plots, not the estimator itself.
        self.sensor_status_timer = None
        if not self.flight_optimized:
            self.sensor_status_timer = rospy.Timer(
                rospy.Duration(1.0), self.publish_sensor_status
            )
        
        # NEW: Alternative initialization if ArUco not enabled
        wait_for_aruco = self.config['initialization'].get('wait_for_aruco', True)
        aruco_enabled = self.config['sensors']['aruco']['enabled']
        
        if not wait_for_aruco or not aruco_enabled:
            # Initialize at default position or wait for UWB
            rospy.Timer(rospy.Duration(2.0), self.try_alternative_initialization, oneshot=True)
        
        rospy.loginfo("[EKF] Node initialized (V4 - Modular)")
        rospy.loginfo(f"[EKF] Process model: {self.config['process_model']['type']}")
        rospy.loginfo(
            f"[EKF] Fixed process dt: {self.dt:.6f} s "
            f"({self.process_sample_rate_hz:.3f} Hz PX4 velocity source)"
        )
        rospy.loginfo(f"[EKF] ArUco: {self.config['sensors']['aruco']['enabled']}")
        rospy.loginfo(f"[EKF] Laser: {self.config['sensors']['laser']['enabled']}")
        rospy.loginfo(f"[EKF] UWB: {self.config['sensors']['uwb']['enabled']}")
        
        if wait_for_aruco and aruco_enabled:
            rospy.loginfo("[EKF] Waiting for first ArUco detection...")
        else:
            rospy.loginfo("[EKF] ArUco initialization disabled - using alternative init")
    
    
    def on_shutdown(self):
        self.shutting_down = True
    
    # Alternative initialization method
    def try_alternative_initialization(self, event=None):
        """
        Alternative initialization for testing without ArUco.

        Uses position/orientation from config file.

        WARNING: This assumes the drone starts at the configured pose
        relative to the landing pad. Only use for:
        - Static testing on the ground
        - Debugging the EKF prediction loop
        - Testing with other absolute position sensors (e.g., UWB)

        For real flights, always use ArUco initialization (wait_for_aruco: true)
        """
        if self.mode == EKFState.TRACKING:
            return

        default_position = np.array(
            self.config['initialization'].get('default_position', [0.0, 0.0, 0.1])
        )
        default_quaternion = np.array(
            self.config['initialization'].get('default_orientation', [0.0, 0.0, 0.0, 1.0])
        )
        default_quaternion = self.normalize_quaternion(default_quaternion)

        self.state[0:3] = default_position
        self.state[3:6] = np.zeros(3)
        self.state[6:10] = default_quaternion
        self.dead_reckoning_pos = default_position.copy()
        self.dead_reckoning_quat = default_quaternion.copy()

        self.init_covariance()
        cov_scale = self.config['initialization'].get('alternative_covariance_scale', 2.0)
        self.P *= cov_scale
        self.mode = EKFState.TRACKING

        # Force dashboard refresh after the state and mode really changed.
        self.maybe_render_debug_dashboard(force=True)

        rospy.loginfo("=" * 60)
        rospy.loginfo("[INIT] Alternative initialization complete")
        rospy.loginfo(f"[INIT] Position: [{default_position[0]:.3f}, "
                    f"{default_position[1]:.3f}, {default_position[2]:.3f}]")
        rospy.loginfo(f"[INIT] Orientation: [{default_quaternion[0]:.3f}, "
                    f"{default_quaternion[1]:.3f}, {default_quaternion[2]:.3f}, "
                    f"{default_quaternion[3]:.3f}]")
        roll, pitch, yaw = tft.euler_from_quaternion(default_quaternion)
        rospy.loginfo(f"[INIT] Euler (deg): roll={np.degrees(roll):.1f}, "
                    f"pitch={np.degrees(pitch):.1f}, yaw={np.degrees(yaw):.1f}")
        rospy.loginfo(f"[INIT] Covariance scale: {cov_scale}x")
        rospy.loginfo("=" * 60)
    # =========================================================================
    # CONFIGURATION LOADING
    # =========================================================================
    
    def load_config(self):
        """Load configuration from YAML file."""
        config_path = rospy.get_param('~config_file', '')
        
        if not config_path:
            # Try default locations
            pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_path, 'config', 'ekf_params.yaml')
        
        if not os.path.exists(config_path):
            rospy.logerr(f"[EKF] Config file not found: {config_path}")
            rospy.signal_shutdown("Config file not found")
            return
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Debug configuration
        debug_defaults = {
            'log_innovations': False,
            'log_covariance': False,
            'log_prediction': False
        }
        self.debug_config = debug_defaults.copy()
        self.debug_config.update(self.config.get('debug', {}))
        self.flight_optimized = bool(self.config.get('flight_optimized', False))
        self.debug_topics_enabled = not self.flight_optimized
        if self.flight_optimized:
            self.debug_config.update({
                'enabled': False,
                'dashboard': False,
                'log_innovations': False,
                'log_covariance': False,
                'log_prediction': False,
            })
        
        rospy.loginfo(f"[EKF] Loaded config from: {config_path}")
        
        # Parse transforms
        self.parse_transforms()
        
        # Build noise matrices
        self.build_noise_matrices()
    
    def parse_transforms(self):
        """Parse 4x4 transformation matrices from config."""
        self.transforms = {}
        
        transform_keys = [
            'T_cam_to_drone', 'T_laser_to_drone',
            'T_uwb_to_drone'
        ]
        if 'T_thermal_to_drone' in self.config.get('transforms', {}):
            transform_keys.append('T_thermal_to_drone')
        
        for key in transform_keys:
            matrix_flat = self.config['transforms'][key]['matrix']
            self.transforms[key] = np.array(matrix_flat).reshape(4, 4)
            self.validate_transform(self.transforms[key], key)
        
        # UWB map to landpad transform
        matrix_flat = self.config['T_uwb_map_to_landpad']['matrix']
        self.transforms['T_uwb_map_to_landpad'] = np.array(matrix_flat).reshape(4, 4)
        self.validate_transform(self.transforms['T_uwb_map_to_landpad'], 'T_uwb_map_to_landpad')
        
    def validate_transform(self, T, name):
        """Validate 4x4 rigid-body transformation matrix."""
        R = T[:3, :3]
        det = np.linalg.det(R)
        orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-3)
        
        if not orthogonal:
            rospy.logerr(f"[TRANSFORM] {name}: Rotation not orthogonal! R*R^T =\n{R @ R.T}")
            rospy.signal_shutdown("Invalid transform matrix")
        
        if abs(det - 1.0) > 0.01:
            rospy.logerr(f"[TRANSFORM] {name}: Invalid determinant = {det} (should be +1.0)")
            rospy.signal_shutdown("Invalid transform matrix")
        
        rospy.loginfo(f"[TRANSFORM] {name} validated ✓ (det={det:.3f})")
    
    def build_noise_matrices(self):
        """Build Q and R matrices from config."""
        # Process noise Q
        q_pos = self.config['process_model']['Q']['position']
        q_vel = self.config['process_model']['Q']['velocity']
        q_ori = self.config['process_model']['Q']['orientation']
        self.Q = np.diag(q_pos + q_vel + q_ori)
        
        # Default measurement noise matrices
        aruco_noise_cfg = self.config['measurement_noise']['aruco']
        self.R_aruco_pos = np.diag(aruco_noise_cfg['position'])
        self.R_laser = np.diag(self.config['measurement_noise']['laser']['z'])
        self.R_uwb = np.diag(self.config['measurement_noise']['uwb']['xy'])

        # Yaw-only vision update variance in rad^2.
        default_aruco_yaw_var = aruco_noise_cfg['yaw']
        self.R_aruco_yaw = np.array([[default_aruco_yaw_var]])
        thermal_noise = self.config.get('measurement_noise', {}).get('thermal', {})
        self.R_thermal_pos = np.diag(thermal_noise.get('position',
            self.config['measurement_noise']['aruco']['position']))
        self.R_thermal_yaw = np.array([[thermal_noise.get('yaw', self.R_aruco_yaw[0, 0])]])
        
        # Per-marker R matrices (fall back to default if not specified)
        self.R_aruco_per_marker_pos = {}
        self.R_aruco_per_marker_yaw = {}
        markers_cfg = self.config['sensors']['aruco'].get('markers', {})
        for marker_id_str, marker_cfg in markers_cfg.items():
            mid = int(marker_id_str)
            self.R_aruco_per_marker_pos[mid] = np.diag(
                marker_cfg.get('position_noise',
                               self.config['measurement_noise']['aruco']['position'])
            )
            self.R_aruco_per_marker_yaw[mid] = np.array([[
                marker_cfg.get('orientation_noise_yaw', default_aruco_yaw_var)
            ]])
            rospy.loginfo(f"[EKF] Marker {mid} R_pos diag: {np.diag(self.R_aruco_per_marker_pos[mid])}")
        
        # Mahalanobis gates. New-style config uses explicit enable flags;
        # old-style negative thresholds still disable gates for compatibility.
        self.mahal_gates = self.config.get('mahalanobis_gates', {
            'aruco_position': -1.0,
            'aruco_xy': -1.0,
            'aruco_z': -1.0,
            'aruco_yaw': -1.0,
            'thermal_position': -1.0,
            'thermal_yaw': -1.0,
            'uwb': -1.0
        })
        self.mahal_cfg = self.config.get('mahalanobis', {})
    
    def init_covariance(self):
        """Initialize covariance matrix from config."""
        p_pos = self.config['initialization']['P_init']['position']
        p_vel = self.config['initialization']['P_init']['velocity']
        p_ori = self.config['initialization']['P_init']['orientation']
        self.P = np.diag(p_pos + p_vel + p_ori)
        self.stabilize_covariance()

    # =========================================================================
    # ROS SETUP
    # =========================================================================
    
    def setup_subscribers(self):
        """Setup ROS subscribers based on configuration."""
        sensors = self.config['sensors']
        process_type = self.config['process_model']['type']
        
        # Process model subscribers
        if process_type == 'PX4_Velocity' and sensors['px4_velocity']['enabled']:
            rospy.Subscriber(
                sensors['px4_velocity']['topic'],
                TwistStamped, self.px4_velocity_callback, queue_size=1
            )
            rospy.loginfo(f"[EKF] Subscribed to PX4 velocity: {sensors['px4_velocity']['topic']}")
        else:
            rospy.logwarn(
                f"[EKF] Unsupported or disabled process model '{process_type}'. "
                "Configure process_model.type='PX4_Velocity' and enable sensors.px4_velocity."
            )
        
        # Per-marker ArUco subscribers
        if sensors['aruco']['enabled']:
            markers_cfg = sensors['aruco'].get('markers', {})
            for marker_id_str, marker_cfg in markers_cfg.items():
                topic = marker_cfg.get('topic', f"/aruco/pose/marker_{marker_id_str}")
                marker_id = int(marker_id_str)
                rospy.Subscriber(
                    topic, PoseStamped,
                    lambda msg, mid=marker_id: self.aruco_marker_callback(msg, mid),
                    queue_size=1
                )
                rospy.loginfo(f"[EKF] Subscribed to ArUco marker {marker_id}: {topic}")

        thermal_cfg = sensors.get('thermal', {})
        if thermal_cfg.get('enabled', False):
            rospy.Subscriber(
                thermal_cfg.get('topic', '/thermal/pose'),
                PoseStamped, self.thermal_callback, queue_size=1
            )
            rospy.loginfo(f"[EKF] Subscribed to Thermal: {thermal_cfg.get('topic', '/thermal/pose')}")
        
        if sensors['laser']['enabled']:
            rospy.Subscriber(
                sensors['laser']['topic'],
                Altitude, self.laser_callback, queue_size=1
            )
            rospy.loginfo(f"[EKF] Subscribed to Laser: {sensors['laser']['topic']}")
        
        if sensors['uwb']['enabled']:
            rospy.Subscriber(
                sensors['uwb']['topic'],
                PoseStamped, self.uwb_callback, queue_size=1
            )
            rospy.loginfo(f"[EKF] Subscribed to UWB: {sensors['uwb']['topic']}")
    
    def setup_publishers(self):
        """Setup ROS publishers."""
        topics = self.config['output_topics']
        
        self.ekf_pose_pub = rospy.Publisher(topics['ekf_pose'], PoseStamped, queue_size=1)
        self.ekf_odom_pub = rospy.Publisher(topics['ekf_odom'], Odometry, queue_size=1)
        self.dead_reckoning_pub = None
        if not self.flight_optimized:
            self.dead_reckoning_pub = rospy.Publisher(
                topics['dead_reckoning'], PoseStamped, queue_size=1
            )
        
        # Per-sensor measurement publishers (transformed to landpad frame)
        self.aruco_marker_meas_pubs = {}
        self.thermal_meas_pub = None
        self.laser_meas_pub = None
        self.uwb_meas_pub = None
        if not self.flight_optimized:
            aruco_prefix = topics.get(
                'aruco_marker_measurement_prefix',
                '/ekf/measurements/aruco/marker_'
            )
            for marker_id_str in self.config.get(
                    'sensors', {}).get('aruco', {}).get('markers', {}):
                mid = int(marker_id_str)
                self.aruco_marker_meas_pubs[mid] = rospy.Publisher(
                    f"{aruco_prefix}{mid}", PoseStamped, queue_size=1
                )
            self.thermal_meas_pub = rospy.Publisher(
                topics.get('thermal_measurement', '/ekf/measurements/thermal'),
                PoseStamped, queue_size=1
            )
            self.laser_meas_pub = rospy.Publisher(
                topics['laser_measurement'], PointStamped, queue_size=1
            )
            self.uwb_meas_pub = rospy.Publisher(
                topics['uwb_measurement'], PointStamped, queue_size=1
            )
        
        self.sensor_status_pub = None
        self.diagnostics_pub = None
        self.innovation_pub = None
        self.covariance_pub = None
        self.kalman_gain_pub = None
        self.timing_pub = None
        if self.flight_optimized:
            return

        # Optional status and diagnostics publishers.
        self.sensor_status_pub = rospy.Publisher(
            topics['sensor_status'], String, queue_size=1
        )
        
        self.diagnostics_pub = rospy.Publisher('/ekf/diagnostics', DiagnosticArray, queue_size=1)
        self.innovation_pub = rospy.Publisher(
            topics.get('innovation_debug', '/ekf/debug/innovation'),
            String, queue_size=200
        )
        self.covariance_pub = rospy.Publisher(
            topics.get('covariance_debug', '/ekf/debug/covariance'),
            String, queue_size=100
        )
        self.kalman_gain_pub = rospy.Publisher(
            topics.get('kalman_gain_debug', '/ekf/debug/kalman_gain'),
            String, queue_size=200
        )
        self.timing_pub = rospy.Publisher(
            topics.get('timing_debug', '/ekf/debug/timing'),
            String, queue_size=200
        )
    
    def publish_static_landpad_frame(self):
        """Publish landpad frame to TF for RViz visualization."""
        static_transform = TransformStamped()
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "world"
        static_transform.child_frame_id = "landpad"
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(static_transform)
    
    def publish_sensor_status(self, event=None):
        """Publish sensor status for plotter configuration."""
        if self.sensor_status_pub is None:
            return
        current_time = rospy.Time.now().to_sec()
        timeout = self.sensor_status.timeout
        
        status = {
            'aruco': {
                'enabled': self.config['sensors']['aruco']['enabled'],
                'active': (current_time - self.sensor_status.last_aruco_time) < timeout
            },
            'laser': {
                'enabled': self.config['sensors']['laser']['enabled'],
                'active': (current_time - self.sensor_status.last_laser_time) < timeout
            },
            'uwb': {
                'enabled': self.config['sensors']['uwb']['enabled'],
                'active': (current_time - self.sensor_status.last_uwb_time) < timeout
            },
            'thermal': {
                'enabled': self.config['sensors'].get('thermal', {}).get('enabled', False),
                'active': (current_time - self.sensor_status.last_thermal_time) < timeout
            },
            'process_model': self.config['process_model']['type']
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.sensor_status_pub.publish(msg)
        
    def publish_diagnostics(self, header):
        """Publish EKF diagnostics for monitoring."""
        if self.diagnostics_pub is None:
            return
        diag_array = DiagnosticArray()
        diag_array.header.stamp = header.stamp
        
        # Overall EKF status
        status = DiagnosticStatus()
        status.name = "EKF State Estimator"
        status.hardware_id = "drone_ekf"
        
        if self.mode == EKFState.TRACKING:
            status.level = DiagnosticStatus.OK
            status.message = "Tracking"
        else:
            status.level = DiagnosticStatus.WARN
            status.message = "Waiting for initialization"
        
        # Add key metrics
        status.values.append(KeyValue("pos_x", f"{self.state[0]:.3f}"))
        status.values.append(KeyValue("pos_y", f"{self.state[1]:.3f}"))
        status.values.append(KeyValue("pos_z", f"{self.state[2]:.3f}"))
        status.values.append(KeyValue("vel_norm", f"{np.linalg.norm(self.state[3:6]):.3f}"))
        status.values.append(KeyValue("cov_pos", f"{np.trace(self.P[:3,:3]):.4f}"))
        status.values.append(KeyValue("cov_ori", f"{np.trace(self.P[6:10,6:10]):.4f}"))
        status.values.append(KeyValue("fixed_process_dt", f"{self.dt:.6f}"))
        status.values.append(KeyValue(
            "observed_process_dt",
            (
                f"{self.observed_process_dt:.6f}"
                if self.observed_process_dt is not None else "unavailable"
            )
        ))
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)

    def serialize_vector(self, values):
        """Convert numpy vectors to JSON-safe Python floats."""
        return [float(v) for v in np.asarray(values).flatten()]

    def serialize_matrix(self, values):
        """Convert numpy matrices to nested Python float lists."""
        arr = np.asarray(values)
        return [[float(v) for v in row] for row in arr]

    def measurement_age_sec(self, header):
        """Age of a measurement relative to current ROS time."""
        if header.stamp.is_zero():
            return 0.0
        return float((rospy.Time.now() - header.stamp).to_sec())

    def publish_json_debug(self, publisher, payload):
        """Publish a JSON payload on a String topic."""
        if publisher is None:
            return
        msg = String()
        msg.data = json.dumps(payload)
        publisher.publish(msg)

    def publish_covariance_debug(self, header, source):
        """Publish EKF covariance diagonal for offline plotting."""
        if self.covariance_pub is None:
            return
        payload = {
            'stamp': float(header.stamp.to_sec()),
            'source': source,
            'p_diag': self.serialize_vector(np.diag(self.P))
        }
        self.publish_json_debug(self.covariance_pub, payload)

    def publish_timing_debug(self, header, stage, sensor=None, component=None,
                             marker_id=None, accepted=None, dt=None,
                             observed_dt=None):
        """Publish timestamp/age diagnostics for EKF data-flow analysis."""
        if self.timing_pub is None:
            return
        stamp = float(header.stamp.to_sec()) if not header.stamp.is_zero() else rospy.Time.now().to_sec()
        now = rospy.Time.now().to_sec()
        payload = {
            'stamp': stamp,
            'process_stamp': now,
            'age_sec': now - stamp,
            'stage': stage,
            'sensor': sensor,
            'component': component,
            'marker_id': int(marker_id) if marker_id is not None else None,
        }
        if accepted is not None:
            payload['accepted'] = bool(accepted)
        if dt is not None:
            payload['dt'] = float(dt)
        if observed_dt is not None:
            payload['observed_dt'] = float(observed_dt)
        self.publish_json_debug(self.timing_pub, payload)

    def publish_aruco_callback_timing(self, header, marker_id,
                                      callback_start_stamp,
                                      callback_start_wall, status, timings):
        """Publish a consolidated detector-pose-to-EKF callback profile."""
<<<<<<< HEAD
        if self.timing_pub is None:
            return
=======
>>>>>>> e84c788dbc0ef7ce1361fea8f839097d2e8d50db
        now = rospy.Time.now().to_sec()
        stamp = (
            float(header.stamp.to_sec())
            if not header.stamp.is_zero() else callback_start_stamp
        )
        payload = {
            'stamp': stamp,
            'process_stamp': now,
            'callback_start_stamp': float(callback_start_stamp),
            'callback_complete_stamp': now,
            'age_sec': now - stamp,
            'source_to_callback_ms': (callback_start_stamp - stamp) * 1000.0,
            'callback_total_ms': (time.perf_counter() - callback_start_wall) * 1000.0,
            'stage': 'aruco_callback_profile',
            'sensor': 'aruco',
            'marker_id': int(marker_id),
            'status': status,
        }
        payload.update({key: float(value) for key, value in timings.items()})
        self.publish_json_debug(self.timing_pub, payload)

    def publish_update_debug(self, header, sensor_name, component, innovation, S, R,
                             accepted, mahal_sq, gate, K=None, marker_id=None,
                             post_fit_residual=None, measurement_gain=None,
                             rejection_reason=None):
        """Publish innovation, residual, and gain data for tuning plots."""
        if not self.debug_topics_enabled and not self.dashboard_enabled:
            return
        innovation_payload = {
            'stamp': float(header.stamp.to_sec()),
            'sensor': sensor_name,
            'component': component,
            'marker_id': int(marker_id) if marker_id is not None else None,
            'accepted': bool(accepted),
            'innovation': self.serialize_vector(innovation),
            'post_fit_residual': (
                self.serialize_vector(post_fit_residual)
                if post_fit_residual is not None else None
            ),
            's_diag': self.serialize_vector(np.diag(S)),
            'r_diag': self.serialize_vector(np.diag(R)),
            'mahalanobis_sq': float(mahal_sq),
            'gate_threshold': float(gate),
            'age_sec': self.measurement_age_sec(header),
            'rejection_reason': rejection_reason
        }
        self.publish_json_debug(self.innovation_pub, innovation_payload)
        self.publish_timing_debug(
            header, 'measurement_update',
            sensor=sensor_name,
            component=component,
            marker_id=marker_id,
            accepted=accepted
        )

        if K is not None:
            gain_payload = {
                'stamp': float(header.stamp.to_sec()),
                'sensor': sensor_name,
                'component': component,
                'marker_id': int(marker_id) if marker_id is not None else None,
                'accepted': bool(accepted),
                'shape': list(np.asarray(K).shape),
                'kalman_gain': self.serialize_matrix(K)
            }
            if measurement_gain is not None:
                gain_payload['measurement_gain'] = float(measurement_gain)
            self.publish_json_debug(self.kalman_gain_pub, gain_payload)

        self.record_update_dashboard(
            header, sensor_name, component, innovation, accepted, mahal_sq, gate,
            marker_id=marker_id,
            post_fit_residual=post_fit_residual,
            K=K,
            measurement_gain=measurement_gain,
            rejection_reason=rejection_reason
        )

    # =========================================================================
    # HUMAN-FACING DEBUG DASHBOARD
    # =========================================================================

    def setup_debug_dashboard(self):
        """Configure compact terminal output for EKF debug sessions."""
        legacy_verbose = any(
            bool(self.debug_config.get(key, False))
            for key in ('log_innovations', 'log_covariance', 'log_prediction')
        )
        debug_enabled = bool(self.debug_config.get('enabled', False))
        self.dashboard_enabled = bool(
            self.debug_config.get('dashboard', debug_enabled or legacy_verbose)
        )
        self.dashboard_clear_screen = bool(
            self.debug_config.get('dashboard_clear_screen', True)
        )
        self.dashboard_period = float(self.debug_config.get('dashboard_period', 0.5))
        self.dashboard_recent_events = deque(
            maxlen=int(self.debug_config.get('dashboard_recent_events', 8))
        )
        self.dashboard_updates = {}
        self.dashboard_counts = {}
        self.dashboard_last_prediction = {}
        self.dashboard_last_render_wall = 0.0
        
    def dashboard_timer_cb(self, _event):
        """Refresh the terminal dashboard periodically, even if no EKF event occurs."""
        self.maybe_render_debug_dashboard()

    def fmt_num(self, value, digits=3, unit=''):
        if value is None:
            return 'n/a'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 'n/a'
        if not np.isfinite(value):
            return 'n/a'
        return f"{value:.{digits}f}{unit}"

    def fmt_vec(self, values, digits=3, degrees=False, max_items=None):
        if values is None:
            return '[n/a]'
        arr = np.asarray(values, dtype=float).flatten()
        if degrees:
            arr = np.degrees(arr)
        if max_items is not None:
            arr = arr[:max_items]
        return '[' + ' '.join(self.fmt_num(v, digits) for v in arr) + ']'

    def gate_summary(self):
        specs = [
            ('aruco', 'xy', 'aruco_xy', 9.210),
            ('aruco', 'z', 'aruco_z', 6.635),
            ('aruco', 'yaw', 'aruco_yaw', 6.635),
            ('laser', 'z', 'laser', 6.635),
            ('uwb', 'xy', 'uwb', 9.210),
            ('thermal', 'position', 'thermal_position', 11.345),
            ('thermal', 'yaw', 'thermal_yaw', 6.635),
        ]
        parts = []
        for sensor, component, legacy_key, default in specs:
            enabled, threshold = self.gate_enabled_and_threshold(
                sensor, component, legacy_key, default
            )
            if enabled:
                parts.append(f"{sensor}:{component}<={threshold:.2f}")
        if not parts:
            parts.append('mahalanobis off')
        return ' | '.join(parts)

    def sensor_summary(self):
        now = rospy.Time.now().to_sec()
        timeout = self.sensor_status.timeout
        specs = [
            ('aruco', self.sensor_status.last_aruco_time),
            ('laser', self.sensor_status.last_laser_time),
            ('uwb', self.sensor_status.last_uwb_time),
            ('thermal', self.sensor_status.last_thermal_time),
        ]
        parts = []
        for name, last_time in specs:
            enabled = bool(self.config.get('sensors', {}).get(name, {}).get('enabled', False))
            if last_time > 0.0 and now > 0.0:
                age = max(0.0, now - last_time)
                active = age < timeout
                parts.append(
                    f"{name}:{'on' if enabled else 'off'}/"
                    f"{'active' if active else 'stale'} {age:.2f}s"
                )
            else:
                parts.append(f"{name}:{'on' if enabled else 'off'}/waiting")
        return ' | '.join(parts)

    def record_prediction_dashboard(self, header, v_body, v_landpad, omega_body):
        if not self.dashboard_enabled:
            return
        self.dashboard_last_prediction = {
            'stamp': float(header.stamp.to_sec()) if not header.stamp.is_zero() else rospy.Time.now().to_sec(),
            'dt': float(self.dt),
            'v_body': self.serialize_vector(v_body),
            'v_landpad': self.serialize_vector(v_landpad),
            'omega_body': self.serialize_vector(omega_body),
        }
        self.maybe_render_debug_dashboard()

    def record_update_dashboard(self, header, sensor_name, component, innovation,
                                accepted, mahal_sq, gate, marker_id=None,
                                post_fit_residual=None, K=None,
                                measurement_gain=None, rejection_reason=None):
        if not self.dashboard_enabled:
            return
        stamp = float(header.stamp.to_sec()) if not header.stamp.is_zero() else rospy.Time.now().to_sec()
        update = {
            'wall_time': rospy.Time.now().to_sec(),
            'stamp': stamp,
            'sensor': sensor_name,
            'component': component,
            'marker_id': marker_id,
            'accepted': bool(accepted),
            'innovation': self.serialize_vector(innovation),
            'post_fit_residual': (
                self.serialize_vector(post_fit_residual)
                if post_fit_residual is not None else None
            ),
            'mahalanobis_sq': float(mahal_sq),
            'gate_threshold': float(gate),
            'rejection_reason': rejection_reason,
            'gain_norm': float(np.linalg.norm(K)) if K is not None else None,
            'measurement_gain': (
                float(measurement_gain)
                if measurement_gain is not None else None
            )
        }
        key = (sensor_name, marker_id, component)
        self.dashboard_updates[key] = update
        self.dashboard_recent_events.appendleft(update)

        count_key = (sensor_name, component, 'ok' if accepted else 'rejected')
        self.dashboard_counts[count_key] = self.dashboard_counts.get(count_key, 0) + 1
        self.maybe_render_debug_dashboard(force=not accepted)

    def format_update_line(self, event):
        marker = event.get('marker_id')
        sensor = event.get('sensor', 'unknown')
        label = f"{sensor}:{marker}" if marker is not None else sensor
        component = event.get('component', 'unknown')
        accepted = bool(event.get('accepted', False))
        status = 'OK ' if accepted else 'REJ'
        gate = event.get('gate_threshold')
        nis = self.fmt_num(event.get('mahalanobis_sq'), 2)
        if gate is not None and float(gate) > 0.0:
            nis = f"{nis}/{float(gate):.2f}"
        else:
            nis = f"{nis}/off"
        degrees = component == 'yaw'
        unit = 'deg' if degrees else 'm'
        innovation = self.fmt_vec(event.get('innovation'), 3, degrees=degrees)
        residual = self.fmt_vec(event.get('post_fit_residual'), 3, degrees=degrees)
        gain = event.get('measurement_gain')
        if gain is None:
            gain = event.get('gain_norm')
        gain_text = self.fmt_num(gain, 3)
        reason = event.get('rejection_reason') or ''
        reason = f" | {reason}" if reason else ''
        return (
            f"{status} {label:<12} {component:<8} "
            f"NIS {nis:<13} innov {innovation:<24} "
            f"res {residual:<24} K {gain_text:<7} {unit}{reason}"
        )

    def counts_summary(self):
        parts = []
        keys = sorted(self.dashboard_counts)
        for sensor, component, status in keys:
            parts.append(f"{sensor}:{component}:{status}={self.dashboard_counts[(sensor, component, status)]}")
        return ' | '.join(parts) if parts else 'no measurement updates yet'

    def render_debug_dashboard(self):
        yaw_deg = np.degrees(self.yaw_from_quaternion(self.state[6:10]))
        p_diag = np.diag(self.P)
        p_std = np.sqrt(np.maximum(p_diag, 0.0))
        pred = self.dashboard_last_prediction
        now = rospy.Time.now().to_sec()
        wall_stamp = time.strftime('%H:%M:%S')

        lines = [
            "DRONE EKF DEBUG",
            f"time {wall_stamp} | ros {now:.3f}s | mode {self.mode.name} | process {self.config['process_model']['type']}",
            f"state  pos {self.fmt_vec(self.state[0:3])} m | vel {self.fmt_vec(self.state[3:6])} m/s | yaw {yaw_deg:.2f} deg | dt {self.dt:.3f}s",
            f"cov    std_pos {self.fmt_vec(p_std[0:3])} | std_vel {self.fmt_vec(p_std[3:6])} | std_quat {self.fmt_vec(p_std[6:10], max_items=4)}",
            f"sensor {self.sensor_summary()}",
            f"gates  {self.gate_summary()}",
        ]

        if pred:
            lines.append(
                "pred   "
                f"v_body {self.fmt_vec(pred.get('v_body'))} -> "
                f"v_landpad {self.fmt_vec(pred.get('v_landpad'))} | "
                f"omega {self.fmt_vec(pred.get('omega_body'))} rad/s"
            )
        else:
            if self.mode == EKFState.TRACKING:
                lines.append("pred   waiting for PX4 velocity")
            else:
                lines.append("pred   inactive until EKF enters TRACKING")

        lines.append("updates last events")
        if self.dashboard_recent_events:
            for event in list(self.dashboard_recent_events)[:self.dashboard_recent_events.maxlen]:
                lines.append("  " + self.format_update_line(event))
        else:
            lines.append("  waiting for measurement updates")
        lines.append("counts " + self.counts_summary())
        return '\n'.join(lines)

    def maybe_render_debug_dashboard(self, force=False):
        if getattr(self, 'shutting_down', False):
            return

        if not self.dashboard_enabled:
            return

        now_wall = time.time()
        if not force and (now_wall - self.dashboard_last_render_wall) < self.dashboard_period:
            return

        self.dashboard_last_render_wall = now_wall

        text = self.render_debug_dashboard()
        try:
            if self.dashboard_clear_screen:
                sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[EKF] Dashboard render failed: %s", exc)

    # =========================================================================
    # MATH HELPER FUNCTIONS
    # =========================================================================
    
    def pose_msg_to_matrix(self, position, orientation):
        """Convert ROS pose message to 4x4 transformation matrix."""
        q = self.normalize_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        T = tft.quaternion_matrix(q)
        T[0, 3] = position.x
        T[1, 3] = position.y
        T[2, 3] = position.z
        return T
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def quaternion_conjugate(self, q):
        """Return conjugate of quaternion [x, y, z, w]."""
        return np.array([-q[0], -q[1], -q[2], q[3]])
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        return tft.quaternion_matrix(q)[:3, :3]
    
    def quaternion_to_rotation_vector(self, q):
        """Convert quaternion to rotation vector (axis-angle)."""
        x, y, z, w = q
        if w < 0:
            x, y, z, w = -x, -y, -z, -w
        if w > 0.9999:
            return 2.0 * np.array([x, y, z])
        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        sin_half = np.sqrt(1 - w*w)
        if sin_half < 1e-6:
            return np.zeros(3)
        return (angle / sin_half) * np.array([x, y, z])
    
    def rotation_vector_to_quaternion(self, rot_vec):
        """Convert rotation vector (axis-angle) to quaternion."""
        angle = np.linalg.norm(rot_vec)
        if angle < 1e-6:
            return np.array([0.5*rot_vec[0], 0.5*rot_vec[1], 0.5*rot_vec[2], 1.0])
        axis = rot_vec / angle
        sin_half = np.sin(angle / 2.0)
        cos_half = np.cos(angle / 2.0)
        return np.array([axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half, cos_half])
    
    def normalize_quaternion(self, q):
        """Normalize quaternion to unit length."""
        q = np.asarray(q, dtype=float)
        norm = np.linalg.norm(q)
        if np.isfinite(norm) and norm > 1e-6:
            return q / norm
        return np.array([0.0, 0.0, 0.0, 1.0])

    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def yaw_from_quaternion(self, q):
        """Extract yaw from quaternion [x, y, z, w]."""
        return tft.euler_from_quaternion(self.normalize_quaternion(q))[2]

    def finite_pose(self, position, quaternion):
        """Reject NaN/Inf measurements before they enter the EKF."""
        return np.all(np.isfinite(position)) and np.all(np.isfinite(quaternion))

    def yaw_quaternion_jacobian(self, q):
        """Jacobian d(yaw)/d(qx,qy,qz,qw) for a normalized quaternion."""
        qx, qy, qz, qw = self.normalize_quaternion(q)
        numerator = 2.0 * (qw * qz + qx * qy)
        denominator = 1.0 - 2.0 * (qy * qy + qz * qz)
        norm_sq = numerator * numerator + denominator * denominator

        if norm_sq < 1e-12:
            return np.array([0.0, 0.0, 2.0, 0.0])

        d_num = np.array([2.0 * qy, 2.0 * qx, 2.0 * qw, 2.0 * qz])
        d_den = np.array([0.0, -4.0 * qy, -4.0 * qz, 0.0])
        return (denominator * d_num - numerator * d_den) / norm_sq

    def right_quaternion_product_matrix(self, q_delta):
        """Matrix such that q_current ⊗ q_delta = M(q_delta) q_current."""
        x2, y2, z2, w2 = q_delta
        return np.array([
            [w2,  z2, -y2, x2],
            [-z2, w2,  x2, y2],
            [y2, -x2,  w2, z2],
            [-x2, -y2, -z2, w2]
        ])

    def rotated_velocity_quaternion_jacobian(self, q, v_body):
        """Numerical Jacobian d(R(q) v_body)/d(qx,qy,qz,qw)."""
        eps = 1e-6
        q = self.normalize_quaternion(q)
        jac = np.zeros((3, 4))

        for idx in range(4):
            dq = np.zeros(4)
            dq[idx] = eps
            q_plus = self.normalize_quaternion(q + dq)
            q_minus = self.normalize_quaternion(q - dq)
            v_plus = self.quaternion_to_rotation_matrix(q_plus) @ v_body
            v_minus = self.quaternion_to_rotation_matrix(q_minus) @ v_body
            jac[:, idx] = (v_plus - v_minus) / (2.0 * eps)

        return jac

    def stabilize_covariance(self):
        """Keep P symmetric after linearized prediction/update steps."""
        self.P = 0.5 * (self.P + self.P.T)

    def normalize_state_quaternion(self):
        """Normalize the state quaternion and project P through that operation."""
        q = self.state[6:10].copy()
        norm = np.linalg.norm(q)

        if not np.isfinite(norm) or norm <= 1e-6:
            self.state[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
            return

        q_normalized = q / norm
        J_norm = (np.eye(4) - np.outer(q_normalized, q_normalized)) / norm
        A = np.eye(10)
        A[6:10, 6:10] = J_norm
        self.state[6:10] = q_normalized
        self.P = A @ self.P @ A.T
        self.stabilize_covariance()

    def is_stale(self, header, sensor_name):
        """Return True when a timestamped measurement is too old. Disabled by default."""
        sensor_cfg = self.config.get('sensors', {}).get(sensor_name, {})
        max_age = sensor_cfg.get('max_age', 0.0)
        if max_age <= 0.0 or header.stamp.is_zero():
            return False
        return (rospy.Time.now() - header.stamp).to_sec() > max_age

    def gate_enabled_and_threshold(self, sensor_name, component, legacy_key, default_threshold):
        """Read Mahalanobis gate settings with explicit flags and old-key fallback."""
        sensor_gate = self.mahal_cfg.get(sensor_name, {})
        if sensor_gate:
            enabled = bool(sensor_gate.get(f'{component}_enabled',
                                           sensor_gate.get('enabled', False)))
            threshold = sensor_gate.get(f'{component}_threshold')
            if threshold is None and component in ('xy', 'z'):
                threshold = sensor_gate.get('position_threshold')
            if threshold is None:
                threshold = sensor_gate.get('threshold', default_threshold)
            return enabled, float(threshold)

        threshold = self.mahal_gates.get(legacy_key)
        if threshold is None and component in ('xy', 'z'):
            threshold = self.mahal_gates.get(f'{sensor_name}_position', -1.0)
        if threshold is None:
            threshold = -1.0
        threshold = float(threshold)
        return threshold > 0.0, threshold

    def mahalanobis_reject(self, innovation, S_inv, sensor_name, component, legacy_key, default_threshold):
        """Return (reject, distance², threshold) for optional Mahalanobis gates."""
        mahal_sq = float(innovation.T @ S_inv @ innovation)
        enabled, threshold = self.gate_enabled_and_threshold(
            sensor_name, component, legacy_key, default_threshold
        )
        return enabled and mahal_sq > threshold, mahal_sq, threshold

    # =========================================================================
    # PROCESS MODEL: PX4 VELOCITY
    # =========================================================================
    
    def px4_velocity_callback(self, msg):
        current_time = msg.header.stamp.to_sec()

        if self.last_process_stamp is not None:
            self.observed_process_dt = current_time - self.last_process_stamp
        self.last_process_stamp = current_time
        
        vz = msg.twist.linear.z
        
        # ← ADD THIS BLOCK
        use_z = self.config['sensors']['px4_velocity'].get('use_z', True)
        if not use_z:
            vz = 0.0
        
        self.last_body_velocity = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            vz   
        ])
        
        self.last_angular_velocity = np.array([
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z
        ])
        
        if self.mode != EKFState.TRACKING:
            return
        
        self.predict_step(msg.header)

    # =========================================================================
    # EKF PREDICTION STEP
    # =========================================================================
    
    def predict_step(self, header):
        """EKF Prediction Step - integrates velocity to predict state."""
        
        # Get body frame velocity
        v_body = self.last_body_velocity.copy()
        
        # Get current orientation estimate from EKF state
        self.normalize_state_quaternion()
        q_prior = self.state[6:10].copy()
        R_body_to_landpad = self.quaternion_to_rotation_matrix(q_prior)
        
        # Transform body velocity to landpad frame
        v_landpad = R_body_to_landpad @ v_body
        
        # === EKF Position Integration ===
        self.state[0] += v_landpad[0] * self.dt
        self.state[1] += v_landpad[1] * self.dt
        self.state[2] += v_landpad[2] * self.dt
        self.state[3:6] = v_landpad
        
        # === Orientation Integration ===
        omega_body = self.last_angular_velocity
        delta_angle = omega_body * self.dt
        delta_quat = self.rotation_vector_to_quaternion(delta_angle)
        
        # State quaternion maps drone/body coordinates into the landpad frame.
        # PX4 velocity_body angular rates are body-frame rates, so the
        # incremental rotation must be applied on the right: q <- q ⊗ δq_body.
        self.state[6:10] = self.quaternion_multiply(self.state[6:10], delta_quat)
        self.state[6:10] = self.normalize_quaternion(self.state[6:10])
        
        # Dead reckoning exists only as a comparison/debug stream.
        if not self.flight_optimized:
            R_dr = self.quaternion_to_rotation_matrix(self.dead_reckoning_quat)
            v_landpad_dr = R_dr @ v_body
            self.dead_reckoning_pos += v_landpad_dr * self.dt
            self.dead_reckoning_quat = self.quaternion_multiply(
                self.dead_reckoning_quat, delta_quat
            )
            self.dead_reckoning_quat = self.normalize_quaternion(
                self.dead_reckoning_quat
            )
        
        # === Covariance Prediction ===
        vel_q_jac = self.rotated_velocity_quaternion_jacobian(q_prior, v_body)
        F = np.eye(10)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:10] = vel_q_jac * self.dt
        F[3:6, 6:10] = vel_q_jac
        F[6:10, 6:10] = self.right_quaternion_product_matrix(delta_quat)
        self.P = F @ self.P @ F.T + self.Q * self.dt
        self.normalize_state_quaternion()
        
        # === Logging ===
        if self.debug_config['log_prediction']:
            rospy.loginfo_throttle(2.0, 
                f"[PREDICT] v_body: [{v_body[0]:.3f}, {v_body[1]:.3f}, {v_body[2]:.3f}] → "
                f"v_landpad: [{v_landpad[0]:.3f}, {v_landpad[1]:.3f}, {v_landpad[2]:.3f}] | "
                f"dt: {self.dt:.3f} | pos: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}]")
        
        self.record_prediction_dashboard(header, v_body, v_landpad, omega_body)
        self.publish_timing_debug(
            header, 'prediction', sensor='px4_velocity',
            dt=self.dt, observed_dt=self.observed_process_dt
        )
        self.publish_estimates(header, source='prediction')
    # =========================================================================
    # MEASUREMENT UPDATE: ARUCO
    # =========================================================================
    
    def aruco_marker_callback(self, msg, marker_id):
        """Per-marker ArUco callback. Transforms and updates EKF with marker-specific noise."""
        callback_start_wall = time.perf_counter()
        callback_start_stamp = rospy.Time.now().to_sec()
        timings = {
            'stale_check_ms': 0.0,
            'transform_ms': 0.0,
            'validation_ms': 0.0,
            'measurement_publish_ms': 0.0,
            'initialization_ms': 0.0,
            'xy_update_ms': 0.0,
            'z_update_ms': 0.0,
            'yaw_update_ms': 0.0,
            'estimate_publish_ms': 0.0,
        }

        stage_start = time.perf_counter()
        stale = self.is_stale(msg.header, 'aruco')
        timings['stale_check_ms'] = (time.perf_counter() - stage_start) * 1000.0
        if stale:
            rospy.logwarn_throttle(2.0, f"[ARUCO] Dropping stale marker {marker_id} measurement")
            self.publish_aruco_callback_timing(
                msg.header, marker_id, callback_start_stamp,
                callback_start_wall, 'rejected_stale', timings
            )
            return

        # Transform ArUco measurement to drone pose in landpad frame
        stage_start = time.perf_counter()
        pos_drone_landpad, quat_drone_landpad = self.transform_aruco_to_drone_frame(msg)
        timings['transform_ms'] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        valid_pose = self.finite_pose(pos_drone_landpad, quat_drone_landpad)
        timings['validation_ms'] = (time.perf_counter() - stage_start) * 1000.0
        if not valid_pose:
            rospy.logwarn_throttle(2.0, f"[ARUCO] Dropping invalid marker {marker_id} measurement")
            self.publish_aruco_callback_timing(
                msg.header, marker_id, callback_start_stamp,
                callback_start_wall, 'rejected_invalid', timings
            )
            return
        
        # Update sensor status
        self.sensor_status.last_aruco_time = rospy.Time.now().to_sec()
        
        # Publish raw measurement for visualization
        stage_start = time.perf_counter()
        self.publish_aruco_measurement(msg.header, pos_drone_landpad, quat_drone_landpad, marker_id)
        timings['measurement_publish_ms'] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        
        # State machine - initialize from first ArUco
        if self.mode == EKFState.WAITING:
            stage_start = time.perf_counter()
            self.initialize_from_aruco(pos_drone_landpad, quat_drone_landpad)
            rospy.loginfo(f"[EKF] First ArUco detected (marker {marker_id})! Switching to TRACKING mode.")
            rospy.loginfo(f"[EKF] Initial position: [{pos_drone_landpad[0]:.3f}, "
                        f"{pos_drone_landpad[1]:.3f}, {pos_drone_landpad[2]:.3f}]")
            yaw = np.degrees(tft.euler_from_quaternion(quat_drone_landpad)[2])
            rospy.loginfo(f"[EKF] Initial yaw: {yaw:.1f}°")
            self.mode = EKFState.TRACKING

            # Force dashboard refresh immediately after initialization.
            self.maybe_render_debug_dashboard(force=True)
            timings['initialization_ms'] = (
                time.perf_counter() - stage_start
            ) * 1000.0
            self.publish_aruco_callback_timing(
                msg.header, marker_id, callback_start_stamp,
                callback_start_wall, 'initialized', timings
            )
            return
        
        # Kalman update with marker-specific noise
        self.update_aruco(
            pos_drone_landpad, quat_drone_landpad,
            msg.header, marker_id, timings
        )
        self.publish_aruco_callback_timing(
            msg.header, marker_id, callback_start_stamp,
            callback_start_wall, 'updated', timings
        )
        
        
    
    # def aruco_callback(self, msg): DEPRECATED - using per-marker callback instead
    #     """Measurement update from ArUco marker detection."""
    #     # Transform ArUco measurement to drone pose in landpad frame
    #     pos_drone_landpad, quat_drone_landpad = self.transform_aruco_to_drone_frame(msg)
        
    #     # Update sensor status
    #     self.sensor_status.last_aruco_time = rospy.Time.now().to_sec()
        
    #     # Publish raw measurement
    #     self.publish_aruco_measurement(msg.header, pos_drone_landpad, quat_drone_landpad)
        
    #     # State machine - initialize from first ArUco
    #     if self.mode == EKFState.WAITING:
    #         self.initialize_from_aruco(pos_drone_landpad, quat_drone_landpad)
    #         rospy.loginfo("[EKF] First ArUco detected! Switching to TRACKING mode.")
    #         rospy.loginfo(f"[EKF] Initial position: [{pos_drone_landpad[0]:.3f}, "
    #                      f"{pos_drone_landpad[1]:.3f}, {pos_drone_landpad[2]:.3f}]")
    #         yaw = np.degrees(tft.euler_from_quaternion(quat_drone_landpad)[2])
    #         rospy.loginfo(f"[EKF] Initial yaw: {yaw:.1f}°")
    #         self.mode = EKFState.TRACKING
    #         return
        
    #     # Kalman update
    #     self.update_aruco(pos_drone_landpad, quat_drone_landpad, msg.header)

    
    def transform_aruco_to_drone_frame(self, msg):
        """Transform detector output to drone pose in landpad frame."""
        # Detector publishes landpad pose in camera frame
        T_cam_landpad = self.pose_msg_to_matrix(msg.pose.position, msg.pose.orientation)
        
        # Chain: T_drone_landpad = T_drone_camera × T_camera_landpad
        T_drone_cam = self.transforms['T_cam_to_drone']
        T_drone_landpad = np.dot(T_drone_cam, T_cam_landpad)
        
        # Invert to obtain drone pose in landpad frame
        T_landpad_drone = tft.inverse_matrix(T_drone_landpad)
        
        pos_drone_landpad = T_landpad_drone[:3, 3]
        quat_drone_landpad = self.normalize_quaternion(tft.quaternion_from_matrix(T_landpad_drone))
        
        return pos_drone_landpad, quat_drone_landpad
    
    def initialize_from_aruco(self, position, quaternion):
        """Initialize EKF from first ArUco detection."""
        quaternion = self.normalize_quaternion(quaternion)
        self.state[0:3] = position
        self.state[3:6] = np.zeros(3)
        self.state[6:10] = quaternion
        
        self.dead_reckoning_pos = position.copy()
        self.dead_reckoning_quat = quaternion.copy()
        
        # Reset covariance
        self.init_covariance()
        # self.P *= 0.1
        
        rospy.loginfo(f"[EKF] Initialized at position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        yaw = np.degrees(tft.euler_from_quaternion(quaternion)[2])
        rospy.loginfo(f"[EKF] Initialized with yaw: {yaw:.1f}°")
    
    def update_aruco(self, z_pos, z_quat, header, marker_id=None,
                     timing_profile=None):
        """EKF update using independently gated ArUco x/y, z, and yaw."""
        sensors = self.config['sensors']['aruco']

        # Select marker-specific or default R matrices
        if marker_id and marker_id in self.R_aruco_per_marker_pos:
            R_pos = self.R_aruco_per_marker_pos[marker_id]
            R_yaw = self.R_aruco_per_marker_yaw[marker_id]
        else:
            R_pos = self.R_aruco_pos
            R_yaw = self.R_aruco_yaw

        if sensors.get('position_update', True):
            if sensors.get('xy_update', True):
                stage_start = time.perf_counter()
                self.update_position_xy(z_pos, R_pos, header, 'aruco', 'xy',
                                        'aruco_xy', 9.210,
                                        f"ARUCO XY marker={marker_id}",
                                        marker_id=marker_id)
                if timing_profile is not None:
                    timing_profile['xy_update_ms'] = (
                        time.perf_counter() - stage_start
                    ) * 1000.0
            if sensors.get('z_update', True):
                stage_start = time.perf_counter()
                self.update_position_z(z_pos, R_pos, header, 'aruco', 'z',
                                       'aruco_z', 6.635,
                                       f"ARUCO Z marker={marker_id}",
                                       marker_id=marker_id)
                if timing_profile is not None:
                    timing_profile['z_update_ms'] = (
                        time.perf_counter() - stage_start
                    ) * 1000.0

        if sensors.get('orientation_update', True):
            stage_start = time.perf_counter()
            self.update_yaw(z_quat, R_yaw, header, 'aruco', 'yaw',
                            'aruco_yaw', 6.635,
                            f"ARUCO YAW marker={marker_id}",
                            marker_id=marker_id)
            if timing_profile is not None:
                timing_profile['yaw_update_ms'] = (
                    time.perf_counter() - stage_start
                ) * 1000.0

        stage_start = time.perf_counter()
        self.publish_estimates(header)
        if timing_profile is not None:
            timing_profile['estimate_publish_ms'] = (
                time.perf_counter() - stage_start
            ) * 1000.0

    def transform_thermal_to_drone_frame(self, msg):
        """Transform thermal detector output to drone pose in landpad frame."""
        # Thermal detector should publish landpad pose in the thermal camera frame.
        T_thermal_landpad = self.pose_msg_to_matrix(msg.pose.position, msg.pose.orientation)
        T_drone_thermal = self.transforms.get('T_thermal_to_drone', self.transforms['T_cam_to_drone'])
        T_drone_landpad = np.dot(T_drone_thermal, T_thermal_landpad)
        T_landpad_drone = tft.inverse_matrix(T_drone_landpad)
        return T_landpad_drone[:3, 3], self.normalize_quaternion(tft.quaternion_from_matrix(T_landpad_drone))

    def thermal_callback(self, msg):
        """Thermal camera measurement update: x/y/z + yaw only."""
        if self.is_stale(msg.header, 'thermal'):
            rospy.logwarn_throttle(2.0, "[THERMAL] Dropping stale measurement")
            return

        pos_drone_landpad, quat_drone_landpad = self.transform_thermal_to_drone_frame(msg)
        if not self.finite_pose(pos_drone_landpad, quat_drone_landpad):
            rospy.logwarn_throttle(2.0, "[THERMAL] Dropping invalid measurement")
            return

        self.sensor_status.last_thermal_time = rospy.Time.now().to_sec()
        self.publish_thermal_measurement(msg.header, pos_drone_landpad, quat_drone_landpad)

        if self.mode == EKFState.WAITING:
            if self.config.get('sensors', {}).get('thermal', {}).get('allow_initialization', False):
                self.initialize_from_aruco(pos_drone_landpad, quat_drone_landpad)
                self.mode = EKFState.TRACKING
                rospy.loginfo("[EKF] Initialized from thermal measurement")
                self.maybe_render_debug_dashboard(force=True)
            return

        thermal_cfg = self.config.get('sensors', {}).get('thermal', {})
        if thermal_cfg.get('position_update', True):
            self.update_position_xyz(pos_drone_landpad, self.R_thermal_pos, msg.header,
                                     'thermal', 'position', 'thermal_position', 11.345,
                                     "THERMAL POS")
        if thermal_cfg.get('orientation_update', True):
            self.update_yaw(quat_drone_landpad, self.R_thermal_yaw, msg.header,
                            'thermal', 'yaw', 'thermal_yaw', 6.635,
                            "THERMAL YAW")
        self.publish_estimates(msg.header)

    def update_position_xyz(self, z_pos, R_pos, header, sensor_name, component,
                            legacy_gate_key, default_gate, log_prefix, marker_id=None):
        """Shared x/y/z position update. Returns True if rejected."""
        H = np.zeros((3, 10))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = z_pos - H @ self.state
        S = H @ self.P @ H.T + R_pos
        S_inv = np.linalg.inv(S)
        mahal_rejected, mahal_sq, gate = self.mahalanobis_reject(
            y, S_inv, sensor_name, component, legacy_gate_key, default_gate
        )
        rejection_reason = 'mahalanobis' if mahal_rejected else None
        if mahal_rejected:
            self.publish_update_debug(
                header, sensor_name, component, y, S, R_pos, False,
                mahal_sq, gate, marker_id=marker_id,
                rejection_reason=rejection_reason
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn_throttle(1.0,
                    f"[{log_prefix}] REJECTED Mahal²={mahal_sq:.2f} > gate={gate:.2f} | "
                    f"innovation=[{y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}]")
            return True

        K = self.P @ H.T @ S_inv
        self.state = self.state + K @ y
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_pos @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = z_pos - H @ self.state
        self.publish_update_debug(
            header, sensor_name, component, y, S, R_pos, True,
            mahal_sq, gate, K=K, marker_id=marker_id,
            post_fit_residual=post_fit_residual
        )
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[{log_prefix}] Mahal²={mahal_sq:.2f} Innovation="
                f"[{y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}]")
        return False

    def update_position_xy(self, z_pos, R_pos, header, sensor_name, component,
                           legacy_gate_key, default_gate, log_prefix, marker_id=None):
        """Shared x/y position update. Returns True if rejected."""
        H = np.zeros((2, 10))
        H[0, 0] = H[1, 1] = 1.0
        z = z_pos[:2]
        R_xy = R_pos[:2, :2]
        y = z - H @ self.state
        S = H @ self.P @ H.T + R_xy
        S_inv = np.linalg.inv(S)
        mahal_rejected, mahal_sq, gate = self.mahalanobis_reject(
            y, S_inv, sensor_name, component, legacy_gate_key, default_gate
        )
        rejection_reason = 'mahalanobis' if mahal_rejected else None
        if mahal_rejected:
            self.publish_update_debug(
                header, sensor_name, component, y, S, R_xy, False,
                mahal_sq, gate, marker_id=marker_id,
                rejection_reason=rejection_reason
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn_throttle(1.0,
                    f"[{log_prefix}] REJECTED Mahal²={mahal_sq:.2f} > gate={gate:.2f} | "
                    f"innovation=[{y[0]:.3f}, {y[1]:.3f}]")
            return True

        K = self.P @ H.T @ S_inv
        self.state = self.state + K @ y
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_xy @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = z - H @ self.state
        self.publish_update_debug(
            header, sensor_name, component, y, S, R_xy, True,
            mahal_sq, gate, K=K, marker_id=marker_id,
            post_fit_residual=post_fit_residual
        )
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[{log_prefix}] Mahal²={mahal_sq:.2f} Innovation="
                f"[{y[0]:.3f}, {y[1]:.3f}]")
        return False

    def update_position_z(self, z_pos, R_pos, header, sensor_name, component,
                          legacy_gate_key, default_gate, log_prefix, marker_id=None):
        """Shared z position update. Returns True if rejected."""
        H = np.zeros((1, 10))
        H[0, 2] = 1.0
        z = np.array([z_pos[2]])
        R_z = np.array([[R_pos[2, 2]]])
        y = z - H @ self.state
        S = H @ self.P @ H.T + R_z
        S_inv = np.linalg.inv(S)
        mahal_rejected, mahal_sq, gate = self.mahalanobis_reject(
            y, S_inv, sensor_name, component, legacy_gate_key, default_gate
        )
        rejection_reason = 'mahalanobis' if mahal_rejected else None
        if mahal_rejected:
            self.publish_update_debug(
                header, sensor_name, component, y, S, R_z, False,
                mahal_sq, gate, marker_id=marker_id,
                rejection_reason=rejection_reason
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn_throttle(1.0,
                    f"[{log_prefix}] REJECTED Mahal²={mahal_sq:.2f} > gate={gate:.2f} | "
                    f"innovation={y[0]:.3f}")
            return True

        K = self.P @ H.T @ S_inv
        self.state = self.state + (K @ y).flatten()
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_z @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = z - H @ self.state
        self.publish_update_debug(
            header, sensor_name, component, y, S, R_z, True,
            mahal_sq, gate, K=K, marker_id=marker_id,
            post_fit_residual=post_fit_residual
        )
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[{log_prefix}] Mahal²={mahal_sq:.2f} Innovation={y[0]:.3f}")
        return False

    def update_yaw(self, z_quat, R_yaw, header, sensor_name, component,
                   legacy_gate_key, default_gate, log_prefix, marker_id=None):
        """Yaw-only update using a small-angle error around landpad Z."""
        self.normalize_state_quaternion()
        yaw_est = self.yaw_from_quaternion(self.state[6:10])
        yaw_meas = self.yaw_from_quaternion(z_quat)
        y = np.array([self.wrap_angle(yaw_meas - yaw_est)])

        # The measurement is yaw, not qz. Linearize yaw(q) around the current
        # quaternion so the innovation units (rad) match the measurement model.
        H = np.zeros((1, 10))
        H[0, 6:10] = self.yaw_quaternion_jacobian(self.state[6:10])
        S = H @ self.P @ H.T + R_yaw
        S_inv = np.linalg.inv(S)
        mahal_rejected, mahal_sq, gate = self.mahalanobis_reject(
            y, S_inv, sensor_name, component, legacy_gate_key, default_gate
        )
        rejection_reason = 'mahalanobis' if mahal_rejected else None
        if mahal_rejected:
            self.publish_update_debug(
                header, sensor_name, component, y, S, R_yaw, False,
                mahal_sq, gate, marker_id=marker_id,
                rejection_reason=rejection_reason
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn_throttle(1.0,
                    f"[{log_prefix}] REJECTED Mahal²={mahal_sq:.2f} > gate={gate:.2f} | "
                    f"yaw innovation={np.degrees(y[0]):.2f} deg")
            return True

        K = self.P @ H.T @ S_inv
        delta_state = (K @ y).flatten()
        self.state[0:6] += delta_state[0:6]
        self.state[6:10] = self.state[6:10] + delta_state[6:10]

        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_yaw @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = np.array([
            self.wrap_angle(yaw_meas - self.yaw_from_quaternion(self.state[6:10]))
        ])
        measurement_gain = float((H @ K)[0, 0])
        self.publish_update_debug(
            header, sensor_name, component, y, S, R_yaw, True,
            mahal_sq, gate, K=K, marker_id=marker_id,
            post_fit_residual=post_fit_residual,
            measurement_gain=measurement_gain
        )
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[{log_prefix}] Mahal²={mahal_sq:.2f} Yaw innovation={np.degrees(y[0]):.2f} deg")
        return False

    # =========================================================================
    # MEASUREMENT UPDATE: LASER ALTIMETER
    # =========================================================================
    
    def laser_callback(self, msg):
        """Measurement update from laser altimeter (z only)."""
        if self.mode != EKFState.TRACKING:
            return
        
        # Get bottom clearance (height above ground)
        z_measured_sensor = msg.bottom_clearance
        
        if np.isnan(z_measured_sensor) or z_measured_sensor <= 0:
            return
        
        # Update sensor status
        self.sensor_status.last_laser_time = rospy.Time.now().to_sec()
        
        # Transform laser measurement to drone frame
        # Laser measures distance in sensor frame, transform to drone Z
        T_laser = self.transforms['T_laser_to_drone']
        
        # The laser measures distance along its Z axis
        # In drone frame, this becomes height (with offset from mounting)
        laser_offset_z = T_laser[2, 3]  # Z offset of laser mounting
        
        # Drone height in landpad frame = laser_measurement + offset
        # Note: Sign depends on laser orientation (pointing down = positive height)
        z_drone_landpad = z_measured_sensor - laser_offset_z
        
        # Publish measurement
        self.publish_laser_measurement(msg.header, z_drone_landpad)
        
        # === EKF Update (z only) with Mahalanobis gate ===
        H = np.zeros((1, 10))
        H[0, 2] = 1.0  # Measures z
        
        z = np.array([z_drone_landpad])
        y = z - H @ self.state
        
        S = H @ self.P @ H.T + self.R_laser
        S_inv = np.linalg.inv(S)
        
        # Optional legacy laser gate remains disabled by default.
        mahal_rejected, mahal_sq, gate = self.mahalanobis_reject(
            y, S_inv, 'laser', 'z', 'laser', 6.635
        )
        rejection_reason = 'mahalanobis' if mahal_rejected else None
        if mahal_rejected:
            self.publish_update_debug(
                msg.header, 'laser', 'z', y, S, self.R_laser, False,
                mahal_sq, gate,
                rejection_reason=rejection_reason
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn(
                    f"[LASER] REJECTED Mahal²={mahal_sq:.1f} > gate={gate:.1f} | "
                    f"innovation={y[0]:.3f}"
                )
            return
        
        K = self.P @ H.T @ S_inv
        self.state = self.state + (K @ y).flatten()
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_laser @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = z - H @ self.state
        self.publish_update_debug(
            msg.header, 'laser', 'z', y, S, self.R_laser, True,
            mahal_sq, gate, K=K, post_fit_residual=post_fit_residual
        )
        
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[LASER] Mahal²={mahal_sq:.2f} Innovation: {y[0]:.3f}")
        
        self.publish_estimates(msg.header)

    # =========================================================================
    # MEASUREMENT UPDATE: UWB
    # =========================================================================
    
    def uwb_callback(self, msg):
        """Measurement update from UWB (x, y only)."""
        if self.mode != EKFState.TRACKING:
            return

        if self.is_stale(msg.header, 'uwb'):
            rospy.logwarn_throttle(2.0, "[UWB] Dropping stale measurement")
            return

        uwb_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        uwb_pos_msg = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        if not self.finite_pose(uwb_pos_msg, uwb_quat):
            rospy.logwarn_throttle(2.0, "[UWB] Dropping invalid measurement")
            return
        
        # UWB position in UWB map frame
        uwb_pos_map = np.array([
            uwb_pos_msg[0],
            uwb_pos_msg[1],
            uwb_pos_msg[2],
            1.0
        ])
        
        # Transform UWB map frame to landpad frame
        T_uwb_landpad = self.transforms['T_uwb_map_to_landpad']
        uwb_pos_landpad = T_uwb_landpad @ uwb_pos_map
        
        # Apply UWB-to-drone offset
        T_uwb_drone = self.transforms['T_uwb_to_drone']
        uwb_offset = T_uwb_drone[:3, 3]
        
        # Drone position = UWB position - offset (rotated by drone orientation)
        R_drone = self.quaternion_to_rotation_matrix(self.state[6:10])
        drone_pos_landpad = uwb_pos_landpad[:3] - R_drone @ uwb_offset
        if not np.all(np.isfinite(drone_pos_landpad)):
            rospy.logwarn_throttle(2.0, "[UWB] Dropping invalid transformed measurement")
            return

        # Update sensor status after stale/finite gates pass.
        self.sensor_status.last_uwb_time = rospy.Time.now().to_sec()
        
        # Publish measurement
        self.publish_uwb_measurement(msg.header, drone_pos_landpad[:2])
        
        # === EKF Update (x, y only) with Mahalanobis gate ===
        H = np.zeros((2, 10))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        
        z = drone_pos_landpad[:2]
        y = z - H @ self.state
        
        S = H @ self.P @ H.T + self.R_uwb
        S_inv = np.linalg.inv(S)
        
        rejected, mahal_sq, gate = self.mahalanobis_reject(y, S_inv, 'uwb', 'xy', 'uwb', 9.210)
        if rejected:
            self.publish_update_debug(
                msg.header, 'uwb', 'xy', y, S, self.R_uwb, False,
                mahal_sq, gate
            )
            if self.debug_config['log_innovations']:
                rospy.logwarn(
                    f"[UWB] REJECTED Mahal²={mahal_sq:.1f} > gate={gate:.1f} | "
                    f"innovation=[{y[0]:.3f}, {y[1]:.3f}]"
                )
            return
        
        K = self.P @ H.T @ S_inv
        self.state = self.state + (K @ y).flatten()
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_uwb @ K.T
        self.normalize_state_quaternion()
        post_fit_residual = z - H @ self.state
        self.publish_update_debug(
            msg.header, 'uwb', 'xy', y, S, self.R_uwb, True,
            mahal_sq, gate, K=K, post_fit_residual=post_fit_residual
        )
        
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[UWB] Mahal²={mahal_sq:.2f} Innovation: [{y[0]:.3f}, {y[1]:.3f}]")
        
        self.publish_estimates(msg.header)

    # =========================================================================
    # PUBLISHERS
    # =========================================================================
    
    def publish_estimates(self, header, source='measurement'):
        """Publish EKF estimate and diagnostics.

        Dead reckoning is prediction-only because it only advances in
        predict_step(). Measurement updates change the fused EKF state but not
        the dead-reckoning state, so republishing it there creates misleading
        repeated samples in plots.
        """
        stamp_mode = self.config.get('output', {}).get('stamp_mode', 'current')
        if stamp_mode == 'measurement' and not header.stamp.is_zero():
            stamp = header.stamp
        else:
            stamp = rospy.Time.now()
            if stamp.is_zero() and not header.stamp.is_zero():
                stamp = header.stamp
        if not np.all(np.isfinite(self.state[0:6])):
            rospy.logwarn_throttle(
                1.0,
                "[EKF] Skipping estimate publish because position/velocity state is not finite"
            )
            return
        state_quat = self.normalize_quaternion(self.state[6:10])
        self.state[6:10] = state_quat
        
        # === EKF Pose ===
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "landpad"
        pose_msg.pose.position.x = self.state[0]
        pose_msg.pose.position.y = self.state[1]
        pose_msg.pose.position.z = self.state[2]
        pose_msg.pose.orientation.x = state_quat[0]
        pose_msg.pose.orientation.y = state_quat[1]
        pose_msg.pose.orientation.z = state_quat[2]
        pose_msg.pose.orientation.w = state_quat[3]
        self.ekf_pose_pub.publish(pose_msg)
        
        # === EKF Odometry ===
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "landpad"
        odom_msg.child_frame_id = "base_link_ekf"
        odom_msg.pose.pose = pose_msg.pose
        v_body = self.quaternion_to_rotation_matrix(state_quat).T @ self.state[3:6]
        odom_msg.twist.twist.linear.x = v_body[0]
        odom_msg.twist.twist.linear.y = v_body[1]
        odom_msg.twist.twist.linear.z = v_body[2]
        for i in range(3):
            odom_msg.pose.covariance[i*7] = self.P[i, i]
        self.ekf_odom_pub.publish(odom_msg)
        
        if source == 'prediction' and self.dead_reckoning_pub is not None:
            # === Dead Reckoning ===
            dead_reckoning_quat = self.normalize_quaternion(self.dead_reckoning_quat)
            self.dead_reckoning_quat = dead_reckoning_quat
            dr_msg = PoseStamped()
            dr_msg.header.stamp = stamp
            dr_msg.header.frame_id = "landpad"
            dr_msg.pose.position.x = self.dead_reckoning_pos[0]
            dr_msg.pose.position.y = self.dead_reckoning_pos[1]
            dr_msg.pose.position.z = self.dead_reckoning_pos[2]
            dr_msg.pose.orientation.x = dead_reckoning_quat[0]
            dr_msg.pose.orientation.y = dead_reckoning_quat[1]
            dr_msg.pose.orientation.z = dead_reckoning_quat[2]
            dr_msg.pose.orientation.w = dead_reckoning_quat[3]
            self.dead_reckoning_pub.publish(dr_msg)
        
        # === TF ===
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "landpad"
        t.child_frame_id = "drone_ekf"
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = self.state[2]
        t.transform.rotation.x = state_quat[0]
        t.transform.rotation.y = state_quat[1]
        t.transform.rotation.z = state_quat[2]
        t.transform.rotation.w = state_quat[3]
        self.tf_broadcaster.sendTransform(t)
        self.publish_covariance_debug(odom_msg.header, 'publish_estimate')
        
        if self.debug_config['log_covariance']:
            rospy.loginfo_throttle(2.0, 
                f"[COV] Pos: {np.diag(self.P[:3,:3])}, "
                f"Vel: {np.diag(self.P[3:6,3:6])}, "
                f"Ori: {np.diag(self.P[6:10,6:10])}")

        self.publish_diagnostics(odom_msg.header)
    
    def publish_aruco_measurement(self, header, position, quaternion, marker_id=None):
        """Publish transformed ArUco measurement."""
        if marker_id not in getattr(self, 'aruco_marker_meas_pubs', {}):
            return
        quaternion = self.normalize_quaternion(quaternion)
        msg = PoseStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "landpad"
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        self.aruco_marker_meas_pubs[marker_id].publish(msg)

    def publish_thermal_measurement(self, header, position, quaternion):
        """Publish transformed thermal measurement."""
        if self.thermal_meas_pub is None:
            return
        quaternion = self.normalize_quaternion(quaternion)
        msg = PoseStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "landpad"
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        self.thermal_meas_pub.publish(msg)
    
    def publish_laser_measurement(self, header, z_value):
        """Publish transformed laser measurement."""
        if self.laser_meas_pub is None:
            return
        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "landpad"
        msg.point.x = float('nan')  # Not measured
        msg.point.y = float('nan')  # Not measured
        msg.point.z = z_value
        self.laser_meas_pub.publish(msg)
    
    def publish_uwb_measurement(self, header, xy_values):
        """Publish transformed UWB measurement."""
        if self.uwb_meas_pub is None:
            return
        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "landpad"
        msg.point.x = xy_values[0]
        msg.point.y = xy_values[1]
        msg.point.z = float('nan')  # Not measured by UWB
        self.uwb_meas_pub.publish(msg)


if __name__ == '__main__':
    try:
        ekf = DroneEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
