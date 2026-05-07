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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, PointStamped
from mavros_msgs.msg import Altitude, OpticalFlowRad
from sensor_msgs.msg import Range
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
        self.optical_flow_active = False
        self.last_aruco_time = 0.0
        self.last_laser_time = 0.0
        self.last_uwb_time = 0.0
        self.last_flow_time = 0.0
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
        
        # Timing
        self.last_time = None
        self.dt = 0.033
        
        # Velocity storage
        self.last_body_velocity = np.zeros(3)
        self.last_angular_velocity = np.zeros(3)
        
        # Optical flow specific
        self.last_flow_time = None
        self.last_ground_distance = 1.0
        
        # Sensor status tracking
        self.sensor_status = SensorStatus()
        
        # TF broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Setup subscribers and publishers
        self.setup_subscribers()
        self.setup_publishers()
        
        # Publish static landpad frame
        self.publish_static_landpad_frame()
        
        # Status publishing timer
        rospy.Timer(rospy.Duration(1.0), self.publish_sensor_status)
        
        # NEW: Alternative initialization if ArUco not enabled
        wait_for_aruco = self.config['initialization'].get('wait_for_aruco', True)
        aruco_enabled = self.config['sensors']['aruco']['enabled']
        
        if not wait_for_aruco or not aruco_enabled:
            # Initialize at default position or wait for UWB
            rospy.Timer(rospy.Duration(2.0), self.try_alternative_initialization, oneshot=True)
        
        rospy.loginfo("[EKF] Node initialized (V4 - Modular)")
        rospy.loginfo(f"[EKF] Process model: {self.config['process_model']['type']}")
        rospy.loginfo(f"[EKF] ArUco: {self.config['sensors']['aruco']['enabled']}")
        rospy.loginfo(f"[EKF] Laser: {self.config['sensors']['laser']['enabled']}")
        rospy.loginfo(f"[EKF] UWB: {self.config['sensors']['uwb']['enabled']}")
        
        if wait_for_aruco and aruco_enabled:
            rospy.loginfo("[EKF] Waiting for first ArUco detection...")
        else:
            rospy.loginfo("[EKF] ArUco initialization disabled - using alternative init")
    
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
        
        # Load from config instead of hardcoding
        default_position = np.array(
            self.config['initialization'].get('default_position', [0.0, 0.0, 0.1])
        )
        default_quaternion = np.array(
            self.config['initialization'].get('default_orientation', [0.0, 0.0, 0.0, 1.0])
        )
        
        # Validate quaternion
        default_quaternion = self.normalize_quaternion(default_quaternion)
        
        # Initialize state
        self.state[0:3] = default_position
        self.state[3:6] = np.zeros(3)
        self.state[6:10] = default_quaternion
        
        # Initialize dead reckoning
        self.dead_reckoning_pos = default_position.copy()
        self.dead_reckoning_quat = default_quaternion.copy()
        
        # Initialize covariance with higher uncertainty
        self.init_covariance()
        cov_scale = self.config['initialization'].get('alternative_covariance_scale', 2.0)
        self.P *= cov_scale
        
        # Switch to tracking
        self.mode = EKFState.TRACKING
        
        # Log initialization
        rospy.loginfo("="*60)
        rospy.loginfo("[INIT] Alternative initialization complete")
        rospy.loginfo(f"[INIT] Position: [{default_position[0]:.3f}, "
                    f"{default_position[1]:.3f}, {default_position[2]:.3f}]")
        rospy.loginfo(f"[INIT] Orientation: [{default_quaternion[0]:.3f}, "
                    f"{default_quaternion[1]:.3f}, {default_quaternion[2]:.3f}, "
                    f"{default_quaternion[3]:.3f}]")
        
        # Convert to Euler for readability
        roll, pitch, yaw = tft.euler_from_quaternion(default_quaternion)
        rospy.loginfo(f"[INIT] Euler (deg): roll={np.degrees(roll):.1f}, "
                    f"pitch={np.degrees(pitch):.1f}, yaw={np.degrees(yaw):.1f}")
        rospy.loginfo(f"[INIT] Covariance scale: {cov_scale}x")
        rospy.loginfo("="*60)
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
        self.debug_config = self.config.get('debug', {
            'log_transforms': False,
            'log_innovations': False,
            'log_covariance': False,
            'log_prediction': False,
            'log_measurements': False
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
            'T_uwb_to_drone', 'T_flow_to_drone'
        ]
        
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
        self.R_aruco_pos = np.diag(self.config['measurement_noise']['aruco']['position'])
        self.R_aruco_ori = np.diag(self.config['measurement_noise']['aruco']['orientation_rvec'])
        self.R_laser = np.diag(self.config['measurement_noise']['laser']['z'])
        self.R_uwb = np.diag(self.config['measurement_noise']['uwb']['xy'])
        self.R_flow = np.diag(self.config['measurement_noise']['optical_flow']['velocity_xy'])
        
        # Per-marker R matrices (fall back to default if not specified)
        self.R_aruco_per_marker_pos = {}
        self.R_aruco_per_marker_ori = {}
        markers_cfg = self.config['sensors']['aruco'].get('markers', {})
        for marker_id_str, marker_cfg in markers_cfg.items():
            mid = int(marker_id_str)
            self.R_aruco_per_marker_pos[mid] = np.diag(
                marker_cfg.get('position_noise',
                               self.config['measurement_noise']['aruco']['position'])
            )
            self.R_aruco_per_marker_ori[mid] = np.diag(
                marker_cfg.get('orientation_noise_rvec',
                               self.config['measurement_noise']['aruco']['orientation_rvec'])
            )
            rospy.loginfo(f"[EKF] Marker {mid} R_pos diag: {np.diag(self.R_aruco_per_marker_pos[mid])}")
        
        # Mahalanobis gate thresholds
        self.mahal_gates = self.config.get('mahalanobis_gates', {
            'aruco_position': 11.345,
            'aruco_orientation': 11.345,
            'laser': 6.635,
            'uwb': 9.210
        })
    
    def init_covariance(self):
        """Initialize covariance matrix from config."""
        p_pos = self.config['initialization']['P_init']['position']
        p_vel = self.config['initialization']['P_init']['velocity']
        p_ori = self.config['initialization']['P_init']['orientation']
        self.P = np.diag(p_pos + p_vel + p_ori)

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
            
        elif process_type == 'Optical_Flow' and sensors['optical_flow']['enabled']:
            rospy.Subscriber(
                sensors['optical_flow']['flow_topic'],
                OpticalFlowRad, self.optical_flow_callback, queue_size=1
            )
            rospy.Subscriber(
                sensors['optical_flow']['range_topic'],
                Range, self.flow_range_callback, queue_size=1
            )
            rospy.loginfo(f"[EKF] Subscribed to Optical Flow: {sensors['optical_flow']['flow_topic']}")
        
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
        self.dead_reckoning_pub = rospy.Publisher(topics['dead_reckoning'], PoseStamped, queue_size=1)
        
        # Per-sensor measurement publishers (transformed to landpad frame)
        self.aruco_meas_pub = rospy.Publisher(topics['aruco_measurement'], PoseStamped, queue_size=1)
        self.laser_meas_pub = rospy.Publisher(topics['laser_measurement'], PointStamped, queue_size=1)
        self.uwb_meas_pub = rospy.Publisher(topics['uwb_measurement'], PointStamped, queue_size=1)
        
        # Sensor status publisher
        self.sensor_status_pub = rospy.Publisher(topics['sensor_status'], String, queue_size=1)
        
        # Diagnostics publisher
        self.diagnostics_pub = rospy.Publisher('/ekf/diagnostics', DiagnosticArray, queue_size=1)
    
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
            'process_model': self.config['process_model']['type']
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.sensor_status_pub.publish(msg)
        
    def publish_diagnostics(self, header):
        """Publish EKF diagnostics for monitoring."""
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
        status.values.append(KeyValue("dt", f"{self.dt:.4f}"))
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)

    # =========================================================================
    # MATH HELPER FUNCTIONS
    # =========================================================================
    
    def pose_msg_to_matrix(self, position, orientation):
        """Convert ROS pose message to 4x4 transformation matrix."""
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
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
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            return q / norm
        return np.array([0.0, 0.0, 0.0, 1.0])

    # =========================================================================
    # PROCESS MODEL: PX4 VELOCITY
    # =========================================================================
    
    def px4_velocity_callback(self, msg):
        current_time = msg.header.stamp.to_sec()
        
        if self.last_time is not None:
            self.dt = np.clip(current_time - self.last_time, 0.001, 0.5)
        self.last_time = current_time
        
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
    # PROCESS MODEL: OPTICAL FLOW
    # =========================================================================
    
    def flow_range_callback(self, msg):
        """Update ground distance from range sensor."""
        if msg.range > msg.min_range and msg.range < msg.max_range:
            self.last_ground_distance = msg.range
    
    def optical_flow_callback(self, msg):
        """Process optical flow data for velocity estimation."""
        current_time = msg.header.stamp.to_sec()
        
        # Check quality threshold
        min_quality = self.config['sensors']['optical_flow']['min_quality']
        if msg.quality < min_quality:
            return
        
        # Calculate dt
        if self.last_flow_time is not None:
            self.dt = np.clip(current_time - self.last_flow_time, 0.001, 0.5)
        self.last_flow_time = current_time
        
        # Convert optical flow to velocity
        integration_time = msg.integration_time_us * 1e-6
        
        if integration_time > 0:
            # Flow sensor velocity (body frame)
            flow_vx = (msg.integrated_x / integration_time) * self.last_ground_distance
            flow_vy = (msg.integrated_y / integration_time) * self.last_ground_distance
            
            # Compensate for body rotation (gyro data in flow message)
            gyro_comp_x = msg.integrated_xgyro / integration_time * self.last_ground_distance
            gyro_comp_y = msg.integrated_ygyro / integration_time * self.last_ground_distance
            
            flow_vx -= gyro_comp_x
            flow_vy -= gyro_comp_y
            
            # Transform from flow sensor frame to drone body frame
            T_flow = self.transforms['T_flow_to_drone']
            R_flow = T_flow[:3, :3]
            
            v_flow_sensor = np.array([flow_vx, flow_vy, 0.0])
            v_body = R_flow @ v_flow_sensor
            
            # Store for process model
            self.last_body_velocity = v_body
            
            # Angular velocity from flow gyro
            self.last_angular_velocity = np.array([
                msg.integrated_xgyro / integration_time,
                msg.integrated_ygyro / integration_time,
                msg.integrated_zgyro / integration_time
            ])
        
        self.sensor_status.last_flow_time = current_time
        
        # Only proceed if tracking
        if self.mode != EKFState.TRACKING:
            return
        
        # STEP 1: Prediction using flow velocity
        self.predict_step(msg.header)
        
        # STEP 2: Measurement update to correct velocity estimate
        if integration_time > 0:
            self.update_optical_flow_velocity(v_body, msg.header)


    def update_optical_flow_velocity(self, z_velocity_body, header):
        """
        EKF Measurement Update using optical flow velocity.
        
        Corrects the velocity state [vx, vy, vz] in the EKF using flow measurements.
        This is different from prediction - here we're treating flow as a measurement
        to correct our velocity estimate.
        """
        # Transform measured velocity to landpad frame
        R_drone = self.quaternion_to_rotation_matrix(self.state[6:10])
        z_velocity_landpad = R_drone @ z_velocity_body
        
        # Measurement matrix: we measure vx and vy in landpad frame
        # State is [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        #           [0, 1, 2,  3,  4,  5,  6,  7,  8,  9]
        H = np.zeros((2, 10))
        H[0, 3] = 1.0  # Measures vx (index 3)
        H[1, 4] = 1.0  # Measures vy (index 4)
        
        # Innovation (measurement - prediction)
        z = z_velocity_landpad[:2]  # Only vx, vy (optical flow doesn't measure vz)
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_flow
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + (K @ y).flatten()
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_flow @ K.T
        
        # Normalize quaternion (in case numerical errors accumulated)
        self.state[6:10] = self.normalize_quaternion(self.state[6:10])
        
        # Publish updated estimates
        self.publish_estimates(header)
    
    def predict_step_from_flow(self, header):
        """Prediction step using optical flow velocity."""
        self.predict_step(header)

    # =========================================================================
    # EKF PREDICTION STEP
    # =========================================================================
    
    def predict_step(self, header):
        """EKF Prediction Step - integrates velocity to predict state."""
        
        # Get body frame velocity
        v_body = self.last_body_velocity.copy()
        
        # Get current orientation estimate from EKF state
        q_ekf = self.state[6:10]
        R_body_to_landpad = self.quaternion_to_rotation_matrix(q_ekf)
        
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
        
        self.state[6:10] = self.quaternion_multiply(delta_quat, self.state[6:10])
        self.state[6:10] = self.normalize_quaternion(self.state[6:10])
        
        # === Dead Reckoning Update (same logic) ===
        R_dr = self.quaternion_to_rotation_matrix(self.dead_reckoning_quat)
        v_landpad_dr = R_dr @ v_body
        self.dead_reckoning_pos += v_landpad_dr * self.dt
        self.dead_reckoning_quat = self.quaternion_multiply(delta_quat, self.dead_reckoning_quat)
        self.dead_reckoning_quat = self.normalize_quaternion(self.dead_reckoning_quat)
        
        # === Covariance Prediction ===
        F = np.eye(10)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt
        self.P = F @ self.P @ F.T + self.Q * self.dt
        
        # === Logging ===
        if self.debug_config['log_prediction']:
            rospy.loginfo_throttle(2.0, 
                f"[PREDICT] v_body: [{v_body[0]:.3f}, {v_body[1]:.3f}, {v_body[2]:.3f}] → "
                f"v_landpad: [{v_landpad[0]:.3f}, {v_landpad[1]:.3f}, {v_landpad[2]:.3f}] | "
                f"dt: {self.dt:.3f} | pos: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}]")
        
        self.publish_estimates(header)
    # =========================================================================
    # MEASUREMENT UPDATE: ARUCO
    # =========================================================================
    
    def aruco_marker_callback(self, msg, marker_id):
        """Per-marker ArUco callback. Transforms and updates EKF with marker-specific noise."""
        # Transform ArUco measurement to drone pose in landpad frame
        pos_drone_landpad, quat_drone_landpad = self.transform_aruco_to_drone_frame(msg)
        
        # Update sensor status
        self.sensor_status.last_aruco_time = rospy.Time.now().to_sec()
        
        # Publish raw measurement for visualization
        self.publish_aruco_measurement(msg.header, pos_drone_landpad, quat_drone_landpad)
        
        # State machine - initialize from first ArUco
        if self.mode == EKFState.WAITING:
            self.initialize_from_aruco(pos_drone_landpad, quat_drone_landpad)
            rospy.loginfo(f"[EKF] First ArUco detected (marker {marker_id})! Switching to TRACKING mode.")
            rospy.loginfo(f"[EKF] Initial position: [{pos_drone_landpad[0]:.3f}, "
                         f"{pos_drone_landpad[1]:.3f}, {pos_drone_landpad[2]:.3f}]")
            yaw = np.degrees(tft.euler_from_quaternion(quat_drone_landpad)[2])
            rospy.loginfo(f"[EKF] Initial yaw: {yaw:.1f}°")
            self.mode = EKFState.TRACKING
            return
        
        # Kalman update with marker-specific noise
        self.update_aruco(pos_drone_landpad, quat_drone_landpad, msg.header, marker_id)
        
        
    
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
        quat_drone_landpad = tft.quaternion_from_matrix(T_landpad_drone)
        
        return pos_drone_landpad, quat_drone_landpad
    
    def initialize_from_aruco(self, position, quaternion):
        """Initialize EKF from first ArUco detection."""
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
    
    def update_aruco(self, z_pos, z_quat, header, marker_id=None):
        """EKF Update Step using ArUco measurement with Mahalanobis gating."""
        sensors = self.config['sensors']['aruco']
        
        # Select marker-specific or default R matrices
        if marker_id and marker_id in self.R_aruco_per_marker_pos:
            R_pos = self.R_aruco_per_marker_pos[marker_id]
            R_ori = self.R_aruco_per_marker_ori[marker_id]
        else:
            R_pos = self.R_aruco_pos
            R_ori = self.R_aruco_ori
        
        # === Position Update ===
        if sensors['position_update']:
            H = np.zeros((3, 10))
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            
            y = z_pos - H @ self.state
            S = H @ self.P @ H.T + R_pos
            S_inv = np.linalg.inv(S)
            
            # Mahalanobis gate (chi-squared, 3 DOF)
            mahal_sq = float(y.T @ S_inv @ y)
            gate = self.mahal_gates.get('aruco_position', 11.345)
            
            if gate > 0 and mahal_sq > gate:
                if self.debug_config['log_innovations']:
                    rospy.logwarn(
                        f"[ARUCO POS] REJECTED marker={marker_id} "
                        f"Mahal²={mahal_sq:.1f} > gate={gate:.1f} | "
                        f"innovation=[{y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}]"
                    )
                return  # Skip entire update (position and orientation)
            
            K = self.P @ H.T @ S_inv
            self.state = self.state + K @ y
            I_KH = np.eye(10) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R_pos @ K.T
            
            if self.debug_config['log_innovations']:
                rospy.loginfo_throttle(2.0,
                    f"[ARUCO POS] marker={marker_id} Mahal²={mahal_sq:.2f} "
                    f"Innovation: [{y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}]"
                )
        
        # === Orientation Update ===
        if sensors['orientation_update']:
            q_est = self.state[6:10]
            q_meas = z_quat
            
            q_est_inv = self.quaternion_conjugate(q_est)
            q_err = self.quaternion_multiply(q_meas, q_est_inv)
            rot_err = self.quaternion_to_rotation_vector(q_err)
            
            H_q = np.zeros((3, 10))
            H_q[0:3, 6:9] = np.eye(3)
            
            y_rot = rot_err
            P_q = self.P[6:10, 6:10]
            S_rot = H_q[:, 6:10] @ P_q @ H_q[:, 6:10].T + R_ori
            S_rot_inv = np.linalg.inv(S_rot)
            
            # Mahalanobis gate (chi-squared, 3 DOF)
            mahal_sq_ori = float(y_rot.T @ S_rot_inv @ y_rot)
            gate_ori = self.mahal_gates.get('aruco_orientation', 11.345)
            
            if gate_ori > 0 and mahal_sq_ori > gate_ori:
                if self.debug_config['log_innovations']:
                    rospy.logwarn(
                        f"[ARUCO ORI] REJECTED marker={marker_id} "
                        f"Mahal²={mahal_sq_ori:.1f} > gate={gate_ori:.1f}"
                    )
                self.publish_estimates(header)
                return
            
            K_full = self.P[:, 6:10] @ H_q[:, 6:10].T @ S_rot_inv
            delta_state = K_full @ y_rot
            self.state[0:6] += delta_state[0:6]
            
            delta_q = self.rotation_vector_to_quaternion(delta_state[6:9])
            self.state[6:10] = self.quaternion_multiply(delta_q, q_est)
            self.state[6:10] = self.normalize_quaternion(self.state[6:10])
            
            I_KH = np.eye(10) - K_full @ H_q
            self.P = I_KH @ self.P @ I_KH.T + K_full @ R_ori @ K_full.T
            
            if self.debug_config['log_innovations']:
                rospy.loginfo_throttle(2.0,
                    f"[ARUCO ORI] marker={marker_id} Mahal²={mahal_sq_ori:.2f} "
                    f"Rot Error: {np.degrees(rot_err)}"
                )
        
        self.publish_estimates(header)
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
        
        # Mahalanobis gate (chi-squared, 1 DOF) - disabled if gate < 0
        mahal_sq = float(y.T @ S_inv @ y)
        gate = self.mahal_gates.get('laser', -1.0)
        
        if gate > 0 and mahal_sq > gate:
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
        
        # Update sensor status
        self.sensor_status.last_uwb_time = rospy.Time.now().to_sec()
        
        # UWB position in UWB map frame
        uwb_pos_map = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
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
        
        # Mahalanobis gate (chi-squared, 2 DOF) - disabled if gate < 0
        mahal_sq = float(y.T @ S_inv @ y)
        gate = self.mahal_gates.get('uwb', -1.0)
        
        if gate > 0 and mahal_sq > gate:
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
        
        if self.debug_config['log_innovations']:
            rospy.loginfo_throttle(2.0,
                f"[UWB] Mahal²={mahal_sq:.2f} Innovation: [{y[0]:.3f}, {y[1]:.3f}]")
        
        self.publish_estimates(msg.header)

    # =========================================================================
    # PUBLISHERS
    # =========================================================================
    
    def publish_estimates(self, header):
        """Publish EKF estimate, dead reckoning, and TF."""
        stamp = header.stamp
        
        # === EKF Pose ===
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "landpad"
        pose_msg.pose.position.x = self.state[0]
        pose_msg.pose.position.y = self.state[1]
        pose_msg.pose.position.z = self.state[2]
        pose_msg.pose.orientation.x = self.state[6]
        pose_msg.pose.orientation.y = self.state[7]
        pose_msg.pose.orientation.z = self.state[8]
        pose_msg.pose.orientation.w = self.state[9]
        self.ekf_pose_pub.publish(pose_msg)
        
        # === EKF Odometry ===
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "landpad"
        odom_msg.child_frame_id = "base_link_ekf"
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = self.state[3]
        odom_msg.twist.twist.linear.y = self.state[4]
        odom_msg.twist.twist.linear.z = self.state[5]
        for i in range(3):
            odom_msg.pose.covariance[i*7] = self.P[i, i]
        self.ekf_odom_pub.publish(odom_msg)
        
        # === Dead Reckoning ===
        dr_msg = PoseStamped()
        dr_msg.header.stamp = stamp
        dr_msg.header.frame_id = "landpad"
        dr_msg.pose.position.x = self.dead_reckoning_pos[0]
        dr_msg.pose.position.y = self.dead_reckoning_pos[1]
        dr_msg.pose.position.z = self.dead_reckoning_pos[2]
        dr_msg.pose.orientation.x = self.dead_reckoning_quat[0]
        dr_msg.pose.orientation.y = self.dead_reckoning_quat[1]
        dr_msg.pose.orientation.z = self.dead_reckoning_quat[2]
        dr_msg.pose.orientation.w = self.dead_reckoning_quat[3]
        self.dead_reckoning_pub.publish(dr_msg)
        
        # === TF ===
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "landpad"
        t.child_frame_id = "drone_ekf"
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = self.state[2]
        t.transform.rotation.x = self.state[6]
        t.transform.rotation.y = self.state[7]
        t.transform.rotation.z = self.state[8]
        t.transform.rotation.w = self.state[9]
        self.tf_broadcaster.sendTransform(t)
        
        if self.debug_config['log_covariance']:
            rospy.loginfo_throttle(2.0, 
                f"[COV] Pos: {np.diag(self.P[:3,:3])}, "
                f"Vel: {np.diag(self.P[3:6,3:6])}, "
                f"Ori: {np.diag(self.P[6:10,6:10])}")

        self.publish_diagnostics(header)
    
    def publish_aruco_measurement(self, header, position, quaternion):
        """Publish transformed ArUco measurement."""
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
        self.aruco_meas_pub.publish(msg)
    
    def publish_laser_measurement(self, header, z_value):
        """Publish transformed laser measurement."""
        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "landpad"
        msg.point.x = float('nan')  # Not measured
        msg.point.y = float('nan')  # Not measured
        msg.point.z = z_value
        self.laser_meas_pub.publish(msg)
    
    def publish_uwb_measurement(self, header, xy_values):
        """Publish transformed UWB measurement."""
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