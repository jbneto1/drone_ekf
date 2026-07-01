# Drone EKF - Sensor Fusion Package

Extended Kalman Filter for fusing drone odometry with ArUco marker measurements.

## 📋 Overview

This package implements a simple EKF that:
- **Predicts** drone state using PX4 body velocity from `/mavros/local_position/velocity_body`
- **Corrects** predictions using per-marker ArUco measurements such as `/aruco/pose/marker_417` (update step)
- **Publishes** filtered estimates to `/ekf/pose` and `/ekf/odom`
- **Plots** raw vs filtered data in real-time for tuning

### Fixed process-model period

The PX4/MAVROS body-velocity stream in the recorded flight data averages
29.998 Hz, so the EKF integrates every process sample with the configured
`sensors.px4_velocity.sample_rate_hz: 30.0` (`dt = 1/30 s`). Message timestamp
spacing is still published as `observed_dt` on `/ekf/debug/timing`, but network,
queue, and callback jitter cannot change the integration period.

### ArUco latency profiling

Keep these JSON topics in flight bags:

- `/stereo/debug/timing`: V4L2 read/decode, image split/message construction,
  ROS publish calls, and camera pipeline time.
- `/aruco/debug/timing`: image age at callback, `cv_bridge`, rectification,
  left/right detection, matching, gates, raw PnP, stereo PnP, result/pose
  publication, whole callback time, and camera sequence gaps.
- `/ekf/debug/timing`: pose age at the EKF plus transform, validation,
  measurement publication, x/y, z, yaw, estimate publication, and whole
  callback time.

Run `roslaunch drone_ekf plotter.launch` while collecting data and stop it
normally to generate `aruco_latency_profile_final.png` in the configured plot
directory. The plotter correlates records by the original frame stamp and
marker ID. In particular:

- `ROS left/right image transport/queue` is subscriber delivery minus that
  image's publish completion; `Stereo sync dispatch` then isolates the
  message-filter handoff into the paired callback.
- `ROS images→synchronized callback` is the combined image publish-to-detector
  boundary for comparison.
- `ROS pose transport/queue` is EKF callback start minus marker-pose publish
  completion.
- Camera sequence gaps show frames discarded while the single-threaded
  detector could not keep up.

The camera header is stamped immediately after `VideoCapture::read()` because
this camera path does not expose a hardware exposure timestamp.
`capture_read_ms` therefore reports driver/acquisition blocking separately and
is not part of the header-age calculation.

<<<<<<< HEAD
### Detector pose mode and flight-optimized operation

`iris_land/config/aruco_detector.yaml` selects the pose pipeline:

```yaml
pose_estimation_mode: 'stereo'  # stereo or monocular
flight_optimized: false
```

`stereo` is the existing left→right→left sequential refinement. Its final pose
is expressed in the left-camera frame. `monocular` subscribes only to the left
image and uses OpenCV 4.5.4 `aruco::estimatePoseSingleMarkers()` with
`SOLVEPNP_IPPE_SQUARE`; it preserves that same output frame and therefore uses
the same EKF `T_cam_to_drone` transform.

For deployment, set `flight_optimized: true` in both
`aruco_detector.yaml` and `drone_ekf/config/ekf_params.yaml`. The detector keeps
marker pose output active but disables optional images, dashboards, timing and
quality JSON, and diagnostic single-camera comparisons. In monocular mode the
camera publisher also stops constructing/publishing the unused right ROS image.
The EKF keeps fused pose, odometry, TF, and all filter updates active while
disabling its plot/debug/diagnostic streams and dead-reckoning comparison.

The detector also applies a configurable rotation-continuity gate after pose
estimation to reject implausible frame-to-frame planar-marker orientation
flips. It compares rotation matrices, so equivalent quaternion signs are not
mistaken for a flip.

=======
>>>>>>> e84c788dbc0ef7ce1361fea8f839097d2e8d50db
## 🔧 Installation

### 1. Build the package
```bash
cd ~/drone_ekf_ws
catkin_make
source devel/setup.bash
```

### 2. Install Python dependencies
```bash
pip install matplotlib numpy
```

## 🚀 Usage

### Step 1: Play your rosbag
```bash
rosbag play your_flight_data.bag
```

### Step 2: Run the EKF
```bash
# Option A: Run everything with launch file
roslaunch drone_ekf ekf.launch

# Option B: Run nodes separately
rosrun drone_ekf ekf_node.py      # Terminal 1
rosrun drone_ekf plotter.py        # Terminal 2
```

To launch the plotter and follow the detached saver until it finishes:

```bash
bash "$(rospack find drone_ekf)/scripts/run_plotter_and_watch.sh"
```

`plotter.save_dir` and `plotter.shutdown_save_log` support absolute paths,
`~`, and environment variables. Relative paths such as `./ekf_plots` are
resolved against the `drone_ekf` package directory—not roslaunch's `~/.ros`
working directory. The watcher receives the resolved log path and saver state
directly from the plotter, so it does not pipe `roslaunch` output or assume
`/tmp/ekf_plots`.

### Step 3: Monitor topics
```bash
# Check what's being published
rostopic list | grep ekf

# Echo filtered pose
rostopic echo /ekf/pose

# Check message rate
rostopic hz /ekf/odom
```

## 📊 Understanding the Output

### Published Topics
- `/ekf/pose` (geometry_msgs/PoseStamped): Filtered position
- `/ekf/odom` (nav_msgs/Odometry): Filtered position + velocity

### Plot Interpretation
The plotter shows 6 graphs:
- **Top row**: Position (X, Y, Z)
  - Blue = Raw odometry
  - Red = EKF filtered
  - Green dots = ArUco measurements
- **Bottom row**: Velocity (Vx, Vy, Vz)
  - Blue = Raw
  - Red = Filtered

**What to look for:**
- EKF should smooth noisy raw data
- Green ArUco dots should pull the red line closer
- If red line diverges from blue → increase process noise Q
- If red line is too jittery → decrease measurement noise R

## 🎛️ Tuning the EKF

Edit `scripts/ekf_node.py` lines 22-28:

### Process Noise (Q)
Controls trust in prediction model
```python
self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
#              [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
```
- **Increase** → Filter relies more on measurements (less smooth)
- **Decrease** → Filter relies more on prediction (smoother, slower correction)

### Measurement Noise (R)
Controls trust in ArUco measurements
```python
self.R = np.diag([0.05, 0.05, 0.05])  # [x, y, z]
```
- **Increase** → Less trust in ArUco (slower correction)
- **Decrease** → More trust in ArUco (faster correction)

## 🧪 Testing Process

1. **Run without ArUco** first
   - Disable ArUco subscriber to test process model alone
   - EKF should track odometry closely

2. **Add ArUco measurements**
   - Re-enable ArUco subscriber
   - Watch green dots correct the red line

3. **Tune parameters**
   - Adjust Q and R based on plot behavior
   - Iterate until performance is satisfactory

## 📐 EKF Equations (For Reference)

**State vector:**
```
x = [x, y, z, vx, vy, vz]ᵀ
```

**Prediction:**
```
x_pred = F·x + B·u
P_pred = F·P·Fᵀ + Q
```

**Update:**
```
K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
x_new = x_pred + K·(z - H·x_pred)
P_new = (I - K·H)·P_pred
```

## 🐛 Troubleshooting

**No plot appearing?**
- Check matplotlib backend: `export MPLBACKEND=TkAgg`
- Verify Python 3: `python3 --version`

**EKF not updating?**
- Check topics: `rostopic hz /mavros/local_position/odom`
- Verify timestamps are increasing

**Plot is empty?**
- Wait ~1-2 seconds for data to accumulate
- Check rosbag is playing

**ArUco not showing?**
- Normal if ArUco detections are sparse
- They'll appear as green dots when detected

## 📚 ROS Concepts Used

- **Subscribers**: Listen to topics (odom, aruco)
- **Publishers**: Send filtered data
- **Callbacks**: Functions triggered when messages arrive
- **Message types**: PoseStamped, Odometry
- **Launch files**: Start multiple nodes together

## 🎯 Next Steps

1. ✅ Get basic EKF working (you are here!)
2. Add IMU data for better prediction
3. Include orientation in state vector
4. Add transform broadcaster for RViz
5. Integrate into drone's main software

## 📞 Support

Check parameter tuning in the code comments!
