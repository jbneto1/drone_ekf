# Drone EKF - Sensor Fusion Package

Extended Kalman Filter for fusing drone odometry with ArUco marker measurements.

## 📋 Overview

This package implements a simple EKF that:
- **Predicts** drone state using `/mavros/local_position/odom` (process model)
- **Corrects** predictions using `/aruco/pose` measurements (update step)
- **Publishes** filtered estimates to `/ekf/pose` and `/ekf/odom`
- **Plots** raw vs filtered data in real-time for tuning

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
