# 🚀 Quick Start Guide - Drone EKF

## Installation (5 minutes)

### 1. Copy the package to your workspace
```bash
# Create workspace if you don't have one
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Copy this package into your workspace
cp -r /path/to/drone_ekf .

# Or clone if you've uploaded to git
# git clone <your-repo-url>
```

### 2. Install dependencies
```bash
# ROS dependencies (should already be installed)
sudo apt install ros-noetic-geometry-msgs ros-noetic-nav-msgs

# Python dependencies
pip3 install matplotlib numpy
```

### 3. Build
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 4. Add to bashrc (optional but recommended)
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Running (2 minutes)

### Terminal 1: Play your rosbag
```bash
cd /path/to/your/rosbags
rosbag play your_flight_data.bag
```

### Terminal 2: Run the EKF
```bash
# Easy way - everything at once
roslaunch drone_ekf ekf.launch

# Or separately for more control
rosrun drone_ekf ekf_node.py
```

### Terminal 3 (optional): Run plotter
```bash
rosrun drone_ekf plotter.py
```

---

## What You Should See

1. **Terminal output:**
   ```
   [INFO] EKF Node initialized!
   [INFO] Subscribing to: /mavros/local_position/odom
   [INFO] Subscribing to: /aruco/pose/marker_417
   [INFO] Publishing to: /ekf/pose and /ekf/odom
   ```

2. **Matplotlib window:**
   - 6 graphs showing position and velocity
   - Blue lines (raw data)
   - Red lines (filtered data)
   - Green dots (ArUco measurements)

3. **New topics:**
   ```bash
   rostopic list | grep ekf
   # Should show:
   # /ekf/pose
   # /ekf/odom
   ```

---

## Tuning the Filter

The filter behavior is controlled by two noise matrices in `scripts/ekf_node.py`:

### If the filter is too noisy (jittery):
```python
# Line 24-25: DECREASE process noise
self.Q = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])  # Smaller values
```

### If the filter is too slow to respond:
```python
# Line 29: DECREASE measurement noise
self.R = np.diag([0.01, 0.01, 0.01])  # Smaller values = trust ArUco more
```

### If the filter diverges from raw data:
```python
# Line 24-25: INCREASE process noise
self.Q = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])  # Larger values
```

After changing parameters, restart the node:
- Press Ctrl+C
- Run again: `rosrun drone_ekf ekf_node.py`

---

## Verification Checklist

✅ Check these to ensure everything is working:

```bash
# 1. Is the EKF receiving data?
rostopic hz /mavros/local_position/odom
# Should show: average rate: ~XX.XX

# 2. Is the EKF publishing?
rostopic hz /ekf/pose
# Should show output

# 3. View filtered output
rostopic echo /ekf/pose
# Should show position updating

# 4. Compare to raw data
rostopic echo /mavros/local_position/odom
# Should be similar to EKF output
```

---

## Troubleshooting

### No data in plots?
- Wait 1-2 seconds for data to accumulate
- Check rosbag is playing: `rostopic hz /mavros/local_position/odom`

### Plot window doesn't appear?
```bash
export MPLBACKEND=TkAgg
python3 scripts/plotter.py
```

### "Package not found" error?
```bash
source ~/catkin_ws/devel/setup.bash
```

### Topics not showing up?
```bash
# Check if rosbag is playing
rostopic list | grep mavros

# Play rosbag if needed
rosbag play --clock your_data.bag
```

---

## Next Steps

1. ✅ Get basic version working (start here!)
2. Tune Q and R matrices while watching plots
3. Record your filtered output:
   ```bash
   rosbag record /ekf/pose /ekf/odom -O filtered_flight.bag
   ```
4. Add IMU data to the process model
5. Include orientation estimation
6. Integrate into your drone's control software

---

## File Overview

```
drone_ekf/
├── scripts/
│   ├── ekf_node.py       ← Main EKF implementation (START HERE)
│   └── plotter.py        ← Real-time visualization
├── launch/
│   └── ekf.launch        ← Start everything
├── README.md             ← Detailed documentation
├── ROS_BEGINNER_GUIDE.md ← Learn ROS concepts
└── QUICKSTART.md         ← This file!
```

---

## Getting Help

1. **Read the code comments** - Extensively documented!
2. **Check ROS_BEGINNER_GUIDE.md** - Explains ROS concepts
3. **See README.md** - Full documentation
4. **Test incrementally** - One feature at a time

---

**Ready? Let's go! 🚀**

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roslaunch drone_ekf ekf.launch
```
