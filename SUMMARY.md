# 📦 Drone EKF Package - Complete Summary

## What You're Getting

A complete ROS package for sensor fusion using an Extended Kalman Filter (EKF) that:
- ✅ Fuses drone odometry with ArUco marker measurements
- ✅ Provides real-time plotting for tuning
- ✅ Is fully documented for beginners
- ✅ Can be integrated into your drone's software

---

## 📋 Package Contents

```
drone_ekf/
├── CMakeLists.txt              # Build configuration
├── package.xml                 # Package metadata & dependencies
│
├── scripts/
│   ├── ekf_node.py            # Main EKF implementation (157 lines)
│   └── plotter.py             # Real-time visualization (209 lines)
│
├── launch/
│   └── ekf.launch             # Launch file to start everything
│
├── README.md                   # Detailed documentation
├── ROS_BEGINNER_GUIDE.md      # ROS concepts explained
└── QUICKSTART.md              # 5-minute setup guide
```

---

## 🎯 Answer to Your Question

**Q: Can we use `/mavros/local_position/pose` or do we need `/mavros/local_position/odom`?**

**A: Use `/mavros/local_position/odom`** ✅

**Why?**
- The EKF process model needs **velocity** to predict the next state
- `odom` provides: position, orientation, AND velocity (twist)
- `pose` only provides: position and orientation

The package is configured to use `/mavros/local_position/odom` for the process model.

---

## 🔬 How the EKF Works

### State Vector
```
x = [x, y, z, vx, vy, vz]ᵀ
   position    velocity
```

### Data Flow
```
┌─────────────────────────────────────────────────┐
│                   Rosbag                        │
│  (your recorded flight data)                    │
└──────────┬────────────────────┬─────────────────┘
           │                    │
           ▼                    ▼
    /mavros/local_           /aruco/pose/marker_*
    position/odom         (ArUco measurements)
    (odometry)
           │                    │
           └────────┬───────────┘
                    ▼
            ┌───────────────┐
            │   EKF Node    │  ← Prediction + Correction
            │ (ekf_node.py) │
            └───────┬───────┘
                    │
                    ▼
             /ekf/pose  ─────────→  Plotter
             /ekf/odom              (visualization)
```

### Process Model (Prediction)
```python
# Every time odometry arrives:
1. Extract velocity from odom
2. Predict: position += velocity × dt
3. Update uncertainty (covariance)
```

### Measurement Update (Correction)
```python
# Every time ArUco is detected:
1. Compare ArUco position with predicted position
2. Calculate Kalman gain (how much to trust measurement)
3. Correct the state estimate
4. Reduce uncertainty
```

---

## 🎛️ Tuning Parameters

Located in `scripts/ekf_node.py` (lines 22-30):

### Process Noise Covariance (Q)
```python
self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
```
- Controls trust in the process model
- **Higher** → More responsive, but noisier
- **Lower** → Smoother, but slower to respond

### Measurement Noise Covariance (R)
```python
self.R = np.diag([0.05, 0.05, 0.05])
```
- Controls trust in ArUco measurements
- **Higher** → Measurements have less influence
- **Lower** → Measurements have more influence

---

## 🚀 Quick Setup

### 1. Copy to your workspace
```bash
cp -r drone_ekf ~/catkin_ws/src/
cd ~/catkin_ws
```

### 2. Build
```bash
catkin_make
source devel/setup.bash
```

### 3. Run
```bash
# Terminal 1: Play rosbag
rosbag play your_flight.bag

# Terminal 2: Run EKF + Plotter
roslaunch drone_ekf ekf.launch
```

---

## 📊 Interpreting the Plots

The plotter shows 6 graphs:

### Position Plots (Top Row)
- **Blue line**: Raw odometry from PX4
- **Red line**: EKF filtered estimate
- **Green dots**: ArUco measurements (sparse)

**What to look for:**
- Red line should smooth blue line
- Green dots should pull red line toward them
- Red should not diverge significantly from blue

### Velocity Plots (Bottom Row)
- **Blue line**: Raw velocity from odometry
- **Red line**: EKF estimated velocity

**What to look for:**
- Less noisy than position
- Should track blue line closely

---

## 🔍 Verification Steps

### 1. Check topics are publishing
```bash
rostopic hz /mavros/local_position/odom   # Should show ~50-100 Hz
rostopic hz /aruco/pose/marker_417        # Varies (ArUco detection rate)
rostopic hz /ekf/pose                     # Should match odom rate
```

### 2. Inspect messages
```bash
rostopic echo /ekf/pose
```

### 3. Monitor in RViz (optional)
```bash
rviz
# Add displays for /ekf/pose and /mavros/local_position/pose
```

---

## 🧪 Testing Strategy

### Step 1: Process Model Only
```python
# In ekf_node.py, comment out ArUco subscriber (line 40-41)
# rospy.Subscriber('/aruco/pose/marker_417', PoseStamped, 
#                 self.aruco_callback, queue_size=1)
```
- EKF should track odometry closely
- No green dots in plots
- Red and blue lines should be nearly identical

### Step 2: Add Measurements
```python
# Uncomment ArUco subscriber
rospy.Subscriber('/aruco/pose/marker_417', PoseStamped, 
                self.aruco_callback, queue_size=1)
```
- Green dots appear when ArUco detected
- Red line corrects toward green dots
- Should improve estimate accuracy

### Step 3: Tune Parameters
- Adjust Q and R based on plot behavior
- Restart node after each change
- Compare performance

---

## 💡 Tips for Success

### For Beginners
1. **Read ROS_BEGINNER_GUIDE.md first** - Understand the basics
2. **Start with default parameters** - They're reasonable
3. **Watch the plots** - Visual feedback is invaluable
4. **Test incrementally** - Don't change too much at once

### For Tuning
1. **If too noisy**: Decrease Q (trust model more)
2. **If too slow**: Increase Q or decrease R
3. **If diverging**: Increase Q (trust model less)
4. **Record results**: Use rosbag to save filtered output

### For Integration
1. Test offline first (rosbag)
2. Verify performance thoroughly
3. Add as a node in your launch file
4. Monitor during live flights

---

## 📁 File Descriptions

### Core Files
- **ekf_node.py**: Main EKF implementation
  - Subscribes to odom and ArUco
  - Implements prediction and correction
  - Publishes filtered estimates
  - Well-commented for learning

- **plotter.py**: Real-time visualization
  - Plots raw vs filtered data
  - Shows ArUco measurements
  - Updates at 5 Hz
  - Useful for tuning

### Configuration
- **ekf.launch**: Starts both nodes together
- **package.xml**: Dependencies and metadata
- **CMakeLists.txt**: Build instructions

### Documentation
- **QUICKSTART.md**: 5-minute setup
- **README.md**: Complete documentation
- **ROS_BEGINNER_GUIDE.md**: ROS concepts
- **SUMMARY.md**: This file!

---

## 🎓 Learning Outcomes

After using this package, you'll understand:
- ✅ How EKF works for sensor fusion
- ✅ ROS subscribers and publishers
- ✅ Message types (PoseStamped, Odometry)
- ✅ Process and measurement models
- ✅ Tuning Kalman filters
- ✅ Real-time plotting in ROS
- ✅ Package structure and launch files

---

## 🔄 Next Steps

### Short Term (This Week)
1. ✅ Get package running
2. ✅ Observe plots with default parameters
3. ✅ Experiment with Q and R tuning
4. ✅ Record filtered output

### Medium Term (This Month)
1. Add IMU data to process model
2. Include orientation in state vector
3. Add multiple ArUco marker support
4. Implement outlier rejection

### Long Term (This Semester)
1. Integrate into drone's main software
2. Test on live flights
3. Compare with other fusion methods
4. Document results in your report

---

## 📞 Support Resources

### Within Package
- Code comments (extensively documented)
- ROS_BEGINNER_GUIDE.md (concepts)
- README.md (detailed how-to)

### External
- ROS Tutorials: http://wiki.ros.org/ROS/Tutorials
- Kalman Filter: https://www.kalmanfilter.net/
- PX4 MAVROS: https://docs.px4.io/main/en/ros/mavros_installation.html

---

## ✅ Checklist

Before you start:
- [ ] Downloaded/extracted the package
- [ ] Have ROS Noetic installed
- [ ] Have rosbags with required topics
- [ ] Read QUICKSTART.md

Installation:
- [ ] Copied to workspace
- [ ] Installed dependencies (matplotlib, numpy)
- [ ] Built with catkin_make
- [ ] Sourced workspace

First Run:
- [ ] Played rosbag
- [ ] Started EKF node
- [ ] Saw plots appear
- [ ] Verified topics publishing

Understanding:
- [ ] Read code comments in ekf_node.py
- [ ] Understand Q and R matrices
- [ ] Know how to tune parameters
- [ ] Can interpret plots

---

## 🎉 You're Ready!

This package gives you everything you need to:
1. Implement sensor fusion for your drone
2. Learn ROS fundamentals
3. Tune and validate your EKF
4. Integrate into your project

**Start with QUICKSTART.md and good luck! 🚀**

---

*Package created for drone sensor fusion project*
*Extended Kalman Filter with odometry + ArUco markers*
*ROS 1 Noetic | Python 3 | Beginner-friendly*
