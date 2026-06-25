# ROS Basics for Beginners 🎓

A practical guide to understanding ROS concepts used in this EKF package.

## What is ROS?

**ROS (Robot Operating System)** is a middleware that helps different programs ("nodes") communicate with each other. Think of it as a messaging system for robots.

---

## 🔑 Key ROS Concepts

### 1. Nodes
**What:** Independent programs that perform specific tasks
**Example:** `ekf_node.py` is a node that runs the EKF

```bash
# See all running nodes
rosnode list

# Get info about a node
rosnode info /drone_ekf
```

### 2. Topics
**What:** Named channels where nodes send/receive messages
**Think of it like:** Radio stations - nodes can broadcast or listen

```bash
# List all topics
rostopic list

# See what's being published
rostopic echo /mavros/local_position/odom

# Check message frequency
rostopic hz /ekf/pose
```

### 3. Messages
**What:** Data structures sent through topics
**Example:** `PoseStamped` contains position + orientation + timestamp

```bash
# See message structure
rosmsg show geometry_msgs/PoseStamped

# View one message
rostopic echo -n 1 /aruco/pose/marker_417
```

### 4. Subscribers
**What:** Code that listens to a topic
**In our code:**
```python
rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
#                 ↑ topic name              ↑ msg type  ↑ function to call
```

### 5. Publishers
**What:** Code that sends messages to a topic
**In our code:**
```python
self.ekf_pub = rospy.Publisher('/ekf/pose', PoseStamped, queue_size=10)
#                               ↑ topic     ↑ msg type   ↑ buffer size
```

### 6. Callbacks
**What:** Functions that run automatically when a message arrives
**Example:**
```python
def odom_callback(self, msg):
    # This runs every time odometry is received
    x = msg.pose.pose.position.x
```

---

## 🎯 How Our EKF Package Works

### Data Flow
```
Rosbag → /mavros/local_position/odom → EKF Node → /ekf/pose → Plotter
         /aruco/pose/marker_* ------↗
```

### Step-by-Step

1. **Rosbag plays** recorded flight data
2. **Topics publish** odometry and ArUco data
3. **EKF node subscribes** to these topics
4. **Callbacks trigger** when messages arrive:
   - `odom_callback` → prediction step
   - `aruco_callback` → correction step
5. **EKF publishes** filtered estimate
6. **Plotter subscribes** and visualizes results

---

## 📦 Working with Packages

### Package Structure
```
drone_ekf/
├── CMakeLists.txt      # Build instructions
├── package.xml         # Package metadata & dependencies
├── scripts/            # Python scripts
│   ├── ekf_node.py
│   └── plotter.py
├── launch/             # Launch files (start multiple nodes)
│   └── ekf.launch
└── README.md
```

### Building
```bash
cd ~/drone_ekf_ws
catkin_make              # Compiles the workspace
source devel/setup.bash  # Makes package available
```

### Running
```bash
# Method 1: Launch file (starts everything)
roslaunch drone_ekf ekf.launch

# Method 2: Individual nodes
rosrun drone_ekf ekf_node.py
rosrun drone_ekf plotter.py
```

---

## 🔍 Useful Commands

### Investigating Topics
```bash
# What topics exist?
rostopic list

# What's the message type?
rostopic info /aruco/pose/marker_417

# What's in the message?
rostopic echo /aruco/pose/marker_417

# How fast are messages coming?
rostopic hz /ekf/pose

# See message structure
rosmsg show geometry_msgs/PoseStamped
```

### Debugging
```bash
# Is ROS running?
roscore  # Start ROS master (usually auto-starts)

# What nodes are running?
rosnode list

# View computation graph
rqt_graph

# See all topics and their connections
rqt_graph
```

### Working with Rosbags
```bash
# Play a bag file
rosbag play flight_data.bag

# Play at half speed
rosbag play -r 0.5 flight_data.bag

# See what topics are in a bag
rosbag info flight_data.bag

# Record new topics
rosbag record /ekf/pose /ekf/odom -O filtered_output.bag
```

---

## 🧩 Understanding Message Types

### PoseStamped (geometry_msgs/PoseStamped)
```
header:
  stamp: time              # When was this measured?
  frame_id: "map"          # What coordinate frame?
pose:
  position: {x, y, z}      # Location in meters
  orientation: {x, y, z, w}  # Rotation (quaternion)
```

### Odometry (nav_msgs/Odometry)
```
header: ...
child_frame_id: "base_link"
pose:
  pose: {position, orientation}
  covariance: [36 values]   # Uncertainty
twist:
  twist: {linear, angular}  # Velocities
  covariance: [36 values]
```

---

## 🎓 Common Patterns

### Subscribe and Process
```python
def __init__(self):
    rospy.Subscriber('/topic_name', MessageType, self.callback)

def callback(self, msg):
    # Process the message
    value = msg.data
```

### Publish Periodically
```python
pub = rospy.Publisher('/output', MessageType, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz

while not rospy.is_shutdown():
    msg = MessageType()
    msg.data = compute_something()
    pub.publish(msg)
    rate.sleep()
```

### Time Handling
```python
# Get current time
now = rospy.Time.now()

# Convert to seconds
secs = msg.header.stamp.to_sec()

# Calculate time difference
dt = current_time - last_time
```

---

## 💡 Tips for Beginners

1. **Always check topics first**
   ```bash
   rostopic list
   rostopic echo /topic_name
   ```

2. **Use tab completion**
   - Type `rostopic ` then press TAB twice
   - Saves typing and prevents errors

3. **Read the messages**
   ```bash
   rosmsg show geometry_msgs/PoseStamped
   ```

4. **Monitor with rqt**
   ```bash
   rqt_plot /ekf/pose/pose/position/x
   ```

5. **Start simple**
   - Test one component at a time
   - Add complexity gradually

6. **Check the logs**
   ```bash
   rosnode info /your_node
   ```

---

## 🚨 Common Errors & Solutions

### "Unable to contact my own server"
**Cause:** ROS master not running
**Fix:** `roscore` in a separate terminal

### "No such file or directory"
**Cause:** Forgot to source workspace
**Fix:** `source ~/drone_ekf_ws/devel/setup.bash`

### "Could not find package"
**Cause:** Package not built
**Fix:** `catkin_make` then source again

### "Message type not found"
**Cause:** Wrong message import
**Fix:** Check `rosmsg show MessageType` for correct path

---

## 📚 Learn More

- **ROS Tutorials:** http://wiki.ros.org/ROS/Tutorials
- **Message Types:** http://wiki.ros.org/common_msgs
- **ROS Answers:** https://answers.ros.org

---

## ✅ Checklist for This Project

- [ ] Built workspace with `catkin_make`
- [ ] Sourced workspace: `source devel/setup.bash`
- [ ] Made scripts executable: `chmod +x scripts/*.py`
- [ ] Can list topics: `rostopic list`
- [ ] Can play rosbag: `rosbag play your_bag.bag`
- [ ] EKF node starts without errors
- [ ] Plotter displays graphs
- [ ] Understand how to tune Q and R matrices

Good luck! 🚀
