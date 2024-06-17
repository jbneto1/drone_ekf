T1
# For rosbag replay, set /use_sim_time: true in config/ekf_params.yaml.
roslaunch drone_ekf ekf.launch


T2

roslaunch iris_land aruco_stereo.launch

T3 

roslaunch drone_ekf plotter.launch

T4 

rosrun drone_ekf offline_controller_frame_probe.py \
  _pose_topic:=/ekf/pose \
  _local_pose_topic:=/mavros/local_position/pose \
  _yaw_target_rad:=1.57079632679 \
  _csv_path:=/tmp/offline_controller_frame_probe_v9.csv

T5 

rosbag record -O optionA_v9_outputs.bag \
/clock \
/aruco/pose/marker_363 \
/aruco/pose/marker_417 \
/aruco/pose/marker_682 \
/aruco/debug/marker_quality \
/stereo/debug/timing \
/aruco/debug/timing \
/ekf/pose \
/ekf/odom \
/ekf/dead_reckoning \
/ekf/measurements/aruco/marker_363 \
/ekf/measurements/aruco/marker_417 \
/ekf/measurements/aruco/marker_682 \
/ekf/measurements/laser \
/ekf/debug/innovation \
/ekf/debug/covariance \
/ekf/debug/kalman_gain \
/ekf/debug/timing \
/ekf/sensor_status \
/ekf/diagnostics \
/controller/offline_frame_probe \
/controller/offline_cmd/current_mapping_raw \
/controller/offline_cmd/current_mapping_after_cmd_vel \
/controller/offline_cmd/direct_landpad_world \
/controller/offline_cmd/body_from_landpad_raw \
/controller/offline_cmd/body_from_landpad_after_cmd_vel \
/mavros/local_position/pose \
/mavros/local_position/velocity_body \
/mavros/altitude \
/tf \
/tf_static


T7
cd ~/JOAO_MARROM

rosbag play --clock --pause 19_06_26.bag --topics \
/stereo/left/image_raw \
/stereo/right/image_raw \
/mavros/local_position/velocity_body \
/mavros/local_position/pose \
/mavros/altitude \
/tf \
/tf_static


T8

rosbag play --clock --pause bag_stereo_20_03_26.bag --topics \
/stereo/left/image_raw \
/stereo/right/image_raw \
/mavros/local_position/velocity_body \
/mavros/local_position/pose \
/mavros/altitude \
/tf \
/tf_static
