// Copyright 2022 Andrew Symington
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef LIBSURVIVE_ROS2__COMPONENT_HPP_
#define LIBSURVIVE_ROS2__COMPONENT_HPP_

#define SURVIVE_ENABLE_FULL_API

// C system
#include <os_generic.h>

// C++ system
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Other
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "libsurvive/survive_api.h"
#include "libsurvive/survive.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

namespace libsurvive_ros2
{

class Component : public rclcpp::Node
{
public:
  explicit Component(const rclcpp::NodeOptions & options);
  virtual ~Component();
  rclcpp::Time get_ros_time(const std::string & str, FLT timecode);
  void publish_imu(const sensor_msgs::msg::Imu & msg);
  const std::string & imu_frame() const {return imu_frame_;}
  // Whether `serial` is the car's tracker. claimTrackerFrame() also latches the
  // first matching object as the car tracker when it is not yet decided (auto
  // mode); isActiveTracker() is read-only. Both are thread-safe: pose updates run
  // on the worker thread, IMU on a libsurvive thread.
  bool claimTrackerFrame(const std::string & serial);
  bool isActiveTracker(const std::string & serial);

private:
  void work();
  SurviveSimpleContext * actx_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::KeyValue>::SharedPtr cfg_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr lighthouse_poses_publisher_;
  std::thread worker_thread_;
  rclcpp::Time last_base_station_update_;
  // Lighthouse serials whose static TF has already been latched. Lighthouses do
  // not move, so each frame is broadcast on /tf_static exactly once; re-sending
  // libsurvive's continuously-refined estimate would make the frames wobble.
  std::set<std::string> published_lighthouse_frames_;
  // Fixed offset mapping libsurvive's monotonic clock onto ROS time, captured
  // once from the first sample so every stamp shares the ROS clock. Guarded by a
  // mutex because get_ros_time() is called from both the worker and IMU threads.
  std::mutex time_ref_mutex_;
  bool time_ref_initialized_ = false;
  rclcpp::Duration time_offset_{0, 0};
  std::string tracking_frame_;
  std::string world_frame_;
  std::string tracker_frame_;
  std::string imu_frame_;
  // Which tracked object is the car. `tracker_serial_` is the configured serial
  // ("" = use the first available tracker, i.e. the first OBJECT to report a
  // pose); `active_tracker_serial_` is the latched winner that owns the canonical
  // tracker_link/imu_link frames.
  // Other objects (controllers, extra trackers) keep their own serial frames so
  // they never collide on tracker_link. Guarded by the mutex (worker + IMU thread).
  std::string tracker_serial_;
  std::string active_tracker_serial_;
  std::mutex tracker_serial_mutex_;
  double lighthouse_rate_;
};

}  // namespace libsurvive_ros2

#endif  // LIBSURVIVE_ROS2__COMPONENT_HPP_
