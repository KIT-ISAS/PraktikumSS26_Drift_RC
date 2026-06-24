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

// C++ system
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// Other
#include "libsurvive_ros2/component.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "visualization_msgs/msg/marker.hpp"


// Scale factor to move from G to m/s^2.
constexpr double SI_GRAVITY = 9.80665;

// Rigid transform that maps a pose from the libsurvive tracking frame into a
// physically meaningful, axis-aligned world frame.
//
// It is derived from the known base-station layout in config.json. The four
// lighthouses sit at the following libsurvive XY positions, and we want them to
// land on the physical world rectangle:
//
//   config        libsurvive XY        world XY
//   lighthouse0   ( 0.00,  3.59)  ->  (-2.70, -2.33)
//   lighthouse1   ( 3.54,  0.49)  ->  (-2.70,  2.33)
//   lighthouse2   (-0.02, -3.56)  ->  ( 2.80,  2.33)
//   lighthouse3   (-3.54, -0.56)  ->  ( 2.80, -2.33)
//
// That correspondence is a pure rotation about the (gravity-aligned) Z axis by
// ~130.9 degrees at unit scale, plus a small XY offset of the origin. We do not
// touch Z, since libsurvive's frame is already gravity-aligned via floor-offset.
constexpr double WORLD_PI = 3.14159265358979323846;
constexpr double WORLD_YAW = 130.9 * WORLD_PI / 180.0;
constexpr double WORLD_OFFSET_X = 0.037;
constexpr double WORLD_OFFSET_Y = -0.005;

// We can only ever load one version of the driver, so we store a pointer to the instance of the
// driver here, so the IMU callback can push data to it.
libsurvive_ros2::Component * _singleton = nullptr;

static void imu_func(
  SurviveObject * so, int mask, const FLT * accelgyromag, uint32_t rawtime, int id)
{
  if (_singleton) {
    survive_default_imu_process(so, mask, accelgyromag, rawtime, id);
    FLT timecode = SurviveSensorActivations_runtime(
      &so->activations, so->activations.last_imu) / FLT(1e6);
    sensor_msgs::msg::Imu imu_msg;
    imu_msg.header.frame_id = _singleton->imu_frame();
    imu_msg.header.stamp = _singleton->get_ros_time("inertial", timecode);
    imu_msg.angular_velocity.x = accelgyromag[3];
    imu_msg.angular_velocity.y = accelgyromag[4];
    imu_msg.angular_velocity.z = accelgyromag[5];
    imu_msg.linear_acceleration.x = accelgyromag[0] * SI_GRAVITY;
    imu_msg.linear_acceleration.y = accelgyromag[1] * SI_GRAVITY;
    imu_msg.linear_acceleration.z = accelgyromag[2] * SI_GRAVITY;
    _singleton->publish_imu(imu_msg);
  }
}

static void ros_from_pose(
  geometry_msgs::msg::Transform * const tx, const SurvivePose & pose)
{
  tx->translation.x = pose.Pos[0];
  tx->translation.y = pose.Pos[1];
  tx->translation.z = pose.Pos[2];
  tx->rotation.w = pose.Rot[0];
  tx->rotation.x = pose.Rot[1];
  tx->rotation.y = pose.Rot[2];
  tx->rotation.z = pose.Rot[3];
}

// Converts a raw libsurvive pose into a ROS pose (still in the tracking frame).
static geometry_msgs::msg::Pose pose_from_survive(const SurvivePose & pose)
{
  geometry_msgs::msg::Pose p;
  p.position.x = pose.Pos[0];
  p.position.y = pose.Pos[1];
  p.position.z = pose.Pos[2];
  p.orientation.w = pose.Rot[0];
  p.orientation.x = pose.Rot[1];
  p.orientation.y = pose.Rot[2];
  p.orientation.z = pose.Rot[3];
  return p;
}

// Maps a pose expressed in the libsurvive tracking frame into the world frame,
// applying the rotation about Z and origin offset described above (see
// WORLD_YAW). Both the position and the orientation are rotated.
static geometry_msgs::msg::Pose to_world_pose(const geometry_msgs::msg::Pose & in)
{
  static const double cy = std::cos(WORLD_YAW);
  static const double sy = std::sin(WORLD_YAW);
  static const double cw = std::cos(WORLD_YAW * 0.5);
  static const double sw = std::sin(WORLD_YAW * 0.5);

  geometry_msgs::msg::Pose out;
  out.position.x = cy * in.position.x - sy * in.position.y + WORLD_OFFSET_X;
  out.position.y = sy * in.position.x + cy * in.position.y + WORLD_OFFSET_Y;
  out.position.z = in.position.z;

  // out_orientation = q_yaw * in_orientation, where q_yaw is a rotation of
  // WORLD_YAW about the Z axis: q_yaw = (cw, 0, 0, sw).
  const auto & q = in.orientation;
  out.orientation.w = cw * q.w - sw * q.z;
  out.orientation.x = cw * q.x - sw * q.y;
  out.orientation.y = cw * q.y + sw * q.x;
  out.orientation.z = cw * q.z + sw * q.w;
  return out;
}

namespace libsurvive_ros2
{

Component::Component(const rclcpp::NodeOptions & options)
: Node("libsurvive_ros2", options),
  actx_(nullptr),
  tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
  tf_static_broadcaster_(std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this))
{
  // Store the instance globally to be used by a C callback.
  _singleton = this;

  // Global parameters
  this->declare_parameter("tracking_frame", "libsurvive_frame");
  this->get_parameter("tracking_frame", tracking_frame_);
  this->declare_parameter("world_frame", "world");
  this->get_parameter("world_frame", world_frame_);
  this->declare_parameter("lighthouse_rate", 4.0);
  this->get_parameter("lighthouse_rate", lighthouse_rate_);

  // Frames published for the tracked car. This node only emits the *raw* tracker
  // pose as world -> tracker_link (and the raw IMU on imu_link); the canonical
  // CoG/ISO-8855 pose, odometry and IMU are derived downstream by
  // car_localization_pkg, keeping this adapter free of mount-specific maths.
  this->declare_parameter("tracker_frame", "tracker_link");
  this->get_parameter("tracker_frame", tracker_frame_);
  this->declare_parameter("imu_frame", "imu_link");
  this->get_parameter("imu_frame", imu_frame_);

  // Setup topic for the lighthouse poses in world coordinates. The base stations
  // are static, so we latch the latest sample for late-joining subscribers.
  std::string lighthouse_poses_topic;
  this->declare_parameter("lighthouse_poses_topic", "/map/lighthouses");
  this->get_parameter("lighthouse_poses_topic", lighthouse_poses_topic);
  lighthouse_poses_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
    lighthouse_poses_topic, rclcpp::QoS(1).transient_local());

  // Setup topic for IMU.
  std::string imu_topic;
  this->declare_parameter("imu_topic", "imu");
  this->get_parameter("imu_topic", imu_topic);
  imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(imu_topic, 10);

  // Setup topic for joystick.
  std::string joy_topic;
  this->declare_parameter("joy_topic", "joy");
  this->get_parameter("joy_topic", joy_topic);
  joy_publisher_ = this->create_publisher<sensor_msgs::msg::Joy>(joy_topic, 10);

  // Setup topic for configuration.
  std::string cfg_topic;
  this->declare_parameter("cfg_topic", "cfg");
  this->get_parameter("cfg_topic", cfg_topic);
  cfg_publisher_ = this->create_publisher<diagnostic_msgs::msg::KeyValue>(cfg_topic, 10);

  // Publish the static TF from world to the libsurvive tracking frame.
  // This encodes the same rotation + offset applied in to_world_pose() so that
  // downstream nodes can look up the transform via TF instead of inline math.
  {
    static const double cw = std::cos(WORLD_YAW * 0.5);
    static const double sw = std::sin(WORLD_YAW * 0.5);
    geometry_msgs::msg::TransformStamped world_to_tracking;
    world_to_tracking.header.stamp = this->get_clock()->now();
    world_to_tracking.header.frame_id = world_frame_;
    world_to_tracking.child_frame_id = tracking_frame_;
    world_to_tracking.transform.translation.x = WORLD_OFFSET_X;
    world_to_tracking.transform.translation.y = WORLD_OFFSET_Y;
    world_to_tracking.transform.translation.z = 0.0;
    world_to_tracking.transform.rotation.w = cw;
    world_to_tracking.transform.rotation.x = 0.0;
    world_to_tracking.transform.rotation.y = 0.0;
    world_to_tracking.transform.rotation.z = sw;
    tf_static_broadcaster_->sendTransform(world_to_tracking);
  }

  // Setup driver parameters.
  std::string driver_args;
  this->declare_parameter("driver_args", "--disable-calibrate 1");
  this->get_parameter("driver_args", driver_args);
  std::vector<std::string> tokens;
  std::stringstream driver_ss(driver_args);
  std::string token;
  while (getline(driver_ss, token, ' ')) {
    tokens.push_back(token);
  }
  std::vector<const char *> args;
  // libsurvive's parser treats argv[0] as the program name and skips it,
  // so prepend a placeholder to keep the first real flag from being dropped.
  args.push_back("holodeck");
  for (const auto & t : tokens) {
    args.push_back(t.c_str());
  }

  // Try and initialize survive with the arguments supplied.
  actx_ = survive_simple_init(args.size(), const_cast<char **>(args.data()));
  if (actx_ == nullptr) {
    RCLCPP_FATAL(this->get_logger(), "Could not initialize the libsurvive context");
    return;
  }

  // Setup callback for reading IMU data.
  SurviveContext * ctx = survive_simple_get_ctx(actx_);
  survive_install_imu_fn(ctx, imu_func);

  // Initialize the survive thread.
  survive_simple_start_thread(actx_);

  int disableCalibrate = survive_configi(ctx, "disable-calibrate", 0, 0);
  RCLCPP_INFO(this->get_logger(), "Libsurvive driver initialized with disable-calibrate=%d",
      disableCalibrate);

  // Start the work thread
  worker_thread_ = std::thread(&Component::work, this);
}

Component::~Component()
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up.");
  worker_thread_.join();

  RCLCPP_INFO(this->get_logger(), "Shutting down libsurvive driver");
  if (actx_) {
    survive_simple_close(actx_);
  }

  RCLCPP_INFO(this->get_logger(), "Clearing singleton instance");
  _singleton = nullptr;
}

rclcpp::Time Component::get_ros_time(const std::string & /*str*/, FLT timecode)
{
  // libsurvive timecodes are seconds on its own monotonic clock (counting from
  // driver start), not the ROS epoch. Anchor them to ROS time once via a fixed
  // offset captured on the first sample, so the dynamic tracker/IMU/lighthouse
  // stamps share the same clock as the static transforms and consumers. Without
  // this the dynamic frames sit ~1.7e9 s in the past and drift relative to the
  // rest of the TF tree. A constant offset preserves the relative timing the
  // downstream velocity finite-difference relies on.
  const rcl_clock_type_t clock_type = this->get_clock()->get_clock_type();
  const rclcpp::Time survive_time =
    rclcpp::Time(0, 0, clock_type) + rclcpp::Duration(std::chrono::duration<double>(timecode));

  std::lock_guard<std::mutex> lock(time_ref_mutex_);
  if (!time_ref_initialized_) {
    time_offset_ = this->get_clock()->now() - survive_time;
    time_ref_initialized_ = true;
  }
  return survive_time + time_offset_;
}

void Component::publish_imu(const sensor_msgs::msg::Imu & msg)
{
  if (imu_publisher_) {
    imu_publisher_->publish(msg);
  }
}

void Component::work()
{
  RCLCPP_INFO(this->get_logger(), "Start listening for events..");

  // Poll for events.
  struct SurviveSimpleEvent event = {};
  while (survive_simple_wait_for_event(
      actx_,
      &event) != SurviveSimpleEventType_Shutdown && rclcpp::ok())
  {
    // Business logic depends on the event type
    switch (event.event_type) {
      // TYPE: Pose update (limit to non-lighthouses only)
      case SurviveSimpleEventType_PoseUpdateEvent: {
          const struct SurviveSimplePoseUpdatedEvent * pose_event =
            survive_simple_get_pose_updated_event(&event);
          if (survive_simple_object_get_type(pose_event->object) !=
            SurviveSimpleObject_LIGHTHOUSE)
          {
            SurvivePose pose = {};
            auto timecode = survive_simple_object_get_latest_pose(pose_event->object, &pose);
            if (timecode > 0) {
              // Raw tracker pose; mount correction is applied downstream.
              geometry_msgs::msg::TransformStamped pose_msg;
              pose_msg.header.stamp = this->get_ros_time("tracker", timecode);
              pose_msg.header.frame_id = tracking_frame_;
              pose_msg.child_frame_id = tracker_frame_;
              ros_from_pose(&pose_msg.transform, pose);
              tf_broadcaster_->sendTransform(pose_msg);
            }
          }
          break;
        }

      // TYPE: Button update
      case SurviveSimpleEventType_ButtonEvent: {
          const struct SurviveSimpleButtonEvent * button_event = survive_simple_get_button_event(
            &event);
          auto obj = button_event->object;
          sensor_msgs::msg::Joy joy_msg;
          joy_msg.header.frame_id = survive_simple_serial_number(button_event->object);
          joy_msg.header.stamp = this->get_ros_time("button", button_event->time);
          joy_msg.axes.resize(SURVIVE_MAX_AXIS_COUNT * 2);
          joy_msg.buttons.resize(SURVIVE_BUTTON_MAX * 2);
          int64_t mask = survive_simple_object_get_button_mask(obj);
          mask |= (survive_simple_object_get_touch_mask(obj) << SURVIVE_BUTTON_MAX);
          for (int i = 0; i < SURVIVE_MAX_AXIS_COUNT * 2; i++) {
            joy_msg.axes[i] =
              static_cast<float>(survive_simple_object_get_input_axis(obj, (enum SurviveAxis)i));
          }
          for (int i = 0; i < mask && i < static_cast<int>(joy_msg.buttons.size()); i++) {
            joy_msg.buttons[i] = (mask >> i) & 1;
          }
          joy_publisher_->publish(joy_msg);
          break;
        }

      // TYPE: Configuration update
      case SurviveSimpleEventType_ConfigEvent: {
          const struct SurviveSimpleConfigEvent * config_event = survive_simple_get_config_event(
            &event);
          diagnostic_msgs::msg::KeyValue cfg_msg;
          cfg_msg.key = survive_simple_serial_number(config_event->object);
          cfg_msg.value = config_event->cfg;
          cfg_publisher_->publish(cfg_msg);
          break;
        }

      // TYPE: Device add event
      case SurviveSimpleEventType_DeviceAdded: {
          const struct SurviveSimpleObjectEvent * object_event = survive_simple_get_object_event(
            &event);
          RCLCPP_INFO(
            this->get_logger(), "A new device %s was added at time %lf",
            survive_simple_serial_number(object_event->object),
            this->get_ros_time("connect", object_event->time).seconds()
          );
          break;
        }

      // TYPE: no-op
      case SurviveSimpleEventType_None: {
          break;
        }

      // We should never get here.
      default:
        RCLCPP_WARN(this->get_logger(), "Unknown event");
        break;
    }

    // Always update the base stations
    auto time_now = this->get_clock()->now();
    if (time_now.seconds() - last_base_station_update_.seconds() > lighthouse_rate_) {
      last_base_station_update_ = time_now;
      geometry_msgs::msg::PoseArray lighthouse_msg;
      lighthouse_msg.header.stamp = time_now;
      lighthouse_msg.header.frame_id = world_frame_;
      for (const SurviveSimpleObject * it = survive_simple_get_first_object(actx_); it != 0;
        it = survive_simple_get_next_object(actx_, it))
      {
        if (survive_simple_object_get_type(it) == SurviveSimpleObject_LIGHTHOUSE) {
          SurvivePose pose = {};
          auto timecode = survive_simple_object_get_latest_pose(it, &pose);
          if (timecode > 0) {
            // Lighthouses are static: latch each frame on /tf_static exactly once.
            // Re-broadcasting libsurvive's refined estimate would make the frame
            // wobble. The live estimate still streams on the PoseArray topic below.
            std::string serial = survive_simple_serial_number(it);
            if (published_lighthouse_frames_.insert(serial).second) {
              geometry_msgs::msg::TransformStamped pose_msg;
              pose_msg.header.stamp = this->get_ros_time("lighthouse", timecode);
              pose_msg.header.frame_id = tracking_frame_;
              pose_msg.child_frame_id = serial;
              ros_from_pose(&pose_msg.transform, pose);
              tf_static_broadcaster_->sendTransform(pose_msg);
            }

            // Collect the lighthouse pose in world coordinates.
            lighthouse_msg.poses.push_back(to_world_pose(pose_from_survive(pose)));
          }
        }
      }
      if (lighthouse_poses_publisher_ && !lighthouse_msg.poses.empty()) {
        lighthouse_poses_publisher_->publish(lighthouse_msg);
      }
    }
  }
}

}  // namespace libsurvive_ros2

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(libsurvive_ros2::Component)
