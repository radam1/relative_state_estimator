#pragma once
// Sensor measurement types consumed by the EKF.
//
// These are deliberately independent of any I/O: the offline CSV reader
// (csv_io.hpp) and the future ROS wrapper both produce exactly these structs,
// so the filter never needs to know where its data came from.
//
// Time is always an integer nanosecond stamp (int64_t).  Epoch nanoseconds
// (~1.7e18) do not fit in a double's 53-bit mantissa, so only differences
// (dt) should ever be converted to floating-point seconds.

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace flycrane {

// sensor_msgs/Imu, drone body frame.  acc is specific force [m/s^2], gyr [rad/s].
struct ImuSample {
  int64_t t_ns;
  Eigen::Vector3d acc;
  Eigen::Vector3d gyr;
};

// nav_msgs/Odometry. 
struct PoseSample {
  int64_t t_ns;
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  Eigen::Vector3d v;
  Eigen::Vector3d w;
};

// agiros_msgs/MotorSpeeds, rotor angular velocities [rad/s].
struct MotorSample {
  int64_t t_ns;
  Eigen::Vector4d w;
};

// Cable state from agiros_pilot/wrench_observed:
struct CableSample {
  int64_t t_ns;
  Eigen::Vector3d s_body;
  double tension;
  bool tension_valid;
};

// One peer-to-peer UWB range
struct UwbRange {
  int64_t t_ns;
  uint8_t host;
  uint8_t peer;
  double range;
};

// Sensor kind (order here is priority order for same-timestamped messages in the filter)
enum class Sensor : uint8_t { Imu = 0, Motors = 1, Cable = 2, Uwb = 3, Mocap = 4 };

inline const char* to_string(Sensor s) {
  switch (s) {
    case Sensor::Imu:    return "imu";
    case Sensor::Motors: return "motors";
    case Sensor::Cable:  return "cable";
    case Sensor::Uwb:    return "uwb";
    case Sensor::Mocap:  return "mocap";
  }
  return "?";
}

}  // namespace flycrane
