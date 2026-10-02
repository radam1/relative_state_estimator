#pragma once
// Parameter loading from the params YAML (e.g. params/uwb_imu_ekf.yaml).
// Every key is required; a missing or invalid one throws std::runtime_error
// naming the file and the full key path, e.g. "payload.inertia: must be > 0".

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace flycrane {

struct PhysicsParams {
  double gravity = 9.81;         // [m/s^2], acts along world -z
  double cable_stiffness = 0.0;  // k [N/m]
  double cable_damping = 0.0;    // c [N s/m]
  bool allow_slack = false;      // true: tension clamped at 0 when a cable is compressed
};

struct PayloadParams {
  double mass = 0.0;                                  // [kg]
  Eigen::Vector3d inertia = Eigen::Vector3d::Zero();  // principal moments [kg m^2]
};

// One drone and its cable.
struct DroneParams {
  std::string name;                                                // namespace, e.g. "falcon6"
  double mass = 0.0;                                               // [kg]
  Eigen::Vector3d inertia = Eigen::Vector3d::Zero();               // [kg m^2]
  double cable_length = 0.0;                                       // rest length [m]
  Eigen::Vector3d attach_point_payload = Eigen::Vector3d::Zero();  // rho_i, payload frame
  Eigen::Vector3d hook_offset = Eigen::Vector3d::Zero();           // d_i, drone body frame
  Eigen::Vector3d uwb_antenna = Eigen::Vector3d::Zero();           // u_i^ant, drone body frame
};

struct FilterParams {
  bool estimate_imu_bias = false;
  int anchor_drone = 0;               // drone index with MoCap position + attitude
  double max_integration_step = 1e-3; // [s]
  bool use_mocap = true;
  bool use_uwb = true;
  bool use_cable = true;
  double mocap_rate_hz = 50.0;
  double log_rate_hz = 50.0;
  double init_pose_offset = 0.0; 
  double init_ang_offset = 0.0; 
};

// 1-sigma of the initial covariance.
struct InitialSigma {
  double payload_pos = 0.0, payload_vel = 0.0, payload_att = 0.0, payload_rate = 0.0;
  double drone_pos = 0.0, drone_vel = 0.0, drone_att = 0.0;
  double gyro_bias = 0.0, acc_bias = 0.0;
};

// Continuous-time noise densities [unit/sqrt(Hz)].
struct ProcessNoise {
  double acc = 0.0, gyro = 0.0;
  double acc_bias_walk = 0.0, gyro_bias_walk = 0.0;
  double payload_force = 0.0, payload_torque = 0.0;
};

// 1-sigma measurement noise.
struct MeasurementNoise {
  double uwb = 0.0;        // [m]
  double cable_dir = 0.0;  // [rad]
  double mocap_pos = 0.0;  // [m]
  double mocap_att = 0.0;  // [rad]
};

struct PlotParams {
  bool enabled = true;
  bool show = false;
  std::string output_folder;  // folder the plots are saved to
};

struct DiagnosticsParams {
  double pose_convergence_tol = 0.1; // [m] At what level of accuracy the filter is said to have "Converged" for a given drone
  double ang_convergence_tol = 5;    // [deg] 
};

struct UwbImuEkfParams {
  std::vector<std::string> drone_names;  // index order, same as `drones`
  PhysicsParams physics;
  PayloadParams payload;
  std::vector<DroneParams> drones;       // drones[i].name == drone_names[i]
  FilterParams filter;
  InitialSigma initial_sigma;
  ProcessNoise process_noise;
  MeasurementNoise measurement_noise;
  PlotParams plot;
  DiagnosticsParams diagnostics; 
};

// Load and validate the whole params file.
UwbImuEkfParams load_params(const std::filesystem::path& yaml_file);

// Ordered drone namespaces from the top-level `drones:` list.  The order fixes
// each drone's integer index, so it must match the --drones order that
// bag_converter.py was run with (UWB host/peer ids refer to it).
// Throws std::runtime_error if the file or key is missing or malformed.
std::vector<std::string> load_drone_names(const std::filesystem::path& yaml_file);

}  // namespace flycrane
