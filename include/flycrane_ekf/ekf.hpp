#pragma once
// Extended Kalman filter for the CAMLS payload/drones, built on the model in dynamics.hpp
//
// How to use:
//   UwbImuEkf ekf(params);
//   ekf.initialize(t0_ns, payload_pose, drone_poses);
//   for each event:
//     imu   -> ekf.set_imu(drone, sample);
//     uwb   -> ekf.update_uwb(range);
//     cable -> ekf.update_cable(drone, sample);
//     mocap -> ekf.update_mocap(anchor_pose);

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "flycrane_ekf/dynamics.hpp"
#include "flycrane_ekf/measurements.hpp"
#include "flycrane_ekf/params.hpp"

namespace flycrane {

// Snapshot of the filter for logging and plotting.
struct EkfSnapshot {
  int64_t t_ns = 0;
  Eigen::VectorXd x;      // state estimate, indexed with UwbImuEkf::layout()
  Eigen::VectorXd sigma;  // 1-sigma of each state, sqrt(diag(P))
};

// Outcome of an update call. Helps with debugging rejected measurements
enum class UpdateStatus {
  Applied,
  Disabled,   // sensor disabled in the params
  Gated,      // NIS above the gate threshold
  Skipped,    // cable tension below cable_min_tension
  Numerical,  // innovation covariance not positive definite
};

class UwbImuEkf {
 public:
  // constructor
  explicit UwbImuEkf(const UwbImuEkfParams& params);

  // Initialize filter with either the full state or the payload
  void initialize(int64_t t_ns, const Eigen::VectorXd& x0);
  void initialize(int64_t t_ns, const PoseSample& payload, const std::vector<PoseSample>& drones);

  // Methods:

  // Prediction Step: propagate state and covariance to t_ns
  void predict_to(int64_t t_ns);

  // Update IMU reading(zero-order hold)
  void set_imu(int drone, const ImuSample& imu);

  // Update Steps per-sensor
  UpdateStatus update_uwb(const UwbRange& range);
  UpdateStatus update_cable(int drone, const CableSample& cable);
  UpdateStatus update_mocap(const PoseSample& pose);  // pose of the anchor drone

  // Calculate cable noise.
  Eigen::Matrix3d cable_noise(const CableSample& cable) const;

  // Report the estimated states for the plotting and diagnostics
  EkfSnapshot report_current_readings() const;

  const StateLayout& layout() const { return dynamics_.layout(); }
  const UwbEkfDynamics& dynamics() const { return dynamics_; }
  const Eigen::VectorXd& state() const { return x_now_; }
  const Eigen::MatrixXd& covariance() const { return P_; }
  int64_t time_ns() const { return t_ns_; }
  bool initialized() const { return initialized_; }
  // Get the normalized innovation squared for the previous measurement
  double last_nis() const { return last_nis_; }

 private:
  // Helper Functions
  // Shared EKF update for a residual r with Jacobian H and noise R.
  // `gate` is the chi^2 threshold (<= 0: off), `rejects` the stream's count of
  // consecutive gated updates.
  UpdateStatus update(const Eigen::VectorXd& r, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R, double gate,
                      int& rejects);
  void require_initialized(const char* caller) const;
  void check_drone(int drone) const;

  // parameter variables (imported from uwb_imu_ekf.yaml)
  UwbEkfDynamics dynamics_;
  double h_max_;  // longest integration sub-step [s]
  bool use_uwb_;
  bool use_mocap_;
  bool use_cable_;
  int anchor_drone_;
  int n_;  // number of states

  // Noise models
  Eigen::MatrixXd P0_;         
  Eigen::MatrixXd Sigma_imu_;  
  Eigen::MatrixXd Q_;          
  double R_uwb_;
  bool cable_tension_scaling_;
  double cable_dir_sigma_;
  double cable_force_sigma_;
  double cable_min_tension_;
  Eigen::Matrix<double, 6, 6> R_mocap_;

  // Gating-related setup
  double gate_uwb_;
  double gate_cable_;
  double gate_mocap_;
  int max_rejects_;
  int64_t warmup_ns_;
  int64_t gate_start_ns_ = 0;   
  int uwb_rejects_ = 0;        
  std::vector<int> cable_rejects_;
  int mocap_rejects_ = 0;

  // Private Variables for EKF (states, inputs, matrices, time)
  // States:
  Eigen::VectorXd x_now_;

  // Inputs: latest IMU sample of each drone
  Inputs u_;

  // Matrices for EKF
  Eigen::MatrixXd phi_;  // transition matrix of the last sub-step
  Eigen::MatrixXd P_;

  // Time-related vars. ROS time doesn't fit in a double, so use int nanoseconds.
  int64_t t_ns_ = 0;
  bool initialized_ = false;
  double last_nis_ = 0.0;
};

}  // namespace flycrane
