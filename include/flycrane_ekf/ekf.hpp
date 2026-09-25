#pragma once
// Extended Kalman filter for the CAMLS payload + drones, built on the model in
// dynamics.hpp (see there for the state layout and conventions).
//
//   prediction   RK4 on x' = f(x, u) with the drones' IMU readings as u,
//                P <- Phi P Phi^T + (B Sigma_imu B^T + Q) h   per sub-step
//   updates      UWB ranges, body-frame cable directions, anchor-drone MoCap
//
// Usage -- one call per timeline event, in time order:
//
//   UwbImuEkf ekf(params);
//   ekf.initialize(t0_ns, payload_pose, drone_poses);
//   for each event:
//     imu   -> ekf.set_imu(drone, sample);
//     uwb   -> ekf.update_uwb(range);
//     cable -> ekf.update_cable(drone, sample);
//     mocap -> ekf.update_mocap(anchor_pose);
//
// Every call first propagates the filter to the sample's timestamp, so the
// caller only needs predict_to() directly to evaluate the estimate at other
// times (e.g. logging at a fixed rate).

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

class UwbImuEkf {
 public:
  // constructor
  explicit UwbImuEkf(const UwbImuEkfParams& params);

  // Start the filter at t_ns with P = P0 from `initial_sigma`.  Either give the
  // full state, or the payload and drone poses (e.g. ground truth); biases
  // then start at zero.
  void initialize(int64_t t_ns, const Eigen::VectorXd& x0);
  void initialize(int64_t t_ns, const PoseSample& payload, const std::vector<PoseSample>& drones);

  // Methods:
  // Prediction Step: propagate state and covariance to t_ns with the held IMU
  // inputs.  Does nothing if t_ns is not later than the filter time.
  void predict_to(int64_t t_ns);

  // Input: predict to the sample's time, then hold drone i's IMU reading until
  // its next sample (zero-order hold).
  void set_imu(int drone, const ImuSample& imu);

  // Update Steps.  Each returns false if the sensor is disabled in the params
  // or the innovation covariance was not positive definite.
  bool update_uwb(const UwbRange& range);
  bool update_cable(int drone, const CableSample& cable);
  bool update_mocap(const PoseSample& pose);  // pose of the anchor drone

  // Report the estimated states for the plotter
  EkfSnapshot report_current_readings() const;

  const StateLayout& layout() const { return dynamics_.layout(); }
  const UwbEkfDynamics& dynamics() const { return dynamics_; }
  const Eigen::VectorXd& state() const { return x_now_; }
  const Eigen::MatrixXd& covariance() const { return P_; }
  int64_t time_ns() const { return t_ns_; }
  bool initialized() const { return initialized_; }
  // Normalised innovation squared r^T S^-1 r of the last accepted update.
  // Averages to the measurement dimension when the filter is consistent.
  double last_nis() const { return last_nis_; }

 private:
  // Helper Functions
  // Shared EKF update for a residual r = z - h(x) with Jacobian H and noise R.
  bool update(const Eigen::VectorXd& r, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
  void require_initialized(const char* caller) const;
  void check_drone(int drone) const;

  // parameter variables (imported from uwb_imu_ekf.yaml)
  UwbEkfDynamics dynamics_;
  double h_max_;  // longest integration sub-step [s]
  bool use_uwb_;
  bool use_mocap_;
  bool use_cable_;
  bool estimate_bias_;
  int anchor_drone_;
  int n_;  // number of states

  // Noise models
  Eigen::MatrixXd P0_;         // initial covariance
  Eigen::MatrixXd Sigma_imu_;  // 6N x 6N, IMU noise densities^2, mapped through B
  Eigen::MatrixXd Q_;          // n x n, unmodelled payload dynamics + bias random walks
  double R_uwb_;
  Eigen::Matrix3d R_cable_;
  Eigen::Matrix<double, 6, 6> R_mocap_;

  // Private Variables for EKF (states, inputs, matrices, time)
  // States:
  Eigen::VectorXd x_now_;

  // Inputs: latest IMU sample of each drone
  Inputs u_;

  // Matrices for EKF
  Eigen::MatrixXd phi_;  // transition matrix of the last sub-step
  Eigen::MatrixXd P_;

  // Time-related: integer nanoseconds, like the CSV stamps.  Epoch time does
  // not fit in a double exactly, so only differences are converted to seconds.
  int64_t t_ns_ = 0;
  bool initialized_ = false;
  double last_nis_ = 0.0;
};

}  // namespace flycrane
