#include "flycrane_ekf/ekf.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace flycrane {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr int kPos = StateLayout::kPayloadPos;
constexpr int kVel = StateLayout::kPayloadVel;
constexpr int kAtt = StateLayout::kPayloadAtt;
constexpr int kRate = StateLayout::kPayloadRate;

double sq(double v) { return v * v; }

}  // namespace

// Constructor 
UwbImuEkf::UwbImuEkf(const UwbImuEkfParams& params)
    : dynamics_(params),
      h_max_(params.filter.max_integration_step),
      use_uwb_(params.filter.use_uwb),
      use_mocap_(params.filter.use_mocap),
      use_cable_(params.filter.use_cable),
      anchor_drone_(params.filter.anchor_drone),
      n_(dynamics_.layout().size()) {
  const StateLayout& L = dynamics_.layout();

  // Initial covariance: P0 = diag(initial_sigma^2)
  const InitialSigma& s0 = params.initial_sigma;
  VectorXd p0(n_);
  p0.segment<3>(kPos).setConstant(sq(s0.payload_pos));
  p0.segment<3>(kVel).setConstant(sq(s0.payload_vel));
  p0.segment<3>(kAtt).setConstant(sq(s0.payload_att));
  p0.segment<3>(kRate).setConstant(sq(s0.payload_rate));
  for (int i = 0; i < L.n_drones; ++i) {
    p0.segment<3>(L.drone_pos(i)).setConstant(sq(s0.drone_pos));
    p0.segment<3>(L.drone_vel(i)).setConstant(sq(s0.drone_vel));
    p0.segment<3>(L.drone_att(i)).setConstant(sq(s0.drone_att));
    if (L.with_gyro_bias) p0.segment<3>(L.gyro_bias(i)).setConstant(sq(s0.gyro_bias));
    if (L.with_acc_bias) p0.segment<3>(L.acc_bias(i)).setConstant(sq(s0.acc_bias));
  }
  P0_ = p0.asDiagonal();

  // Process noise (Modeled as continuous-time spectral densities) 
  const ProcessNoise& q = params.process_noise;
  VectorXd sigma_u(L.input_size());
  for (int i = 0; i < L.n_drones; ++i) {
    sigma_u.segment<3>(StateLayout::input_acc(i)).setConstant(sq(q.acc));
    sigma_u.segment<3>(StateLayout::input_gyr(i)).setConstant(sq(q.gyro));
  }
  Sigma_imu_ = sigma_u.asDiagonal();

  // Noise on the model itself
  VectorXd q_model = VectorXd::Zero(n_);
  q_model.segment<3>(kVel).setConstant(sq(q.payload_force));
  q_model.segment<3>(kRate).setConstant(sq(q.payload_torque));
  for (int i = 0; i < L.n_drones; ++i) {
    if (L.with_gyro_bias) q_model.segment<3>(L.gyro_bias(i)).setConstant(sq(q.gyro_bias_walk));
    if (L.with_acc_bias) q_model.segment<3>(L.acc_bias(i)).setConstant(sq(q.acc_bias_walk));
  }
  Q_ = q_model.asDiagonal();

  // Measurement noise
  const MeasurementNoise& r = params.measurement_noise;
  R_uwb_ = sq(r.uwb);
  cable_tension_scaling_ = params.filter.cable_tension_scaling;
  cable_dir_sigma_ = r.cable_dir;
  cable_force_sigma_ = r.cable_force;
  cable_min_tension_ = r.cable_min_tension;
  R_mocap_.setZero();
  R_mocap_.topLeftCorner<3, 3>() = sq(r.mocap_pos) * Matrix3d::Identity();
  R_mocap_.bottomRightCorner<3, 3>() = sq(r.mocap_att) * Matrix3d::Identity();

  // Loading parameters for the NIS gates (if not enabled gate at 0.0 to allow all measurements to pass)
  const GatingParams& g = params.gating;
  gate_uwb_ = g.enabled ? g.uwb : 0.0;
  gate_cable_ = g.enabled ? g.cable : 0.0;
  gate_mocap_ = g.enabled ? g.mocap : 0.0;
  max_rejects_ = g.max_consecutive_rejects;
  warmup_ns_ = static_cast<int64_t>(std::llround(1e9 * g.warmup_s));
  cable_rejects_.assign(L.n_drones, 0);

  // Initialize the inputs: until a drone's first IMU sample arrives, assume it hovers.
  u_.resize(L.n_drones);
  for (ImuSample& s : u_) {
    s.t_ns = 0;
    s.acc = Vector3d(0.0, 0.0, params.physics.gravity);
    s.gyr = Vector3d::Zero();
  }

  x_now_ = VectorXd::Zero(n_);
  P_ = P0_;
  phi_ = MatrixXd::Identity(n_, n_);
}

void UwbImuEkf::initialize(int64_t t_ns, const VectorXd& x0) {
  if (x0.size() != n_) {
    throw std::invalid_argument("UwbImuEkf::initialize: state has " + std::to_string(x0.size()) +
                                " entries, expected " + std::to_string(n_));
  }
  x_now_ = x0;
  wrap_angles(dynamics_.layout(), x_now_);
  P_ = P0_;
  t_ns_ = t_ns;
  gate_start_ns_ = t_ns + warmup_ns_;
  uwb_rejects_ = mocap_rejects_ = 0;
  std::fill(cable_rejects_.begin(), cable_rejects_.end(), 0);
  initialized_ = true;
}

void UwbImuEkf::initialize(int64_t t_ns, const PoseSample& payload, const std::vector<PoseSample>& drones) {
  const StateLayout& L = dynamics_.layout();
  if (static_cast<int>(drones.size()) != L.n_drones) {
    throw std::invalid_argument("UwbImuEkf::initialize: got " + std::to_string(drones.size()) + " drone poses for " +
                                std::to_string(L.n_drones) + " drones");
  }
  VectorXd x0 = VectorXd::Zero(n_);
  x0.segment<3>(kPos) = payload.p;
  x0.segment<3>(kVel) = payload.v;
  x0.segment<3>(kAtt) = R_to_euler(payload.q.toRotationMatrix());
  x0.segment<3>(kRate) = payload.w;
  for (int i = 0; i < L.n_drones; ++i) {
    x0.segment<3>(L.drone_pos(i)) = drones[i].p;
    x0.segment<3>(L.drone_vel(i)) = drones[i].v;
    x0.segment<3>(L.drone_att(i)) = R_to_euler(drones[i].q.toRotationMatrix());
  }
  initialize(t_ns, x0);
}

void UwbImuEkf::require_initialized(const char* caller) const {
  if (!initialized_) throw std::logic_error(std::string("UwbImuEkf::") + caller + ": call initialize() first");
}

void UwbImuEkf::check_drone(int drone) const {
  if (drone < 0 || drone >= dynamics_.layout().n_drones) {
    throw std::out_of_range("UwbImuEkf: drone index " + std::to_string(drone) + " is out of range");
  }
}

// Prediction Step
void UwbImuEkf::predict_to(int64_t t_ns) {
  require_initialized("predict_to");
  if (t_ns <= t_ns_) return;  

  const double dt = 1e-9 * static_cast<double>(t_ns - t_ns_); 
  const int steps = std::max(1, static_cast<int>(std::ceil(dt / h_max_)));
  const double h = dt / steps;

  const MatrixXd I = MatrixXd::Identity(n_, n_);
  MatrixXd A, B;
  // Calculate Runge-Kutta step length
  for (int k = 0; k < steps; ++k) {
    dynamics_.jacobians(x_now_, u_, A, B);  // linearise at the start of the sub-step
    const MatrixXd Qc = B * Sigma_imu_ * B.transpose() + Q_;

    x_now_ = dynamics_.rk4_step(x_now_, u_, h);

    // Second-order transition matrix: I + Ah alone slowly inflates the
    // undamped pendulum modes.
    const MatrixXd Ah = A * h;
    phi_ = I + Ah + 0.5 * Ah * Ah;
    P_ = phi_ * P_ * phi_.transpose() + Qc * h;
    P_ = 0.5 * (P_ + P_.transpose());
  }
  wrap_angles(dynamics_.layout(), x_now_);
  t_ns_ = t_ns;
}

void UwbImuEkf::set_imu(int drone, const ImuSample& imu) {
  check_drone(drone);
  predict_to(imu.t_ns);
  u_[drone] = imu;
}

// Update

// Measurements arrive at different rates and times, so each one is applied on its own, at its own timestamp. 
UpdateStatus UwbImuEkf::update_uwb(const UwbRange& range) {
  if (!use_uwb_) return UpdateStatus::Disabled;
  predict_to(range.t_ns);
  RowVectorXd H;
  const double predicted = dynamics_.predict_uwb(x_now_, range.host, range.peer, &H);
  return update(VectorXd::Constant(1, range.range - predicted), H, MatrixXd::Constant(1, 1, R_uwb_), gate_uwb_,
                uwb_rejects_);
}

// If tension-based cov. scaling is enabled, use sigma^2 = sigma_cable_dir^2 + (cable_force / T)^2
Matrix3d UwbImuEkf::cable_noise(const CableSample& cable) const {
  double var = sq(cable_dir_sigma_);
  if (cable_tension_scaling_ && cable.tension_valid) var += sq(cable_force_sigma_ / cable.tension);
  return var * Matrix3d::Identity();
}

UpdateStatus UwbImuEkf::update_cable(int drone, const CableSample& cable) {
  if (!use_cable_) return UpdateStatus::Disabled;
  check_drone(drone);
  // If cable tension is below threshold then ignore measurement
  if (cable_tension_scaling_ && cable.tension_valid && cable.tension < cable_min_tension_) return UpdateStatus::Skipped;
  predict_to(cable.t_ns);
  MatrixXd H;
  const Vector3d predicted = dynamics_.predict_cable_dir(x_now_, drone, &H);
  return update(cable.s_body - predicted, H, cable_noise(cable), gate_cable_, cable_rejects_[drone]);
}

UpdateStatus UwbImuEkf::update_mocap(const PoseSample& pose) {
  if (!use_mocap_) return UpdateStatus::Disabled;
  predict_to(pose.t_ns);
  MatrixXd H;
  const Vector6d r = dynamics_.mocap_residual(x_now_, anchor_drone_, pose.p, pose.q.toRotationMatrix(), &H);
  return update(r, H, R_mocap_, gate_mocap_, mocap_rejects_);
}

UpdateStatus UwbImuEkf::update(const VectorXd& r, const MatrixXd& H, const MatrixXd& R, double gate, int& rejects) {
  require_initialized("update");

  // innovation covariance
  const MatrixXd PHt = P_ * H.transpose();
  const MatrixXd S = H * PHt + R;
  const Eigen::LLT<MatrixXd> S_llt(S);
  if (S_llt.info() != Eigen::Success) return UpdateStatus::Numerical;  // S not positive definite
  last_nis_ = r.dot(S_llt.solve(r));

  // Gating handling: off during warm-up, and lets one update through after
  // max_rejects_ in a row so a drifted filter cannot lock itself out
  if (gate > 0.0 && last_nis_ > gate && t_ns_ >= gate_start_ns_ && (max_rejects_ <= 0 || rejects < max_rejects_)) {
    ++rejects;
    return UpdateStatus::Gated;
  }
  rejects = 0;

  // Kalman gain(S is symmetric, so K^T = S^-1 (P H^T)^T)
  const MatrixXd K = S_llt.solve(PHt.transpose()).transpose();

  // updated estimate
  x_now_ += K * r;
  wrap_angles(dynamics_.layout(), x_now_);

  // updated covariance, Joseph form: stays symmetric positive semi-definite
  const MatrixXd I_KH = MatrixXd::Identity(n_, n_) - K * H;
  P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
  return UpdateStatus::Applied;
}

// Report the estimator readings for plotting
EkfSnapshot UwbImuEkf::report_current_readings() const {
  require_initialized("report_current_readings");
  EkfSnapshot snapshot;
  snapshot.t_ns = t_ns_;
  snapshot.x = x_now_;
  snapshot.sigma = P_.diagonal().cwiseMax(0.0).cwiseSqrt();
  return snapshot;
}

}  // namespace flycrane
