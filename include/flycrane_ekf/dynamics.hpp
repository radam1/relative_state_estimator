#pragma once
// Continuous-time model of the CAMLS system -- a cable-suspended payload
// carried by N drones -- together with its analytic Jacobians and the
// measurement models.  Equation numbers refer to
// CAMLS_Range_Based_Observability_Analysis.pdf.
//
// Differences from the PDF
// ------------------------
//   * Each drone's IMU is an INPUT, as in agilib's EkfImu and ekf_cable_load:
//         v_i'     = g + R_i (f_i - b_a,i)
//         Theta_i' = T(Theta_i)^-1 (w_i - b_g,i)
//     There is no motor model, the drone body rate is not a state, and the
//     accelerometer and gyro are not measurements.
//   * Eq. (11) sign: the cable pulls the payload with -tau_i q_i, so the torque
//     is  -sum_i tau_i rho_i x (R_L^T q_i).  Eq. (36) already uses this sign.
//
// Conventions
// -----------
//   * World frame z up, gravity (0, 0, -g).
//   * Theta = [roll, pitch, yaw] and R = Rz(yaw) Ry(pitch) Rx(roll).  The body
//     rate is w = T(Theta) Theta', with T from eq. (16).  T is singular at
//     pitch = +-90 deg.
//   * q_i = e_i / l_i points from the drone hook h_i to the payload attach
//     point c_i, i.e. downwards for a hanging payload.
//
// State layout (indices from StateLayout)
//   payload   [p_L  v_L  Theta_L  w_L]                 12
//   drone i   [p_i  v_i  Theta_i  (b_g,i  b_a,i)]       9, or 15 with IMU biases
// Input layout: drone i owns input columns 6i..6i+2 (specific force f_i) and
// 6i+3..6i+5 (body rate w_i).

#include <vector>

#include <Eigen/Core>

#include "flycrane_ekf/measurements.hpp"
#include "flycrane_ekf/params.hpp"

namespace flycrane {

using Vector6d = Eigen::Matrix<double, 6, 1>;

// One IMU sample per drone (t_ns is ignored).
using Inputs = std::vector<ImuSample>;

struct StateLayout {
  int n_drones = 0;
  bool with_bias = false;

  static constexpr int kPayloadPos = 0;
  static constexpr int kPayloadVel = 3;
  static constexpr int kPayloadAtt = 6;
  static constexpr int kPayloadRate = 9;
  static constexpr int kPayloadSize = 12;

  int drone_size() const { return with_bias ? 15 : 9; }
  int size() const { return kPayloadSize + n_drones * drone_size(); }

  int drone_pos(int i) const { return kPayloadSize + i * drone_size(); }
  int drone_vel(int i) const { return drone_pos(i) + 3; }
  int drone_att(int i) const { return drone_pos(i) + 6; }
  int gyro_bias(int i) const { return drone_pos(i) + 9; }   // only valid with_bias
  int acc_bias(int i) const { return drone_pos(i) + 12; }   // only valid with_bias

  int input_size() const { return 6 * n_drones; }
  static int input_acc(int i) { return 6 * i; }
  static int input_gyr(int i) { return 6 * i + 3; }
};

// Cable i: geometry, tension and -- if requested -- their Jacobians w.r.t. the
// state (eqs. 28, 30-32).  Bias columns of the Jacobians are left zero: the
// biases enter only through (imu - bias), see UwbEkfDynamics::jacobians().
struct Cable {
  Eigen::Vector3d e;       // c_i - h_i, world frame
  Eigen::Vector3d e_dot;
  Eigen::Vector3d q;       // e / l
  double length = 0.0;     // l
  double tension = 0.0;    // tau, 0 if slack
  bool slack = false;      // true only with allow_slack and a compressed cable

  Eigen::MatrixXd De, De_dot, Dq;  // 3 x n
  Eigen::RowVectorXd Dtau;         // 1 x n
  Eigen::Matrix3d De_dot_gyr;      // d e_dot / d gyr_i
  Eigen::RowVector3d Dtau_gyr;     // d tau / d gyr_i
};

class UwbEkfDynamics {
 public:
  explicit UwbEkfDynamics(const UwbImuEkfParams& params);

  const StateLayout& layout() const { return layout_; }

  // ---- process model
  // Solve the dynamics model for the current state: x' = f(x, u).
  Eigen::VectorXd state_derivative(const Eigen::VectorXd& x, const Inputs& u) const;

  // Solve the Jacobians for the current state: A = df/dx (n x n) and
  // B = df/du (n x 6N).
  void jacobians(const Eigen::VectorXd& x, const Inputs& u, Eigen::MatrixXd& A, Eigen::MatrixXd& B) const;

  // One classic Runge-Kutta 4 step of length h with u held constant.
  Eigen::VectorXd rk4_step(const Eigen::VectorXd& x, const Inputs& u, double h) const;

  // Cable i.  `gyr` is drone i's raw gyro reading; it only affects e_dot and
  // the tension (through the damper).
  Cable compute_cable(const Eigen::VectorXd& x, int i, const Eigen::Vector3d& gyr, bool with_jacobians) const;

  // ---- measurement models.  If H is given it is set to dh/dx.
  // Range between the UWB antennas of two drones, eqs. (41)-(42).
  double predict_uwb(const Eigen::VectorXd& x, int host, int peer, Eigen::RowVectorXd* H = nullptr) const;

  // Unit cable direction in drone i's body frame, s_i = R_i^T q_i, eq. (33).
  Eigen::Vector3d predict_cable_dir(const Eigen::VectorXd& x, int i, Eigen::MatrixXd* H = nullptr) const;

  // MoCap residual for drone i, r = [p_meas - p_i ; Log(R_i^T R_meas)], so the
  // filter applies x += K r directly.  H = [I at p_i ; T(Theta_i) at Theta_i]
  // (eq. 40) is exact for small attitude residuals.
  Vector6d mocap_residual(const Eigen::VectorXd& x, int i, const Eigen::Vector3d& p_meas,
                          const Eigen::Matrix3d& R_meas, Eigen::MatrixXd* H = nullptr) const;

 private:
  void check_sizes(const Eigen::VectorXd& x, const Inputs& u) const;
  void check_drone(int i) const;
  // IMU readings corrected by the bias states (no-op when biases are off).
  Eigen::Vector3d body_rate(const Eigen::VectorXd& x, int i, const Eigen::Vector3d& gyr) const;
  Eigen::Vector3d specific_force(const Eigen::VectorXd& x, int i, const Eigen::Vector3d& acc) const;

  StateLayout layout_;
  PhysicsParams physics_;
  Eigen::Vector3d gravity_;         // world frame, (0, 0, -g)
  double payload_mass_ = 0.0;
  Eigen::Matrix3d payload_inertia_;
  Eigen::Matrix3d payload_inertia_inv_;
  std::vector<DroneParams> drones_;
};

// =============================================================================
// Rotation helpers
// =============================================================================
Eigen::Matrix3d skew(const Eigen::Vector3d& v);             // skew(a) b = a x b
Eigen::Matrix3d euler_to_R(const Eigen::Vector3d& theta);
Eigen::Vector3d R_to_euler(const Eigen::Matrix3d& R);
Eigen::Matrix3d T_matrix(const Eigen::Vector3d& theta);      // eq. (16)
Eigen::Matrix3d T_inverse(const Eigen::Vector3d& theta);
Eigen::Matrix3d dT_droll(const Eigen::Vector3d& theta);
Eigen::Matrix3d dT_dpitch(const Eigen::Vector3d& theta);     // (dT/dyaw = 0)
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);           // rotation vector
double wrap_angle(double a);                                 // to [-pi, pi]
void wrap_angles(const StateLayout& layout, Eigen::VectorXd& x);

}  // namespace flycrane
