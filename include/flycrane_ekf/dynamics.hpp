#pragma once
// Continuous-time model of the CAMLS system -- a cable-suspended payload
// carried by N drones -- together with its analytic Jacobians and the
// measurement models.  Equation numbers refer to
// CAMLS_Range_Based_Observability_Analysis.pdf (see for more context).
// There are some differences between that obsevability analysis and ours, though. 

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
  bool with_gyro_bias = false;
  bool with_acc_bias = false;

  static constexpr int kPayloadPos = 0;
  static constexpr int kPayloadVel = 3;
  static constexpr int kPayloadAtt = 6;
  static constexpr int kPayloadRate = 9;
  static constexpr int kPayloadSize = 12;

  int drone_size() const { return 9 + (with_gyro_bias ? 3 : 0) + (with_acc_bias ? 3 : 0); }
  int size() const { return kPayloadSize + n_drones * drone_size(); }

  int drone_pos(int i) const { return kPayloadSize + i * drone_size(); }
  int drone_vel(int i) const { return drone_pos(i) + 3; }
  int drone_att(int i) const { return drone_pos(i) + 6; }
  int gyro_bias(int i) const { return drone_pos(i) + 9; }                          // only valid with_gyro_bias
  int acc_bias(int i) const { return drone_pos(i) + (with_gyro_bias ? 12 : 9); }  // only valid with_acc_bias

  int input_size() const { return 6 * n_drones; }
  static int input_acc(int i) { return 6 * i; }
  static int input_gyr(int i) { return 6 * i + 3; }
};

// Cable i: geometry, tension and their Jacobians w.r.t. the state
// Bias columns of the Jacobians are left zero(biases enter only through imu-bias).
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

  // One Runge-Kutta 4 step of length h with u held constant
  Eigen::VectorXd rk4_step(const Eigen::VectorXd& x, const Inputs& u, double h) const;

  // Cable i. `gyr` is drone i's raw gyro reading
  Cable compute_cable(const Eigen::VectorXd& x, int i, const Eigen::Vector3d& gyr, bool with_jacobians) const;

  // Range between the UWB antennas of two drones, eqs. (41)-(42).
  double predict_uwb(const Eigen::VectorXd& x, int host, int peer, Eigen::RowVectorXd* H = nullptr) const;

  // Unit cable direction in drone i's body frame, s_i = R_i^T q_i, eq. (33).
  Eigen::Vector3d predict_cable_dir(const Eigen::VectorXd& x, int i, Eigen::MatrixXd* H = nullptr) const;

  // MoCap residual for drone i
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

// Rotation helpers
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
