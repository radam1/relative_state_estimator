#include "flycrane_ekf/dynamics.hpp"

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

// d/dTheta [T(Theta)^-1 w] for a fixed w.  Column k is
// -T^-1 (dT/dTheta_k) T^-1 w, eqs. (35) and (38).  T does not depend on yaw,
// so the last column is zero.
Matrix3d d_Tinv_w(const Vector3d& theta, const Vector3d& w) {
  const Matrix3d Ti = T_inverse(theta);
  const Vector3d euler_rate = Ti * w;
  Matrix3d D = Matrix3d::Zero();
  D.col(0) = -Ti * dT_droll(theta) * euler_rate;
  D.col(1) = -Ti * dT_dpitch(theta) * euler_rate;
  return D;
}

}  // namespace

// =============================================================================
// Setup
// =============================================================================
UwbEkfDynamics::UwbEkfDynamics(const UwbImuEkfParams& params)
    : physics_(params.physics),
      gravity_(0.0, 0.0, -params.physics.gravity),
      payload_mass_(params.payload.mass),
      payload_inertia_(params.payload.inertia.asDiagonal()),
      payload_inertia_inv_(params.payload.inertia.cwiseInverse().asDiagonal()),
      drones_(params.drones) {
  layout_.n_drones = static_cast<int>(drones_.size());
  layout_.with_bias = params.filter.estimate_imu_bias;
}

void UwbEkfDynamics::check_sizes(const VectorXd& x, const Inputs& u) const {
  if (x.size() != layout_.size()) {
    throw std::invalid_argument("state has " + std::to_string(x.size()) + " entries, layout expects " +
                                std::to_string(layout_.size()));
  }
  if (static_cast<int>(u.size()) != layout_.n_drones) {
    throw std::invalid_argument("got " + std::to_string(u.size()) + " IMU inputs for " +
                                std::to_string(layout_.n_drones) + " drones");
  }
}

void UwbEkfDynamics::check_drone(int i) const {
  if (i < 0 || i >= layout_.n_drones) {
    throw std::out_of_range("drone index " + std::to_string(i) + " is not in [0, " +
                            std::to_string(layout_.n_drones) + ")");
  }
}

Vector3d UwbEkfDynamics::body_rate(const VectorXd& x, int i, const Vector3d& gyr) const {
  if (!layout_.with_bias) return gyr;
  return gyr - x.segment<3>(layout_.gyro_bias(i));
}

Vector3d UwbEkfDynamics::specific_force(const VectorXd& x, int i, const Vector3d& acc) const {
  if (!layout_.with_bias) return acc;
  return acc - x.segment<3>(layout_.acc_bias(i));
}

// =============================================================================
// Cable
// =============================================================================
Cable UwbEkfDynamics::compute_cable(const VectorXd& x, int i, const Vector3d& gyr, bool with_jacobians) const {
  check_drone(i);
  const StateLayout& L = layout_;
  const Vector3d& rho = drones_[i].attach_point_payload;
  const Vector3d& hook = drones_[i].hook_offset;

  const Vector3d vL = x.segment<3>(kVel);
  const Vector3d thL = x.segment<3>(kAtt);
  const Vector3d wL = x.segment<3>(kRate);
  const Vector3d vi = x.segment<3>(L.drone_vel(i));
  const Vector3d thi = x.segment<3>(L.drone_att(i));
  const Vector3d wi = body_rate(x, i, gyr);
  const Matrix3d RL = euler_to_R(thL);
  const Matrix3d Ri = euler_to_R(thi);

  // eqs. (4)-(7) and (17)
  Cable c;
  c.e = x.segment<3>(kPos) + RL * rho - x.segment<3>(L.drone_pos(i)) - Ri * hook;
  c.length = c.e.norm();
  c.q = c.e / c.length;
  c.e_dot = vL + RL * wL.cross(rho) - vi - Ri * wi.cross(hook);
  const double tension = physics_.cable_stiffness * (c.length - drones_[i].cable_length) +
                         physics_.cable_damping * c.e_dot.dot(c.q);
  c.slack = physics_.allow_slack && tension < 0.0;
  c.tension = c.slack ? 0.0 : tension;
  if (!with_jacobians) return c;

  const int n = L.size();
  const Matrix3d I = Matrix3d::Identity();
  const Matrix3d TL = T_matrix(thL);
  const Matrix3d Ti = T_matrix(thi);

  // eq. (28)
  c.De = MatrixXd::Zero(3, n);
  c.De.block<3, 3>(0, kPos) = I;
  c.De.block<3, 3>(0, kAtt) = -RL * skew(rho) * TL;
  c.De.block<3, 3>(0, L.drone_pos(i)) = -I;
  c.De.block<3, 3>(0, L.drone_att(i)) = Ri * skew(hook) * Ti;

  // eq. (30).  The drone rate is the gyro input here, so its block becomes
  // De_dot_gyr instead of a state column.
  c.De_dot = MatrixXd::Zero(3, n);
  c.De_dot.block<3, 3>(0, kVel) = I;
  c.De_dot.block<3, 3>(0, kAtt) = -RL * skew(wL.cross(rho)) * TL;
  c.De_dot.block<3, 3>(0, kRate) = -RL * skew(rho);
  c.De_dot.block<3, 3>(0, L.drone_vel(i)) = -I;
  c.De_dot.block<3, 3>(0, L.drone_att(i)) = Ri * skew(wi.cross(hook)) * Ti;
  c.De_dot_gyr = Ri * skew(hook);

  // eq. (31)
  const Matrix3d P = (I - c.q * c.q.transpose()) / c.length;
  c.Dq = P * c.De;

  // eq. (32)
  if (c.slack) {
    c.Dtau = RowVectorXd::Zero(n);
    c.Dtau_gyr.setZero();
  } else {
    const double k = physics_.cable_stiffness;
    const double damping = physics_.cable_damping;
    c.Dtau = k * c.q.transpose() * c.De + damping * (c.q.transpose() * c.De_dot + c.e_dot.transpose() * c.Dq);
    c.Dtau_gyr = damping * c.q.transpose() * c.De_dot_gyr;
  }
  return c;
}

// =============================================================================
// Process model
// =============================================================================
VectorXd UwbEkfDynamics::state_derivative(const VectorXd& x, const Inputs& u) const {
  check_sizes(x, u);
  const StateLayout& L = layout_;
  VectorXd xdot = VectorXd::Zero(L.size());

  const Vector3d thL = x.segment<3>(kAtt);
  const Vector3d wL = x.segment<3>(kRate);
  const Matrix3d RL = euler_to_R(thL);

  Vector3d force = payload_mass_ * gravity_;
  Vector3d torque = -wL.cross(payload_inertia_ * wL);
  for (int i = 0; i < L.n_drones; ++i) {
    // payload, eqs. (9) and (11) with the corrected sign
    const Cable c = compute_cable(x, i, u[i].gyr, false);
    force -= c.tension * c.q;
    torque -= c.tension * drones_[i].attach_point_payload.cross(RL.transpose() * c.q);

    // drone i, driven by its IMU
    const Vector3d thi = x.segment<3>(L.drone_att(i));
    xdot.segment<3>(L.drone_pos(i)) = x.segment<3>(L.drone_vel(i));
    xdot.segment<3>(L.drone_vel(i)) = gravity_ + euler_to_R(thi) * specific_force(x, i, u[i].acc);
    xdot.segment<3>(L.drone_att(i)) = T_inverse(thi) * body_rate(x, i, u[i].gyr);
    // bias rows stay zero (random walk)
  }

  xdot.segment<3>(kPos) = x.segment<3>(kVel);
  xdot.segment<3>(kVel) = force / payload_mass_;
  xdot.segment<3>(kAtt) = T_inverse(thL) * wL;
  xdot.segment<3>(kRate) = payload_inertia_inv_ * torque;
  return xdot;
}

void UwbEkfDynamics::jacobians(const VectorXd& x, const Inputs& u, MatrixXd& A, MatrixXd& B) const {
  check_sizes(x, u);
  const StateLayout& L = layout_;
  const int n = L.size();
  A = MatrixXd::Zero(n, n);
  B = MatrixXd::Zero(n, L.input_size());

  const Matrix3d I = Matrix3d::Identity();
  const Vector3d thL = x.segment<3>(kAtt);
  const Vector3d wL = x.segment<3>(kRate);
  const Matrix3d RL = euler_to_R(thL);
  const Matrix3d TL = T_matrix(thL);
  const Matrix3d& J = payload_inertia_;
  const Matrix3d& Jinv = payload_inertia_inv_;

  // payload position row, attitude eq. (35), and the gyroscopic part of eq. (36)
  A.block<3, 3>(kPos, kVel) = I;
  A.block<3, 3>(kAtt, kAtt) = d_Tinv_w(thL, wL);
  A.block<3, 3>(kAtt, kRate) = T_inverse(thL);
  A.block<3, 3>(kRate, kRate) = Jinv * (skew(J * wL) - skew(wL) * J);

  for (int i = 0; i < L.n_drones; ++i) {
    const Cable c = compute_cable(x, i, u[i].gyr, true);
    const Vector3d& rho = drones_[i].attach_point_payload;

    // eq. (33): cable direction in the payload frame and its Jacobian
    const Vector3d u_L = RL.transpose() * c.q;
    MatrixXd Du = RL.transpose() * c.Dq;
    Du.block<3, 3>(0, kAtt) += skew(u_L) * TL;

    // eq. (34): v_L' = g - sum tau_i q_i / m_L
    A.middleRows<3>(kVel) -= (c.q * c.Dtau + c.tension * c.Dq) / payload_mass_;
    // eq. (36), cable part: w_L' contains -J^-1 sum tau_i rho_i x u_i
    A.middleRows<3>(kRate) -= Jinv * (rho.cross(u_L) * c.Dtau + c.tension * skew(rho) * Du);

    // The gyro reaches the payload only through the cable damper.
    B.block<3, 3>(kVel, StateLayout::input_gyr(i)) = -c.q * c.Dtau_gyr / payload_mass_;
    B.block<3, 3>(kRate, StateLayout::input_gyr(i)) = -Jinv * rho.cross(u_L) * c.Dtau_gyr;

    // drone i: p' = v,  v' = g + R a,  Theta' = T^-1 w
    const Vector3d thi = x.segment<3>(L.drone_att(i));
    const Matrix3d Ri = euler_to_R(thi);
    const Vector3d a = specific_force(x, i, u[i].acc);
    const Vector3d w = body_rate(x, i, u[i].gyr);
    A.block<3, 3>(L.drone_pos(i), L.drone_vel(i)) = I;
    A.block<3, 3>(L.drone_vel(i), L.drone_att(i)) = -Ri * skew(a) * T_matrix(thi);  // rule (26)
    A.block<3, 3>(L.drone_att(i), L.drone_att(i)) = d_Tinv_w(thi, w);                // eq. (38)
    B.block<3, 3>(L.drone_vel(i), StateLayout::input_acc(i)) = Ri;
    B.block<3, 3>(L.drone_att(i), StateLayout::input_gyr(i)) = T_inverse(thi);
  }

  // f depends on the biases only through (imu - bias), so df/db = -df/du.
  if (L.with_bias) {
    for (int i = 0; i < L.n_drones; ++i) {
      A.middleCols<3>(L.gyro_bias(i)) = -B.middleCols<3>(StateLayout::input_gyr(i));
      A.middleCols<3>(L.acc_bias(i)) = -B.middleCols<3>(StateLayout::input_acc(i));
    }
  }
}

VectorXd UwbEkfDynamics::rk4_step(const VectorXd& x, const Inputs& u, double h) const {
  const VectorXd k1 = state_derivative(x, u);
  const VectorXd k2 = state_derivative(x + 0.5 * h * k1, u);
  const VectorXd k3 = state_derivative(x + 0.5 * h * k2, u);
  const VectorXd k4 = state_derivative(x + h * k3, u);
  return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// =============================================================================
// Measurement models
// =============================================================================
double UwbEkfDynamics::predict_uwb(const VectorXd& x, int host, int peer, RowVectorXd* H) const {
  check_drone(host);
  check_drone(peer);
  if (host == peer) throw std::invalid_argument("predict_uwb: host == peer");
  const StateLayout& L = layout_;
  const Vector3d& ant_h = drones_[host].uwb_antenna;
  const Vector3d& ant_p = drones_[peer].uwb_antenna;

  // eq. (41): antenna positions
  const Vector3d th_h = x.segment<3>(L.drone_att(host));
  const Vector3d th_p = x.segment<3>(L.drone_att(peer));
  const Matrix3d R_h = euler_to_R(th_h);
  const Matrix3d R_p = euler_to_R(th_p);
  const Vector3d diff = (x.segment<3>(L.drone_pos(peer)) + R_p * ant_p) - (x.segment<3>(L.drone_pos(host)) + R_h * ant_h);
  const double range = diff.norm();

  if (H) {
    // eq. (42): project the antenna Jacobians onto the line of sight
    const Eigen::RowVector3d nhat = (diff / range).transpose();
    *H = RowVectorXd::Zero(L.size());
    H->segment<3>(L.drone_pos(peer)) += nhat;
    H->segment<3>(L.drone_att(peer)) += nhat * (-R_p * skew(ant_p) * T_matrix(th_p));
    H->segment<3>(L.drone_pos(host)) -= nhat;
    H->segment<3>(L.drone_att(host)) -= nhat * (-R_h * skew(ant_h) * T_matrix(th_h));
  }
  return range;
}

Vector3d UwbEkfDynamics::predict_cable_dir(const VectorXd& x, int i, MatrixXd* H) const {
  // q and Dq do not depend on the gyro, so a zero reading is fine here.
  const Cable c = compute_cable(x, i, Vector3d::Zero(), H != nullptr);
  const Vector3d thi = x.segment<3>(layout_.drone_att(i));
  const Matrix3d Ri = euler_to_R(thi);
  const Vector3d s = Ri.transpose() * c.q;

  if (H) {
    // eq. (33): Ds = R^T Dq + [s^x T]_Theta_i
    *H = Ri.transpose() * c.Dq;
    H->block<3, 3>(0, layout_.drone_att(i)) += skew(s) * T_matrix(thi);
  }
  return s;
}

Vector6d UwbEkfDynamics::mocap_residual(const VectorXd& x, int i, const Vector3d& p_meas, const Matrix3d& R_meas,
                                        MatrixXd* H) const {
  check_drone(i);
  const StateLayout& L = layout_;
  const Vector3d thi = x.segment<3>(L.drone_att(i));

  Vector6d r;
  r.head<3>() = p_meas - x.segment<3>(L.drone_pos(i));
  r.tail<3>() = so3_log(euler_to_R(thi).transpose() * R_meas);

  if (H) {
    // eq. (40)
    *H = MatrixXd::Zero(6, L.size());
    H->block<3, 3>(0, L.drone_pos(i)) = Matrix3d::Identity();
    H->block<3, 3>(3, L.drone_att(i)) = T_matrix(thi);
  }
  return r;
}

// =============================================================================
// Rotation helpers
// =============================================================================
Matrix3d skew(const Vector3d& v) {
  Matrix3d S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return S;
}

Matrix3d euler_to_R(const Vector3d& theta) {
  return (Eigen::AngleAxisd(theta.z(), Vector3d::UnitZ()) * Eigen::AngleAxisd(theta.y(), Vector3d::UnitY()) *
          Eigen::AngleAxisd(theta.x(), Vector3d::UnitX()))
      .toRotationMatrix();
}

Vector3d R_to_euler(const Matrix3d& R) {
  const double roll = std::atan2(R(2, 1), R(2, 2));
  const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  return Vector3d(roll, pitch, yaw);
}

Matrix3d T_matrix(const Vector3d& theta) {
  const double sr = std::sin(theta.x()), cr = std::cos(theta.x());
  const double sp = std::sin(theta.y()), cp = std::cos(theta.y());
  Matrix3d T;
  T << 1.0, 0.0, -sp,
       0.0, cr, sr * cp,
       0.0, -sr, cr * cp;
  return T;
}

Matrix3d T_inverse(const Vector3d& theta) {
  const double sr = std::sin(theta.x()), cr = std::cos(theta.x());
  const double tp = std::tan(theta.y()), cp = std::cos(theta.y());
  Matrix3d Ti;
  Ti << 1.0, sr * tp, cr * tp,
        0.0, cr, -sr,
        0.0, sr / cp, cr / cp;
  return Ti;
}

Matrix3d dT_droll(const Vector3d& theta) {
  const double sr = std::sin(theta.x()), cr = std::cos(theta.x());
  const double cp = std::cos(theta.y());
  Matrix3d D;
  D << 0.0, 0.0, 0.0,
       0.0, -sr, cr * cp,
       0.0, -cr, -sr * cp;
  return D;
}

Matrix3d dT_dpitch(const Vector3d& theta) {
  const double sr = std::sin(theta.x()), cr = std::cos(theta.x());
  const double sp = std::sin(theta.y()), cp = std::cos(theta.y());
  Matrix3d D;
  D << 0.0, 0.0, -cp,
       0.0, 0.0, -sr * sp,
       0.0, 0.0, -cr * sp;
  return D;
}

Vector3d so3_log(const Matrix3d& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

double wrap_angle(double a) { return std::remainder(a, 2.0 * M_PI); }

void wrap_angles(const StateLayout& layout, VectorXd& x) {
  for (int k = 0; k < 3; ++k) {
    x[kAtt + k] = wrap_angle(x[kAtt + k]);
    for (int i = 0; i < layout.n_drones; ++i) x[layout.drone_att(i) + k] = wrap_angle(x[layout.drone_att(i) + k]);
  }
}

}  // namespace flycrane
