// Checks for src/dynamics.cpp, using the geometry in the params YAML:
//   1. rotation helpers: Euler round trip, T(Theta) against its definition
//      R(Theta)^T R(Theta + dTheta) = Exp(T dTheta) (Rule 1, eq. 25), T^-1,
//      and dT/dTheta
//   2. the analytic Jacobians A = df/dx, B = df/du and the UWB, cable and
//      MoCap measurement Jacobians against central finite differences, with
//      and without IMU bias states
//   3. physics: the vertical-cable equilibrium of eq. (18) is a fixed point,
//      and with the drones held still the payload conserves energy when the
//      cables are undamped and loses it when damped.  A wrong sign on any
//      cable force or torque breaks the energy balance.
//
//   test_dynamics [params.yaml]      (default: params/uwb_imu_ekf.yaml)
//
// Exit code 0 if every check passes.

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Dense>

#include "flycrane_ekf/dynamics.hpp"
#include "flycrane_ekf/params.hpp"

namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using flycrane::Inputs;
using flycrane::StateLayout;
using flycrane::UwbEkfDynamics;
using flycrane::UwbImuEkfParams;

constexpr int kPos = StateLayout::kPayloadPos;
constexpr int kVel = StateLayout::kPayloadVel;
constexpr int kAtt = StateLayout::kPayloadAtt;
constexpr int kRate = StateLayout::kPayloadRate;

int g_failures = 0;

void report(const std::string& name, bool ok, const std::string& detail) {
  std::cout << (ok ? "  pass  " : "  FAIL  ") << std::left << std::setw(34) << name << std::right << detail << '\n';
  if (!ok) ++g_failures;
}

std::string sci(double v) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(1) << v;
  return os.str();
}

// ---- names, so a failure points at the offending Jacobian entry
const char* const kAxis[] = {"x", "y", "z"};

std::string state_name(const StateLayout& L, int k) {
  static const char* const kPayload[] = {"L.pos", "L.vel", "L.att", "L.rate"};
  static const char* const kDrone[] = {"pos", "vel", "att", "bg", "ba"};
  if (k < StateLayout::kPayloadSize) return std::string(kPayload[k / 3]) + "." + kAxis[k % 3];
  const int i = (k - StateLayout::kPayloadSize) / L.drone_size();
  const int j = (k - StateLayout::kPayloadSize) % L.drone_size();
  return "d" + std::to_string(i) + "." + kDrone[j / 3] + "." + kAxis[j % 3];
}

std::string input_name(int k) { return "u" + std::to_string(k / 6) + (k % 6 < 3 ? ".acc." : ".gyr.") + kAxis[k % 3]; }

std::string row_index(int k) { return "row " + std::to_string(k); }

using VectorFn = std::function<VectorXd(const VectorXd&)>;
using NameFn = std::function<std::string(int)>;

MatrixXd numeric_jacobian(const VectorFn& fn, const VectorXd& x0) {
  const VectorXd y0 = fn(x0);
  MatrixXd J(y0.size(), x0.size());
  for (int k = 0; k < x0.size(); ++k) {
    const double h = 1e-6 * std::max(1.0, std::abs(x0[k]));
    VectorXd xp = x0;
    VectorXd xm = x0;
    xp[k] += h;
    xm[k] -= h;
    J.col(k) = (fn(xp) - fn(xm)) / (2.0 * h);
  }
  return J;
}

// Largest element-wise error |analytic - numeric| / (1 + |analytic|) seen over
// several comparisons, and where it happened.
struct Worst {
  double err = 0.0;
  std::string where;

  void add(const MatrixXd& analytic, const MatrixXd& numeric, const NameFn& row_name, const NameFn& col_name,
           const std::string& context) {
    if (analytic.rows() != numeric.rows() || analytic.cols() != numeric.cols()) {
      err = std::numeric_limits<double>::infinity();
      where = context + ": size mismatch";
      return;
    }
    for (int r = 0; r < analytic.rows(); ++r) {
      for (int c = 0; c < analytic.cols(); ++c) {
        const double e = std::abs(analytic(r, c) - numeric(r, c)) / (1.0 + std::abs(analytic(r, c)));
        if (e > err) {
          err = e;
          std::ostringstream os;
          os << context << " [" << row_name(r) << ", " << col_name(c) << "]  analytic " << analytic(r, c)
             << "  numeric " << numeric(r, c);
          where = os.str();
        }
      }
    }
  }
};

void report_worst(const std::string& name, const Worst& w, double tol = 1e-5) {
  report(name, w.err <= tol, "max err " + sci(w.err) + (w.err <= tol ? "" : "  at " + w.where));
}

// =============================================================================
// Random test points
// =============================================================================
double uniform(std::mt19937& rng, double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); }

Vector3d random_vec(std::mt19937& rng, double a) {
  const double x = uniform(rng, -a, a);
  const double y = uniform(rng, -a, a);
  const double z = uniform(rng, -a, a);
  return Vector3d(x, y, z);
}

// A generic flight state: payload moving and rotating, each drone above and
// outside its attach point with a taut cable splayed 15-25 deg from vertical.
VectorXd random_state(const UwbEkfDynamics& dyn, const UwbImuEkfParams& p, std::mt19937& rng) {
  const StateLayout& L = dyn.layout();
  VectorXd x = VectorXd::Zero(L.size());
  const Vector3d pL = Vector3d(0.2, -0.1, 1.0) + random_vec(rng, 0.2);
  const Vector3d thL = random_vec(rng, 0.3);
  const Matrix3d RL = flycrane::euler_to_R(thL);
  x.segment<3>(kPos) = pL;
  x.segment<3>(kVel) = random_vec(rng, 0.5);
  x.segment<3>(kAtt) = thL;
  x.segment<3>(kRate) = random_vec(rng, 1.0);

  for (int i = 0; i < L.n_drones; ++i) {
    const flycrane::DroneParams& d = p.drones[i];
    const Vector3d c = pL + RL * d.attach_point_payload;
    Vector3d out = c - pL;
    out.z() = 0.0;
    out = out.norm() > 1e-9 ? out.normalized() : Vector3d::UnitX();
    const double splay = uniform(rng, 0.25, 0.45);
    const Vector3d up = std::cos(splay) * Vector3d::UnitZ() + std::sin(splay) * out;
    const double length = d.cable_length + uniform(rng, 0.005, 0.03);
    const double roll = uniform(rng, -0.3, 0.3);
    const double pitch = uniform(rng, -0.3, 0.3);
    const double yaw = uniform(rng, -3.0, 3.0);
    const Vector3d thi(roll, pitch, yaw);

    x.segment<3>(L.drone_pos(i)) = c + length * up - flycrane::euler_to_R(thi) * d.hook_offset;
    x.segment<3>(L.drone_vel(i)) = random_vec(rng, 0.5);
    x.segment<3>(L.drone_att(i)) = thi;
    if (L.with_bias) {
      x.segment<3>(L.gyro_bias(i)) = random_vec(rng, 0.05);
      x.segment<3>(L.acc_bias(i)) = random_vec(rng, 0.3);
    }
  }
  return x;
}

Inputs random_inputs(int n_drones, std::mt19937& rng) {
  Inputs u(n_drones);
  for (auto& s : u) {
    s.t_ns = 0;
    s.acc = Vector3d(0.0, 0.0, 9.81) + random_vec(rng, 1.0);
    s.gyr = random_vec(rng, 1.0);
  }
  return u;
}

VectorXd flatten(const Inputs& u) {
  VectorXd v(6 * u.size());
  for (size_t i = 0; i < u.size(); ++i) {
    v.segment<3>(StateLayout::input_acc(i)) = u[i].acc;
    v.segment<3>(StateLayout::input_gyr(i)) = u[i].gyr;
  }
  return v;
}

Inputs unflatten(const VectorXd& v) {
  Inputs u(v.size() / 6);
  for (size_t i = 0; i < u.size(); ++i) {
    u[i].t_ns = 0;
    u[i].acc = v.segment<3>(StateLayout::input_acc(i));
    u[i].gyr = v.segment<3>(StateLayout::input_gyr(i));
  }
  return u;
}

// =============================================================================
// 1. Rotation helpers
// =============================================================================
void check_rotations(std::mt19937& rng) {
  std::cout << "\nrotation helpers\n";
  double round_trip = 0.0;
  double t_inverse = 0.0;
  Worst t_def, dT;
  const NameFn idx = [](int k) { return std::to_string(k); };

  for (int trial = 0; trial < 20; ++trial) {
    const double roll = uniform(rng, -1.2, 1.2);
    const double pitch = uniform(rng, -1.2, 1.2);
    const double yaw = uniform(rng, -3.0, 3.0);
    const Vector3d th(roll, pitch, yaw);
    const Matrix3d R0 = flycrane::euler_to_R(th);

    round_trip = std::max(round_trip, (flycrane::R_to_euler(R0) - th).norm());
    t_inverse = std::max(t_inverse, (flycrane::T_inverse(th) * flycrane::T_matrix(th) - Matrix3d::Identity()).norm());

    // Body-frame rotation produced by a change in Theta must be T dTheta.
    const MatrixXd T_fd = numeric_jacobian(
        [&](const VectorXd& t) -> VectorXd { return flycrane::so3_log(R0.transpose() * flycrane::euler_to_R(t)); }, th);
    t_def.add(flycrane::T_matrix(th), T_fd, idx, idx, "T");

    // dT/dTheta, with T flattened column-major to a 9-vector.
    const auto vec9 = [](const Matrix3d& M) -> VectorXd { return Eigen::Map<const VectorXd>(M.data(), 9); };
    MatrixXd dT_analytic = MatrixXd::Zero(9, 3);
    dT_analytic.col(0) = vec9(flycrane::dT_droll(th));
    dT_analytic.col(1) = vec9(flycrane::dT_dpitch(th));
    const MatrixXd dT_fd = numeric_jacobian([&](const VectorXd& t) -> VectorXd { return vec9(flycrane::T_matrix(t)); }, th);
    dT.add(dT_analytic, dT_fd, idx, idx, "dT");
  }
  report("euler -> R -> euler", round_trip < 1e-12, "max err " + sci(round_trip));
  report("T^-1 T = I", t_inverse < 1e-12, "max err " + sci(t_inverse));
  report_worst("T is the body-rate map (eq. 16)", t_def);
  report_worst("dT/droll, dT/dpitch", dT);
}

// =============================================================================
// 2. Jacobians
// =============================================================================
void check_jacobians(const UwbImuEkfParams& params, bool with_bias, std::mt19937& rng) {
  UwbImuEkfParams p = params;
  p.filter.estimate_imu_bias = with_bias;
  const UwbEkfDynamics dyn(p);
  const StateLayout& L = dyn.layout();
  std::cout << "\njacobians, " << (with_bias ? "with" : "without") << " IMU bias states (" << L.size()
            << " states)\n";

  const NameFn sname = [&L](int k) { return state_name(L, k); };
  Worst a, b, uwb, cable, mocap;
  for (int trial = 0; trial < 5; ++trial) {
    const VectorXd x = random_state(dyn, p, rng);
    const Inputs u = random_inputs(L.n_drones, rng);
    const std::string ctx = "trial " + std::to_string(trial);

    MatrixXd A, B;
    dyn.jacobians(x, u, A, B);
    a.add(A, numeric_jacobian([&](const VectorXd& xx) -> VectorXd { return dyn.state_derivative(xx, u); }, x),
          sname, sname, ctx);
    b.add(B,
          numeric_jacobian([&](const VectorXd& uu) -> VectorXd { return dyn.state_derivative(x, unflatten(uu)); },
                           flatten(u)),
          sname, input_name, ctx);

    for (int i = 0; i < L.n_drones; ++i) {
      for (int j = 0; j < L.n_drones; ++j) {
        if (i == j) continue;
        RowVectorXd H;
        dyn.predict_uwb(x, i, j, &H);
        const MatrixXd H_fd = numeric_jacobian(
            [&](const VectorXd& xx) -> VectorXd { return VectorXd::Constant(1, dyn.predict_uwb(xx, i, j)); }, x);
        uwb.add(H, H_fd, row_index, sname, ctx + " uwb " + std::to_string(i) + "->" + std::to_string(j));
      }

      MatrixXd H;
      dyn.predict_cable_dir(x, i, &H);
      cable.add(H, numeric_jacobian([&](const VectorXd& xx) -> VectorXd { return dyn.predict_cable_dir(xx, i); }, x),
                row_index, sname, ctx + " cable " + std::to_string(i));

      // H is exact at zero residual, so measure the current pose.  The
      // residual is z - h, hence its Jacobian is -H.
      const Vector3d p_meas = x.segment<3>(L.drone_pos(i));
      const Matrix3d R_meas = flycrane::euler_to_R(x.segment<3>(L.drone_att(i)));
      dyn.mocap_residual(x, i, p_meas, R_meas, &H);
      const MatrixXd r_fd = numeric_jacobian(
          [&](const VectorXd& xx) -> VectorXd { return dyn.mocap_residual(xx, i, p_meas, R_meas); }, x);
      mocap.add(H, -r_fd, row_index, sname, ctx + " mocap " + std::to_string(i));
    }
  }
  report_worst("A = df/dx", a);
  report_worst("B = df/du", b);
  report_worst("UWB H (eq. 42)", uwb);
  report_worst("cable direction H (eq. 33)", cable);
  report_worst("MoCap H (eq. 40)", mocap);
}

// =============================================================================
// 3. Physics
// =============================================================================
// The eq. (18) equilibrium with vertical cables: payload level and at rest,
// each drone straight above its attach point, tensions solving
//     sum tau_i = m_L g,    sum tau_i rho_i x z = 0,
// and cable lengths l_i = l_rest + tau_i / k so the springs supply them.
// Each drone's IMU reads exactly -g in its body frame, so v_i' = 0.
bool vertical_equilibrium(const UwbEkfDynamics& dyn, const UwbImuEkfParams& p, VectorXd& x, Inputs& u) {
  const StateLayout& L = dyn.layout();
  const int n = L.n_drones;
  MatrixXd M(3, n);
  for (int i = 0; i < n; ++i) {
    const Vector3d& rho = p.drones[i].attach_point_payload;
    M.col(i) = Vector3d(1.0, rho.y(), rho.x());
  }
  const Vector3d rhs(p.payload.mass * p.physics.gravity, 0.0, 0.0);
  const VectorXd tau = M.colPivHouseholderQr().solve(rhs);
  if ((M * tau - rhs).norm() > 1e-9 || tau.minCoeff() <= 0.0) return false;

  x = VectorXd::Zero(L.size());
  const Vector3d pL(0.0, 0.0, 1.0);
  x.segment<3>(kPos) = pL;
  u = Inputs(n);
  for (int i = 0; i < n; ++i) {
    const flycrane::DroneParams& d = p.drones[i];
    const double length = d.cable_length + tau[i] / p.physics.cable_stiffness;
    x.segment<3>(L.drone_pos(i)) = pL + d.attach_point_payload - d.hook_offset + length * Vector3d::UnitZ();
    u[i].t_ns = 0;
    u[i].acc = Vector3d(0.0, 0.0, p.physics.gravity);
    u[i].gyr = Vector3d::Zero();
  }
  return true;
}

// Payload kinetic + gravitational energy plus the elastic energy in the cables.
double system_energy(const UwbEkfDynamics& dyn, const UwbImuEkfParams& p, const VectorXd& x) {
  const Vector3d v = x.segment<3>(kVel);
  const Vector3d w = x.segment<3>(kRate);
  double energy = 0.5 * p.payload.mass * v.squaredNorm() +
                  0.5 * w.dot(p.payload.inertia.asDiagonal() * w) +
                  p.payload.mass * p.physics.gravity * x[kPos + 2];
  for (int i = 0; i < dyn.layout().n_drones; ++i) {
    const double stretch = dyn.compute_cable(x, i, Vector3d::Zero(), false).length - p.drones[i].cable_length;
    energy += 0.5 * p.physics.cable_stiffness * stretch * stretch;
  }
  return energy;
}

void check_physics(const UwbImuEkfParams& params) {
  std::cout << "\nphysics\n";
  UwbImuEkfParams p = params;
  p.filter.estimate_imu_bias = false;
  p.physics.allow_slack = false;  // the energy balance needs the spring in both directions

  const UwbEkfDynamics dyn(p);
  VectorXd x_eq;
  Inputs u;
  if (!vertical_equilibrium(dyn, p, x_eq, u)) {
    report("vertical equilibrium exists", false, "no positive tensions balance the payload -- check attach points");
    return;
  }
  const double xdot = dyn.state_derivative(x_eq, u).cwiseAbs().maxCoeff();
  report("equilibrium: x' = 0 (eq. 18)", xdot < 1e-9, "max |x'| " + sci(xdot));

  VectorXd x = x_eq;
  for (int step = 0; step < 1000; ++step) x = dyn.rk4_step(x, u, 1e-3);
  const double drift = (x - x_eq).cwiseAbs().maxCoeff();
  report("equilibrium: 1 s of RK4 stays put", drift < 1e-9, "max drift " + sci(drift));

  // Knock the payload off equilibrium; the drones stay where they are.
  VectorXd x0 = x_eq;
  x0.segment<3>(kPos) += Vector3d(0.03, -0.02, 0.02);
  x0.segment<3>(kVel) = Vector3d(0.1, 0.05, -0.1);
  x0.segment<3>(kAtt) = Vector3d(0.08, -0.05, 0.1);
  x0.segment<3>(kRate) = Vector3d(0.3, -0.2, 0.4);

  for (const bool damped : {false, true}) {
    UwbImuEkfParams pd = p;
    if (!damped) pd.physics.cable_damping = 0.0;
    const UwbEkfDynamics model(pd);
    const double e_eq = system_energy(model, pd, x_eq);
    const double e0 = system_energy(model, pd, x0);
    const double scale = e0 - e_eq;  // energy of the disturbance

    x = x0;
    double e_prev = e0;
    double max_change = 0.0;  // largest |E - E0|
    double max_rise = 0.0;    // largest single-step increase
    for (int step = 0; step < 2000; ++step) {  // 2 s
      x = model.rk4_step(x, u, 1e-3);
      const double e = system_energy(model, pd, x);
      max_change = std::max(max_change, std::abs(e - e0));
      max_rise = std::max(max_rise, e - e_prev);
      e_prev = e;
    }

    if (!damped) {
      const double rel = max_change / scale;
      report("undamped: energy conserved", rel < 1e-3, "max |E - E0| / E_disturbance " + sci(rel));
    } else {
      const double left = (e_prev - e_eq) / scale;
      const double rise = max_rise / scale;
      report("damped: energy decays", left < 0.5 && rise < 1e-6,
             "remaining " + sci(left) + " of disturbance, max step rise " + sci(rise));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string params_file = argc > 1 ? argv[1] : "params/uwb_imu_ekf.yaml";
  try {
    const UwbImuEkfParams params = flycrane::load_params(params_file);
    std::cout << "params: " << params_file << "\n  drones:";
    for (const auto& name : params.drone_names) std::cout << ' ' << name;
    std::cout << "\n  k = " << params.physics.cable_stiffness << " N/m, c = " << params.physics.cable_damping
              << " N s/m, m_L = " << params.payload.mass << " kg\n";

    std::mt19937 rng(42);
    check_rotations(rng);
    check_jacobians(params, false, rng);
    check_jacobians(params, true, rng);
    check_physics(params);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }

  std::cout << '\n' << (g_failures == 0 ? "ALL CHECKS PASSED" : std::to_string(g_failures) + " CHECK(S) FAILED") << '\n';
  return g_failures == 0 ? 0 : 1;
}
