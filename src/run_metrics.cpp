#include "flycrane_ekf/run_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "flycrane_ekf/dynamics.hpp"

namespace flycrane {

namespace metrics {

double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum = 0.0;
  for (double e : v) sum += e;
  return sum / static_cast<double>(v.size());
}

double rms(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum_sq = 0.0;
  for (double e : v) sum_sq += e * e;
  return std::sqrt(sum_sq / static_cast<double>(v.size()));
}

double percent_below(const std::vector<double>& v, double threshold) {
  if (v.empty()) return 0.0;
  const auto n = std::count_if(v.begin(), v.end(), [threshold](double e) { return e < threshold; });
  return 100.0 * static_cast<double>(n) / static_cast<double>(v.size());
}

double angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

CableAngleSeries extract_cable_angles(const std::vector<LogEntry>& log, const std::vector<Eigen::Vector3d>& rho,
                                      const std::vector<Eigen::Vector3d>& hook) {
  constexpr double kRadToDeg = 180.0 / M_PI;
  const std::size_t n_drones = rho.size();
  CableAngleSeries s;
  s.splay_deg.resize(n_drones);
  s.thrust_cable_deg.resize(n_drones);
  for (const LogEntry& entry : log) {
    if (!entry.drone_truth_valid || !entry.payload_truth_valid) continue;
    const PoseSample& payload = entry.payload_truth;
    const Eigen::Matrix3d R_L = payload.q.toRotationMatrix();
    s.t.push_back(entry.t_s);
    for (std::size_t i = 0; i < n_drones; i++) {
      const PoseSample& drone = entry.drone_truth[i];
      const Eigen::Matrix3d R_i = drone.q.toRotationMatrix();
      // From the payload attach point up to the drone hook (minus Cable::e),
      // so a taut hanging cable points along +z, the same way as the thrust.
      const Eigen::Vector3d cable = drone.p + R_i * hook[i] - payload.p - R_L * rho[i];
      s.splay_deg[i].push_back(angle_between(cable, Eigen::Vector3d::UnitZ()) * kRadToDeg);
      s.thrust_cable_deg[i].push_back(angle_between(R_i.col(2), cable) * kRadToDeg);
    }
  }
  return s;
}

}  // namespace metrics

namespace {

constexpr double kRadToDeg = 180.0 / M_PI;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// pos/vel/att_index: state index of x, vx and roll; drone: index, or -1 for the payload
BodyMetrics body_metrics(const std::vector<LogEntry>& log, const std::string& name, int pos_index, int vel_index,
                         int att_index, int drone) {
  std::vector<double> pos_error, vel_error, att_error_deg;
  for (const LogEntry& entry : log) {
    const PoseSample* truth = nullptr;
    if (drone >= 0 && entry.drone_truth_valid) truth = &entry.drone_truth[drone];
    if (drone < 0 && entry.payload_truth_valid) truth = &entry.payload_truth;
    if (truth == nullptr) continue;

    const Eigen::VectorXd& x = entry.estimate.x;
    const Eigen::Matrix3d R_est = euler_to_R(x.segment<3>(att_index));
    pos_error.push_back((x.segment<3>(pos_index) - truth->p).norm());
    vel_error.push_back((x.segment<3>(vel_index) - truth->v).norm());
    att_error_deg.push_back(so3_log(R_est.transpose() * truth->q.toRotationMatrix()).norm() * kRadToDeg);
  }

  BodyMetrics m;
  m.name = name;
  m.has_truth = !pos_error.empty();
  m.pos_rms_m = m.has_truth ? metrics::rms(pos_error) : kNaN;
  m.vel_rms_mps = m.has_truth ? metrics::rms(vel_error) : kNaN;
  m.att_rms_deg = m.has_truth ? metrics::rms(att_error_deg) : kNaN;
  return m;
}

}  // namespace

RunMetrics compute_run_metrics(const UwbImuEkfParams& params, const std::vector<LogEntry>& log) {
  // Same layout as the filter, so the state indices match the log.
  StateLayout layout;
  layout.n_drones = static_cast<int>(params.drones.size());
  layout.with_bias = params.filter.estimate_imu_bias;

  RunMetrics m;
  for (int i = 0; i < layout.n_drones; i++) {
    m.bodies.push_back(body_metrics(log, "Drone" + std::to_string(i + 1), layout.drone_pos(i), layout.drone_vel(i),
                                    layout.drone_att(i), i));
  }
  m.bodies.push_back(body_metrics(log, "Payload", StateLayout::kPayloadPos, StateLayout::kPayloadVel,
                                  StateLayout::kPayloadAtt, -1));

  std::vector<Eigen::Vector3d> rho, hook;
  for (const DroneParams& drone : params.drones) {
    rho.push_back(drone.attach_point_payload);
    hook.push_back(drone.hook_offset);
  }
  const metrics::CableAngleSeries s = metrics::extract_cable_angles(log, rho, hook);
  for (int i = 0; i < layout.n_drones; i++) {
    CableMetrics c;
    c.name = "Drone" + std::to_string(i + 1);
    c.has_truth = !s.t.empty();
    if (c.has_truth) {
      const std::vector<double>& splay = s.splay_deg[i];
      const std::vector<double>& thrust_cable = s.thrust_cable_deg[i];
      c.splay_mean_deg = metrics::mean(splay);
      c.splay_min_deg = *std::min_element(splay.begin(), splay.end());
      c.splay_below_tol_pct = metrics::percent_below(splay, params.diagnostics.splay_vertical_tol);
      c.thrust_cable_mean_deg = metrics::mean(thrust_cable);
      c.thrust_cable_min_deg = *std::min_element(thrust_cable.begin(), thrust_cable.end());
    } else {
      c.splay_mean_deg = c.splay_min_deg = c.splay_below_tol_pct = kNaN;
      c.thrust_cable_mean_deg = c.thrust_cable_min_deg = kNaN;
    }
    m.cables.push_back(c);
  }
  return m;
}

}  // namespace flycrane
