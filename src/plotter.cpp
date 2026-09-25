/*
What is needed to plot?
1. Ground Truth MoCap
2. Estimated Pose for all drones
3. Covariances for each state for each drone

Plot folder outputs:
-<test_name>
----Drone1
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
----Drone2
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
----Drone3
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
----Payload
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
*/

#include "flycrane_ekf/plotter.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>

// Pass data to Python as plain lists instead of numpy arrays, so only the
// Python headers are needed (no numpy C API).
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace flycrane {

namespace fs = std::filesystem;

namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

// x/y/z or roll/pitch/yaw curves of one quantity
using Triple = std::array<std::vector<double>, 3>;

const std::array<std::string, 3> kAxisNames = {"x", "y", "z"};
const std::array<std::string, 3> kAngleNames = {"roll", "pitch", "yaw"};

// Everything one body's figures need, pulled out of the log as plain vectors.
struct BodySeries {
  std::vector<double> t;  // every log time
  Triple pos;             // estimate [m]
  Triple att_deg;         // estimate [deg]
  Triple pos_sigma;       // 1-sigma [m]
  Triple att_sigma_deg;   // 1-sigma [deg]

  std::vector<double> t_truth;       // log times that have ground truth
  Triple pos_truth;                  // [m]
  Triple att_truth_deg;              // [deg]
  std::vector<double> pos_error;     // |p_est - p_truth| [m]
  std::vector<double> att_error_deg; // angle of R_est^T R_truth [deg]
};

BodySeries extract_series(const std::vector<LogEntry>& log, int pos_index, int att_index, int drone) {
  BodySeries s;
  for (const LogEntry& entry : log) {
    const Eigen::VectorXd& x = entry.estimate.x;
    const Eigen::VectorXd& sigma = entry.estimate.sigma;

    // estimate and its uncertainty
    s.t.push_back(entry.t_s);
    for (int k = 0; k < 3; k++) {
      s.pos[k].push_back(x[pos_index + k]);
      s.att_deg[k].push_back(x[att_index + k] * kRadToDeg);
      s.pos_sigma[k].push_back(sigma[pos_index + k]);
      s.att_sigma_deg[k].push_back(sigma[att_index + k] * kRadToDeg);
    }

    // ground truth, if this entry has it for this body
    const PoseSample* truth = nullptr;
    if (drone >= 0 && entry.drone_truth_valid) truth = &entry.drone_truth[drone];
    if (drone < 0 && entry.payload_truth_valid) truth = &entry.payload_truth;
    if (truth == nullptr) continue;

    const Eigen::Matrix3d R_truth = truth->q.toRotationMatrix();
    const Eigen::Vector3d euler_truth = R_to_euler(R_truth);
    s.t_truth.push_back(entry.t_s);
    for (int k = 0; k < 3; k++) {
      s.pos_truth[k].push_back(truth->p[k]);
      s.att_truth_deg[k].push_back(euler_truth[k] * kRadToDeg);
    }

    // Overall errors.  The attitude error is the rotation angle between the
    // estimated and true attitude, so it never wraps like Euler angles do.
    const Eigen::Vector3d p_est = x.segment<3>(pos_index);
    const Eigen::Matrix3d R_est = euler_to_R(x.segment<3>(att_index));
    s.pos_error.push_back((p_est - truth->p).norm());
    s.att_error_deg.push_back(so3_log(R_est.transpose() * R_truth).norm() * kRadToDeg);
  }
  return s;
}

double rms(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum_sq = 0.0;
  for (double e : v) sum_sq += e * e;
  return std::sqrt(sum_sq / static_cast<double>(v.size()));
}

std::string fixed(double v, int digits) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(digits) << v;
  return os.str();
}

// Three stacked panels (x/y/z or roll/pitch/yaw) of the estimate, with the
// ground truth dashed on top when there is any.
void plot_estimate_vs_truth(const std::string& title, const std::vector<double>& t, const Triple& estimate,
                            const std::vector<double>& t_truth, const Triple& truth,
                            const std::array<std::string, 3>& names, const std::string& unit) {
  plt::figure_size(1000, 800);
  for (int k = 0; k < 3; k++) {
    plt::subplot(3, 1, k + 1);
    plt::named_plot("estimate", t, estimate[k], "b-");
    if (!t_truth.empty()) plt::named_plot("ground truth", t_truth, truth[k], "k--");
    plt::ylabel(names[k] + " [" + unit + "]");
    plt::grid(true);
    if (k == 0) {
      plt::title(title);
      plt::legend();
    }
  }
  plt::xlabel("time [s]");
}

// Three stacked panels of the 1-sigma of each component.
void plot_sigma(const std::string& title, const std::vector<double>& t, const Triple& sigma,
                const std::array<std::string, 3>& names, const std::string& unit) {
  plt::figure_size(1000, 800);
  for (int k = 0; k < 3; k++) {
    plt::subplot(3, 1, k + 1);
    plt::plot(t, sigma[k], "r-");
    plt::ylabel("sigma " + names[k] + " [" + unit + "]");
    plt::grid(true);
    if (k == 0) plt::title(title);
  }
  plt::xlabel("time [s]");
}

// Save the current figure, and close it unless it will be shown at the end.
void save_figure(const fs::path& file, bool keep_open) {
  plt::tight_layout();
  plt::save(file.string());
  if (!keep_open) plt::close();
}

}  // namespace

// =============================================================================
// Setup
// =============================================================================
TestPlotter::TestPlotter(const UwbImuEkfParams& params, const std::string& output_dir)
    : drone_names_(params.drone_names), output_dir_(output_dir), show_(params.plot.show) {
  // Same layout as the filter, so the state indices match the log.
  layout_.n_drones = static_cast<int>(params.drones.size());
  layout_.with_bias = params.filter.estimate_imu_bias;
}

// =============================================================================
// Plotting
// =============================================================================
void TestPlotter::plot_test(const std::vector<LogEntry>& log) const {
  if (log.empty()) {
    std::cout << "plotter: the log is empty, nothing to plot\n";
    return;
  }

  // The backend must be chosen before matplotlib is first used.  "Agg" only
  // writes image files, so it also works without a display.
  if (!show_) plt::backend("Agg");

  // Drone1..DroneN (1-based folder names, as in the layout above), then the payload.
  std::vector<Body> bodies;
  for (int i = 0; i < layout_.n_drones; i++) {
    const std::string folder = "Drone" + std::to_string(i + 1);
    bodies.push_back({folder, folder + " (" + drone_names_[i] + ")", layout_.drone_pos(i), layout_.drone_att(i), i});
  }
  bodies.push_back({"Payload", "Payload", StateLayout::kPayloadPos, StateLayout::kPayloadAtt, -1});

  std::cout << "\nplots: " << output_dir_ << '\n';
  for (const Body& body : bodies) plot_body(log, body);

  if (show_) plt::show();
}

void TestPlotter::plot_body(const std::vector<LogEntry>& log, const Body& body) const {
  const fs::path folder = fs::path(output_dir_) / body.folder;
  fs::create_directories(folder);

  const BodySeries s = extract_series(log, body.pos_index, body.att_index, body.drone);
  const bool have_truth = !s.t_truth.empty();

  // Overall Plots: error norms over time (needs ground truth)
  if (have_truth) {
    const double pos_rms = rms(s.pos_error);
    const double att_rms = rms(s.att_error_deg);
    std::cout << "  " << std::left << std::setw(20) << body.title << std::right << "  position RMS "
              << fixed(pos_rms, 3) << " m   attitude RMS " << fixed(att_rms, 2) << " deg\n";

    plt::figure_size(1000, 600);
    plt::subplot(2, 1, 1);
    plt::plot(s.t_truth, s.pos_error, "b-");
    plt::ylabel("position error [m]");
    plt::grid(true);
    plt::title(body.title + " overall error (RMS " + fixed(pos_rms, 3) + " m, " + fixed(att_rms, 2) + " deg)");
    plt::subplot(2, 1, 2);
    plt::plot(s.t_truth, s.att_error_deg, "b-");
    plt::ylabel("attitude error [deg]");
    plt::xlabel("time [s]");
    plt::grid(true);
    save_figure(folder / "overall_error.png", show_);
  } else {
    std::cout << "  " << std::left << std::setw(20) << body.title << std::right
              << "  no ground truth -- estimate and sigma only\n";
  }

  // Position and Orientation: estimated vs ground truth
  plot_estimate_vs_truth(body.title + " position", s.t, s.pos, s.t_truth, s.pos_truth, kAxisNames, "m");
  save_figure(folder / "position.png", show_);
  plot_estimate_vs_truth(body.title + " orientation", s.t, s.att_deg, s.t_truth, s.att_truth_deg, kAngleNames, "deg");
  save_figure(folder / "orientation.png", show_);

  // Position_Covariances and Orientation_Covariances: 1-sigma over time
  plot_sigma(body.title + " position 1-sigma", s.t, s.pos_sigma, kAxisNames, "m");
  save_figure(folder / "position_covariance.png", show_);
  plot_sigma(body.title + " orientation 1-sigma", s.t, s.att_sigma_deg, kAngleNames, "deg");
  save_figure(folder / "orientation_covariance.png", show_);
}

}  // namespace flycrane
