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
-------Thrust_Cable_Angle(Thrust axis vs cable direction over Time)
-------Cable_Direction(wrench_observed vs ground truth cable direction, body frame)
-------Cable_Angle_Error(measured vs true thrust-cable angle, direction error and measured tension)
----Drone2
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
-------Thrust_Cable_Angle(Thrust axis vs cable direction over Time)
-------Cable_Direction(wrench_observed vs ground truth cable direction, body frame)
-------Cable_Angle_Error(measured vs true thrust-cable angle, direction error and measured tension)
----Drone3
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
-------Thrust_Cable_Angle(Thrust axis vs cable direction over Time)
-------Cable_Direction(wrench_observed vs ground truth cable direction, body frame)
-------Cable_Angle_Error(measured vs true thrust-cable angle, direction error and measured tension)
----Payload
-------Overall Plots(Overall Error for Position and Orientation)
-------Position(Estimated vs Ground Truth)
-------Orientation(Estimated vs Ground Truth)
-------Position_Covariances(Evolution over Time)
-------Orientation_Covariances(Evolution over Time)
----overall
-------Box_Plots(Position and Attitude Error of every Body)
*/

#include "flycrane_ekf/plotter.hpp"
#include "flycrane_ekf/run_metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

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
const std::array<std::string, 3> kCableDirNames = {"s_x", "s_y", "s_z"};

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
  Triple pos_comp_error;             // p_est - p_truth per axis [m]
  Triple att_comp_error_deg;         // Euler angle est - truth per axis, wrapped to [-180, 180] [deg]
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
      s.pos_comp_error[k].push_back(x[pos_index + k] - truth->p[k]);
      s.att_comp_error_deg[k].push_back(std::remainder(x[att_index + k] - euler_truth[k], 2.0 * M_PI) * kRadToDeg);
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

// Shared with the experiment runner's results CSV (run_metrics.hpp).
using metrics::CableAngleSeries;
using metrics::CableMeasurementSeries;
using metrics::extract_cable_angles;
using metrics::extract_cable_measurements;
using metrics::mean;
using metrics::percent_below;
using metrics::rms;

std::string fixed(double v, int digits) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(digits) << v;
  return os.str();
}

// Index of the first error sample below the tolerance, i.e. where the estimate
// counts as converged, or v.size() if it never gets there.
std::size_t convergence_index(const std::vector<double>& v, double convergence_tolerance) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] < convergence_tolerance) return i;
  }
  return v.size();
}

std::string generate_convergence_report(const std::vector<double> v, const std::vector<double> t, const double& convergence_tolerance, const bool evaluating_pose){
  if (v.empty()) return "NaN";
  const std::size_t i = convergence_index(v, convergence_tolerance);
  if (i < v.size() && i < t.size()) {
    if (evaluating_pose){
      return " Converged to " + fixed(convergence_tolerance, 3) + "m in " + fixed(t[i], 3)+"s";
    }
    else{
      return " Converged to " + fixed(convergence_tolerance, 3) + "deg in " + fixed(t[i], 3)+"s";
    }
  }
  // it has not converged to within tolerance
  return "NEVER CONVERGED TO " + std::to_string(convergence_tolerance);
}

// One error series of one body, for the box plots.
struct ErrorSeries {
  std::string name;              // e.g. "Drone1"
  std::vector<double> t;         // log times with ground truth
  std::vector<double> error;     // error at those times
};

// Box plot (in the current subplot) of each body's error from its convergence
// on, i.e. without the initial transient.  Bodies that never converge get no
// box and are listed in the title instead.
void plot_converged_boxes(const std::vector<ErrorSeries>& bodies, double convergence_tolerance,
                          const std::string& title, const std::string& unit) {
  std::vector<std::vector<double>> boxes;
  std::vector<std::string> labels;
  std::string never_converged;
  for (const ErrorSeries& b : bodies) {
    const std::size_t i = convergence_index(b.error, convergence_tolerance);
    if (i >= b.error.size()) {
      never_converged += (never_converged.empty() ? "" : ", ") + b.name;
      continue;
    }
    boxes.emplace_back(b.error.begin() + static_cast<std::ptrdiff_t>(i), b.error.end());
    labels.push_back(b.name + "\n(from " + fixed(b.t[i], 2) + " s)");
  }

  // Boxes sit at x = 1..N.  The labels are set with xticks instead of
  // boxplot's own labels kwarg, which newer matplotlib renamed to tick_labels.
  if (!boxes.empty()) {
    std::vector<double> ticks;
    for (std::size_t k = 0; k < boxes.size(); k++) ticks.push_back(static_cast<double>(k + 1));
    plt::boxplot(boxes);
    plt::xticks(ticks, labels);
  }
  plt::ylabel(title + " [" + unit + "]");
  plt::grid(true);
  std::string full_title = title + " after convergence to " + fixed(convergence_tolerance, 3) + " " + unit;
  if (!never_converged.empty()) full_title += " (never converged: " + never_converged + ")";
  plt::title(full_title);
}

// Panel subtitles with the RMS error of each component, e.g. "x: RMS error 0.031 m".
std::array<std::string, 3> rms_subtitles(const Triple& error, const std::array<std::string, 3>& names,
                                         const std::string& unit, int digits) {
  std::array<std::string, 3> subtitles;
  for (int k = 0; k < 3; k++) subtitles[k] = names[k] + ": RMS error " + fixed(rms(error[k]), digits) + " " + unit;
  return subtitles;
}

// Three stacked panels (x/y/z or roll/pitch/yaw) of the estimate, with the
// ground truth dashed on top when there is any.  With subtitles, the title
// goes above the whole figure and each panel gets its subtitle.
void plot_estimate_vs_truth(const std::string& title, const std::vector<double>& t, const Triple& estimate,
                            const std::vector<double>& t_truth, const Triple& truth,
                            const std::array<std::string, 3>& names, const std::string& unit,
                            const std::string& estimate_label = "estimate",
                            const std::array<std::string, 3>& subtitles = {}) {
  const bool with_subtitles = !subtitles[0].empty();
  plt::figure_size(1000, 800);
  if (with_subtitles) plt::suptitle(title);
  for (int k = 0; k < 3; k++) {
    plt::subplot(3, 1, k + 1);
    plt::named_plot(estimate_label, t, estimate[k], "b-");
    if (!t_truth.empty()) plt::named_plot("ground truth", t_truth, truth[k], "k--");
    plt::ylabel(names[k] + " [" + unit + "]");
    plt::grid(true);
    if (with_subtitles) plt::title(subtitles[k], {{"fontsize", "10"}});
    if (k == 0) {
      if (!with_subtitles) plt::title(title);
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
    : drone_names_(params.drone_names), 
      output_dir_(output_dir), 
      show_(params.plot.show),
      save_plots_(params.plot.save_plots),
      pose_convergence_tol_(params.diagnostics.pose_convergence_tol), 
      ang_convergence_tol_(params.diagnostics.ang_convergence_tol),
      splay_vertical_tol_(params.diagnostics.splay_vertical_tol),
      all_covariances_(8),
      all_covariance_idx_(0),
      display_splay_(params.diagnostics.display_splay)  {
  // Same layout as the filter, so the state indices match the log.
  layout_.n_drones = static_cast<int>(params.drones.size());
  layout_.with_gyro_bias = params.filter.estimate_gyro_bias;
  layout_.with_acc_bias = params.filter.estimate_acc_bias;
  for (const DroneParams& drone : params.drones) {
    attach_points_.push_back(drone.attach_point_payload);
    hook_offsets_.push_back(drone.hook_offset);
  }
}

// =============================================================================
// Plotting
// =============================================================================
void TestPlotter::plot_test(const std::vector<LogEntry>& log) {
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
  for (std::size_t i = 0; i < all_covariances_.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << all_covariances_[i];
  }
  std::cout << "\n";
  if (display_splay_) {
    plot_cable_angles(log, bodies); 
  }
  if (save_plots_){
    plot_box_plots(log, bodies);
  }
  
  if (show_) plt::show();
}

void TestPlotter::plot_body(const std::vector<LogEntry>& log, const Body& body) {
  const fs::path folder = fs::path(output_dir_) / body.folder;
  if (save_plots_){
    fs::create_directories(folder);
  }

  const BodySeries s = extract_series(log, body.pos_index, body.att_index, body.drone);
  const bool have_truth = !s.t_truth.empty();

  // Overall Plots: error norms over time (needs ground truth)
  if (have_truth) {
    const double pos_rms = rms(s.pos_error);
    const double att_rms = rms(s.att_error_deg);
    all_covariances_[all_covariance_idx_] = pos_rms; 
    all_covariances_[all_covariance_idx_+1] = att_rms; 
    all_covariance_idx_ += 2; 

    const std::string pose_convergence_report = generate_convergence_report(s.pos_error, s.t_truth, pose_convergence_tol_, true); 
    const std::string angular_convergence_report = generate_convergence_report(s.att_error_deg, s.t_truth, ang_convergence_tol_, false); 
    std::cout << "  " << std::left << std::setw(20) << body.title << std::right << "  position RMS "
              << fixed(pos_rms, 3) << " m   attitude RMS " << fixed(att_rms, 2) << " deg | "  
              << pose_convergence_report << " | " << angular_convergence_report << "\n";
    if (save_plots_) {
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
    }
  } else {
    std::cout << "  " << std::left << std::setw(20) << body.title << std::right
              << "  no ground truth -- estimate and sigma only\n";
  }
  
  if (save_plots_) {
    // Position and Orientation: estimated vs ground truth, with the RMS error
    // of each component as the panel subtitle
    std::array<std::string, 3> pos_subtitles, att_subtitles;
    if (have_truth) {
      pos_subtitles = rms_subtitles(s.pos_comp_error, kAxisNames, "m", 3);
      att_subtitles = rms_subtitles(s.att_comp_error_deg, kAngleNames, "deg", 2);
    }
    plot_estimate_vs_truth(body.title + " position", s.t, s.pos, s.t_truth, s.pos_truth, kAxisNames, "m",
                           "estimate", pos_subtitles);
    save_figure(folder / "position.png", show_);
    plot_estimate_vs_truth(body.title + " orientation", s.t, s.att_deg, s.t_truth, s.att_truth_deg, kAngleNames,
                           "deg", "estimate", att_subtitles);
    save_figure(folder / "orientation.png", show_);

    // Position_Covariances and Orientation_Covariances: 1-sigma over time
    plot_sigma(body.title + " position 1-sigma", s.t, s.pos_sigma, kAxisNames, "m");
    save_figure(folder / "position_covariance.png", show_);
    plot_sigma(body.title + " orientation 1-sigma", s.t, s.att_sigma_deg, kAngleNames, "deg");
    save_figure(folder / "orientation_covariance.png", show_);
  }
}

void TestPlotter::plot_box_plots(const std::vector<LogEntry>& log, const std::vector<Body>& bodies) const {
  // The error series of every body with ground truth.
  std::vector<ErrorSeries> pos_errors;
  std::vector<ErrorSeries> att_errors_deg;
  for (const Body& body : bodies) {
    BodySeries s = extract_series(log, body.pos_index, body.att_index, body.drone);
    if (s.t_truth.empty()) continue;
    pos_errors.push_back({body.folder, s.t_truth, std::move(s.pos_error)});
    att_errors_deg.push_back({body.folder, std::move(s.t_truth), std::move(s.att_error_deg)});
  }
  if (pos_errors.empty()) return;

  const fs::path folder = fs::path(output_dir_) / "overall";

  if (save_plots_){
    fs::create_directories(folder);

    plt::figure_size(1000, 800);
    plt::subplot(2, 1, 1);
    plot_converged_boxes(pos_errors, pose_convergence_tol_, "position error", "m");
    plt::subplot(2, 1, 2);
    plot_converged_boxes(att_errors_deg, ang_convergence_tol_, "attitude error", "deg");
    save_figure(folder / "box_plots.png", show_);
  }
}

void TestPlotter::plot_cable_angles(const std::vector<LogEntry>& log, const std::vector<Body>& bodies) const {
  std::cout << "\ncable angles (ground truth, and wrench_observed vs ground truth):\n";
  const CableAngleSeries s = extract_cable_angles(log, attach_points_, hook_offsets_);
  if (s.t.empty()) {
    std::cout << "  no drone + payload ground truth -- skipped\n";
    return;
  }
  const std::vector<CableMeasurementSeries> meas = extract_cable_measurements(log, attach_points_, hook_offsets_);

  // Every drone has a sample at every time in s.t, so the mean of the
  // per-drone means is also the mean over all drones and times.
  double splay_mean_sum = 0.0;
  double thrust_cable_mean_sum = 0.0;
  int n_drones = 0;
  for (const Body& body : bodies) {
    if (body.drone < 0) continue;
    const std::vector<double>& splay = s.splay_deg[body.drone];
    const std::vector<double>& thrust_cable = s.thrust_cable_deg[body.drone];
    const double splay_mean = mean(splay);
    const double thrust_cable_mean = mean(thrust_cable);
    splay_mean_sum += splay_mean;
    thrust_cable_mean_sum += thrust_cable_mean;
    n_drones++;
    
    std::cout << "  " << std::left << std::setw(20) << body.title << std::right << "  splay mean "
              << fixed(splay_mean, 2) << " deg, min " << fixed(*std::min_element(splay.begin(), splay.end()), 2)
              << " deg, " << fixed(percent_below(splay, splay_vertical_tol_), 1) << " % below "
              << fixed(splay_vertical_tol_, 1) << " deg | thrust-cable mean " << fixed(thrust_cable_mean, 2)
              << " deg, min " << fixed(*std::min_element(thrust_cable.begin(), thrust_cable.end()), 2) << " deg\n";
    if (save_plots_){
      plt::figure_size(1000, 400);
      plt::plot(s.t, thrust_cable, "b-");
      plt::ylabel("thrust axis vs cable [deg]");
      plt::xlabel("time [s]");
      plt::grid(true);
      plt::title(body.title + " angle between thrust axis and cable (mean " + fixed(thrust_cable_mean, 2) + " deg)");
      save_figure(fs::path(output_dir_) / body.folder / "thrust_cable_angle.png", show_);
    }

    // measured (wrench_observed) cable direction against ground truth
    const CableMeasurementSeries& m = meas[body.drone];
    if (m.t.empty()) {
      std::cout << "  " << std::left << std::setw(20) << body.title << std::right
                << "  no cable measurement -- direction error skipped\n";
      continue;
    }
    std::vector<double> tilt_diff(m.t.size());
    for (std::size_t k = 0; k < m.t.size(); k++) tilt_diff[k] = m.tilt_meas_deg[k] - m.tilt_truth_deg[k];
    const double dir_mean = mean(m.dir_error_deg);
    const double dir_rms = rms(m.dir_error_deg);
    const double tilt_bias = mean(tilt_diff);

    // directions synthesised from geometry (no tension) are marked separately,
    // and only measured tensions are plotted
    std::vector<double> t_synth, err_synth, t_tension, tension;
    for (std::size_t k = 0; k < m.t.size(); k++) {
      if (m.tension_valid[k]) {
        t_tension.push_back(m.t[k]);
        tension.push_back(m.tension[k]);
      } else {
        t_synth.push_back(m.t[k]);
        err_synth.push_back(m.dir_error_deg[k]);
      }
    }

    std::cout << "  " << std::left << std::setw(20) << body.title << std::right << "  cable dir error mean "
              << fixed(dir_mean, 2) << " deg, RMS " << fixed(dir_rms, 2) << " deg, max "
              << fixed(*std::max_element(m.dir_error_deg.begin(), m.dir_error_deg.end()), 2) << " deg | tilt bias "
              << (tilt_bias >= 0.0 ? "+" : "") << fixed(tilt_bias, 2) << " deg";
    if (!t_synth.empty()) std::cout << " | " << t_synth.size() << " of " << m.t.size() << " synthesised";
    std::cout << "\n";

    if (save_plots_) {
      const fs::path folder = fs::path(output_dir_) / body.folder;
      plot_estimate_vs_truth(body.title + " cable direction, drone body frame (hook to payload)", m.t, m.s_meas,
                             m.t, m.s_truth, kCableDirNames, "-", "wrench_observed");
      save_figure(folder / "cable_direction.png", show_);

      plt::figure_size(1000, 800);
      plt::subplot(3, 1, 1);
      plt::named_plot("wrench_observed", m.t, m.tilt_meas_deg, "b-");
      plt::named_plot("ground truth", m.t, m.tilt_truth_deg, "k--");
      plt::ylabel("thrust axis vs cable [deg]");
      plt::grid(true);
      plt::title(body.title + " measured vs true cable angle (tilt bias " + fixed(tilt_bias, 2) + " deg)");
      plt::legend();
      plt::subplot(3, 1, 2);
      plt::plot(m.t, m.dir_error_deg, "b-");
      if (!t_synth.empty()) {
        plt::named_plot("synthesised (no tension)", t_synth, err_synth, "rx");
        plt::legend();
      }
      plt::ylabel("direction error [deg]");
      plt::grid(true);
      plt::title("angle between measured and true cable direction (mean " + fixed(dir_mean, 2) + " deg, RMS " +
                 fixed(dir_rms, 2) + " deg)");
      plt::subplot(3, 1, 3);
      if (!tension.empty()) plt::plot(t_tension, tension, "b-");
      plt::ylabel("tension [N]");
      plt::xlabel("time [s]");
      plt::grid(true);
      plt::title(tension.empty() ? std::string("measured cable tension (none: every direction synthesised)")
                                 : "measured cable tension (mean " + fixed(mean(tension), 2) + " N, min " +
                                       fixed(*std::min_element(tension.begin(), tension.end()), 2) + " N)");
      save_figure(folder / "cable_angle_error.png", show_);
    }
  }
  if (n_drones == 0) return;
  std::cout << "  " << std::left << std::setw(20) << "all drones" << std::right << "  splay mean "
            << fixed(splay_mean_sum / n_drones, 2) << " deg | thrust-cable mean "
            << fixed(thrust_cable_mean_sum / n_drones, 2) << " deg\n";
}

}  // namespace flycrane
