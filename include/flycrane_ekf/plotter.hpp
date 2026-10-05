#pragma once
// Plots the result of an EKF test run (run_ekf_test) with matplotlib-cpp.
//
// Output, one folder per body under the output directory:
//   Drone1/  Drone2/  Drone3/  Payload/
//     overall_error.png            position error norm and attitude error angle
//     position.png                 x, y, z: estimate vs ground truth
//     orientation.png              roll, pitch, yaw: estimate vs ground truth
//     position_covariance.png      1-sigma of x, y, z over time
//     orientation_covariance.png   1-sigma of roll, pitch, yaw over time
//   overall/
//     box_plots.png                position error norm and attitude error angle
//                                  of every body after it converges, as box plots
// Ground-truth curves, overall_error.png and the body's boxes are skipped for
// a body without ground truth (e.g. the payload, when the dataset has no
// payload.csv).

#include <string>
#include <vector>

#include "flycrane_ekf/dynamics.hpp"
#include "flycrane_ekf/ekf.hpp"
#include "flycrane_ekf/measurements.hpp"
#include "flycrane_ekf/params.hpp"

namespace flycrane {

// One logged instant of a test run: the filter estimate and the ground truth
// at the same time.
struct LogEntry {
  double t_s = 0.0;                     // seconds since the start of the dataset
  EkfSnapshot estimate;
  std::vector<PoseSample> drone_truth;  // interpolated odom, one per drone
  bool drone_truth_valid = false;       // false if the odom does not cover t
  PoseSample payload_truth{};           // interpolated payload.csv
  bool payload_truth_valid = false;     // false if payload.csv is absent or does not cover t
};

class TestPlotter {
 public:
  // Figures are written below output_dir (created if needed).
  TestPlotter(const UwbImuEkfParams& params, const std::string& output_dir);

  // Write every figure of the run.  With `plot.show: true` in the params the
  // figures are also opened in windows at the end.
  void plot_test(const std::vector<LogEntry>& log);

 private:
  // One body to plot: where its states are and where its figures go.
  struct Body {
    std::string folder;  // e.g. "Drone1"
    std::string title;   // e.g. "Drone1 (falcon6)"
    int pos_index;       // state index of x
    int att_index;       // state index of roll
    int drone;           // drone index, or -1 for the payload
  };

  void plot_body(const std::vector<LogEntry>& log, const Body& body);
  void plot_box_plots(const std::vector<LogEntry>& log, const std::vector<Body>& bodies) const;

  StateLayout layout_;
  std::vector<std::string> drone_names_;
  std::string output_dir_;
  bool show_;

  double pose_convergence_tol_; 
  double ang_convergence_tol_; 
  std::vector<double> all_covariances_;
  int all_covariance_idx_;  
};

}  // namespace flycrane
