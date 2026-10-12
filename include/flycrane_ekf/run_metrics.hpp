#pragma once
// Summary metrics of one test run, computed from its log: the RMS errors of
// every body against ground truth, the ground-truth cable angles of every
// drone, and the error of the measured (wrench_observed) cable direction.  The plotter prints these numbers and the experiment runner writes
// them to its results CSV, so both use the helpers below.

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "flycrane_ekf/params.hpp"
#include "flycrane_ekf/plotter.hpp"

namespace flycrane {

namespace metrics {

// helper functions

//mean and rms calculations
double mean(const std::vector<double>& v);
double rms(const std::vector<double>& v);

// Percentage of the samples below the threshold
double percent_below(const std::vector<double>& v, double threshold);

// helper function for getting the angle between two vectors(a and b) 
double angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

// Ground-truth cable angles of every drone
struct CableAngleSeries {
  std::vector<double> t;
  std::vector<std::vector<double>> splay_deg;         // cable vs world vertical
  std::vector<std::vector<double>> thrust_cable_deg;  // drone body z vs cable
};

// rho is attachment points in payload frame, hook are offsets in drone frame
CableAngleSeries extract_cable_angles(const std::vector<LogEntry>& log, const std::vector<Eigen::Vector3d>& rho,
                                      const std::vector<Eigen::Vector3d>& hook);

// Measured cable direction (wrench_observed) of one drone against ground truth.
// Both are unit vectors in the drone body frame pointing from the drone hook
// down to the payload attach point, the convention of CableSample::s_body.
// Only log entries with drone + payload truth and a cable sample count, so
// every drone has its own times.
struct CableMeasurementSeries {
  std::vector<double> t;
  std::array<std::vector<double>, 3> s_meas;   // x, y, z of the measured direction
  std::array<std::vector<double>, 3> s_truth;  // x, y, z of the true direction
  std::vector<double> tilt_meas_deg;           // thrust axis vs cable, measured
  std::vector<double> tilt_truth_deg;          // thrust axis vs cable, ground truth
  std::vector<double> dir_error_deg;           // angle between measured and true direction
  std::vector<double> tension;                 // measured cable tension [N]
  std::vector<bool> tension_valid;             // false: direction synthesised from geometry
};

std::vector<CableMeasurementSeries> extract_cable_measurements(const std::vector<LogEntry>& log,
                                                               const std::vector<Eigen::Vector3d>& rho,
                                                               const std::vector<Eigen::Vector3d>& hook);

}  // namespace metrics

// RMS errors of one body over every log entry with ground truth
struct BodyMetrics {
  std::string name;          
  bool has_truth = false;
  double pos_rms_m = 0.0;    
  double vel_rms_mps = 0.0;  
  double att_rms_deg = 0.0; 
};

// Ground-truth cable angles of one drone.  NaN without drone + payload truth.
struct CableMetrics {
  std::string name;                  
  bool has_truth = false;
  double splay_mean_deg = 0.0;        
  double splay_min_deg = 0.0;
  double splay_below_tol_pct = 0.0;   
  double thrust_cable_mean_deg = 0.0;
  double thrust_cable_min_deg = 0.0;

  // Measured (wrench_observed) vs ground-truth cable direction.  NaN without
  // a cable sample at any log entry with drone + payload truth.
  bool has_cable_meas = false;
  double cable_dir_err_mean_deg = 0.0;
  double cable_dir_err_rms_deg = 0.0;
  double cable_dir_err_max_deg = 0.0;
  double cable_tilt_bias_deg = 0.0;   // mean of measured - true thrust-cable angle
};

struct RunMetrics {
  std::vector<BodyMetrics> bodies;    
  std::vector<CableMetrics> cables;   
};

RunMetrics compute_run_metrics(const UwbImuEkfParams& params, const std::vector<LogEntry>& log);

}  // namespace flycrane
