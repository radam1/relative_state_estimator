#pragma once
// Summary metrics of one test run, computed from its log: the RMS errors of
// every body against ground truth and the ground-truth cable angles of every
// drone.  The plotter prints these numbers and the experiment runner writes
// them to its results CSV, so both use the helpers below.

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
};

struct RunMetrics {
  std::vector<BodyMetrics> bodies;    
  std::vector<CableMetrics> cables;   
};

RunMetrics compute_run_metrics(const UwbImuEkfParams& params, const std::vector<LogEntry>& log);

}  // namespace flycrane
