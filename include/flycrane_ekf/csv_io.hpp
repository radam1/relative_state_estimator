#pragma once
// Offline loader for the CSV tables written by bag_converter.py.
//
// Every stream is kept as its own time-sorted, typed vector
// `timeline` is a merged, fully ordered list of small Events(see obj) 

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "flycrane_ekf/measurements.hpp"

namespace flycrane::io {

// Create stream for a single drone to hold drone-specific data
struct DroneStreams {
  std::string name;                  // namespace(e.g. falcon1)
  std::vector<ImuSample> imu;        
  std::vector<PoseSample> odom;      // ground truth
  std::vector<MotorSample> motors;   
  std::vector<CableSample> cable;    
};

// Events to construct timeline of all measurements
struct Event {
  int64_t t_ns;
  Sensor type;
  uint8_t drone;
  uint32_t idx;
};

// What happened to one CSV file during loading for debugging
struct StreamReport {
  std::string file;
  bool present = false;
  size_t rows_read = 0;      // data rows in the file
  size_t dropped = 0;        // rows rejected as non-informative (e.g. zero cable force)
  size_t out_of_order = 0;   // rows whose stamp was earlier than the previous row's
  size_t duplicates = 0;     // rows removed because their key was already present
  size_t kept = 0;           // rows in the final vector
  int64_t t_min = 0;
  int64_t t_max = 0;
};

// full dataset object, holding all drones streams, the full timeline, and any warnings generated
struct Dataset {
  std::vector<DroneStreams> drones;  
  std::vector<UwbRange> uwb;         
  std::vector<PoseSample> payload;   
  std::vector<Event> timeline;       
  int64_t t_first_ns = 0;            
  int64_t t_last_ns = 0;
  std::vector<StreamReport> report;
  std::vector<std::string> warnings;

  const ImuSample& imu(const Event& e) const { return drones[e.drone].imu[e.idx]; }
  const MotorSample& motors(const Event& e) const { return drones[e.drone].motors[e.idx]; }
  const CableSample& cable(const Event& e) const { return drones[e.drone].cable[e.idx]; }
  const UwbRange& uwb_at(const Event& e) const { return uwb[e.idx]; }
  const PoseSample& mocap(const Event& e) const { return drones[e.drone].odom[e.idx]; }
  const std::vector<PoseSample>& ground_truth(size_t drone) const { return drones[drone].odom; }

  // Seconds since t_first_ns(for logging and plotting)
  double rel_s(int64_t t_ns) const { return 1e-9 * static_cast<double>(t_ns - t_first_ns); }
};

struct LoadOptions {
  std::filesystem::path dir;             // directory produced by bag_converter.py
  std::vector<std::string> drone_names;  
  bool use_motors = true;
  bool use_cable = true;
  bool use_uwb = true;
  // Whether or not to crop the timeline to where all sensors are available
  bool crop_to_overlap = true;
  // Cross-check drone_names against manifest.json
  bool check_manifest = true;
  int mocap_drone = -1; 
  double mocap_rate_hz = 0.0;
};

// Load, validate, sort and merge a converted bag. 
Dataset load_dataset(const LoadOptions& opts);

// Summary of the loaded dataset 
void print_report(const Dataset& ds, std::ostream& os);

// Ground-truth pose at t_ns(interpolated from other poses)
std::optional<PoseSample> interpolate(const std::vector<PoseSample>& track, int64_t t_ns);

}  // namespace flycrane::io
