// Load a directory produced by bag_converter.py and sanity-check it.
//
//   dataset_info <csv_dir> [params.yaml] [--no-crop] [--events N]
//
// Prints the per-stream load report, the first N timeline events, and two
// physics checks against ground truth:
//   * UWB range vs. odom centre distance per (host, peer) pair.  The residual
//     should be within antenna offsets + noise (~0.2 m); a large mean points at
//     a time-base, unit or file mix-up.  NOTE: this can NOT reliably detect a
//     wrong drone order -- in a near-equilateral formation every pair has the
//     same length, so swapped drones give identical residuals.  The drone order
//     is guarded by load_dataset's manifest.json check instead.
//   * Mean IMU specific-force magnitude per drone (~9.81 m/s^2 near hover).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "flycrane_ekf/csv_io.hpp"
#include "flycrane_ekf/params.hpp"

namespace {

using flycrane::Sensor;
namespace io = flycrane::io;

void print_event(const io::Dataset& ds, const io::Event& e) {
  std::cout << "  " << std::setw(10) << ds.rel_s(e.t_ns) << "  " << std::left << std::setw(7)
            << flycrane::to_string(e.type) << std::right;
  switch (e.type) {
    case Sensor::Imu: {
      const auto& s = ds.imu(e);
      std::cout << ds.drones[e.drone].name << "  acc " << s.acc.transpose() << "  gyr " << s.gyr.transpose();
      break;
    }
    case Sensor::Motors:
      std::cout << ds.drones[e.drone].name << "  w " << ds.motors(e).w.transpose();
      break;
    case Sensor::Cable: {
      const auto& s = ds.cable(e);
      std::cout << ds.drones[e.drone].name << "  s " << s.s_body.transpose() << "  T "
                << (s.tension_valid ? std::to_string(s.tension) : std::string("n/a"));
      break;
    }
    case Sensor::Uwb: {
      const auto& u = ds.uwb_at(e);
      std::cout << ds.drones[u.host].name << " -> " << ds.drones[u.peer].name << "  r " << u.range;
      break;
    }
    case Sensor::Mocap:
      std::cout << ds.drones[e.drone].name << "  p " << ds.mocap(e).p.transpose();
      break;
  }
  std::cout << '\n';
}

struct Stats {
  size_t n = 0;
  double sum = 0, sum_sq = 0, max_abs = 0;
  void add(double x) {
    ++n;
    sum += x;
    sum_sq += x * x;
    max_abs = std::max(max_abs, std::abs(x));
  }
  double mean() const { return n ? sum / n : 0.0; }
  double stddev() const { return n > 1 ? std::sqrt(std::max(0.0, sum_sq / n - mean() * mean())) : 0.0; }
};

// Returns false if any pair's mean residual is implausibly large.
bool check_uwb_against_odom(const io::Dataset& ds) {
  if (ds.uwb.empty()) {
    std::cout << "\nuwb vs odom: no UWB data\n";
    return true;
  }
  std::map<std::pair<int, int>, Stats> pairs;
  size_t no_gt = 0;
  for (const auto& u : ds.uwb) {
    const auto a = io::interpolate(ds.ground_truth(u.host), u.t_ns);
    const auto b = io::interpolate(ds.ground_truth(u.peer), u.t_ns);
    if (!a || !b) {
      ++no_gt;
      continue;
    }
    pairs[{u.host, u.peer}].add(u.range - (a->p - b->p).norm());
  }

  constexpr double kMaxMeanResidual = 0.5;  // [m] antenna offsets are ~0.1 m per drone
  bool ok = true;
  std::cout << "\nuwb range - odom centre distance [m]   (expect |mean| < ~0.2, std ~ noise)\n";
  for (const auto& [key, s] : pairs) {
    const bool pair_ok = std::abs(s.mean()) < kMaxMeanResidual;
    ok &= pair_ok;
    std::cout << "  " << std::left << std::setw(20)
              << (ds.drones[key.first].name + " -> " + ds.drones[key.second].name) << std::right
              << "  n " << std::setw(7) << s.n << "  mean " << std::setw(7) << s.mean() << "  std "
              << std::setw(6) << s.stddev() << "  max|.| " << std::setw(6) << s.max_abs
              << (pair_ok ? "" : "   <-- SUSPICIOUS") << '\n';
  }
  if (no_gt) std::cout << "  (" << no_gt << " ranges outside odom coverage skipped)\n";
  if (!ok) std::cout << "  ! large residuals: ranges and odom disagree (time base, units, wrong files?)\n";
  return ok;
}

void check_imu(const io::Dataset& ds) {
  std::cout << "\nimu mean |specific force| [m/s^2]   (expect ~9.81 near hover)\n";
  for (const auto& d : ds.drones) {
    Stats s;
    for (const auto& m : d.imu) s.add(m.acc.norm());
    std::cout << "  " << std::left << std::setw(10) << d.name << std::right << "  " << s.mean() << '\n';
  }
}

int usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <csv_dir> [params.yaml] [--no-crop] [--events N]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  io::LoadOptions opts;
  std::string params = "params/uwb_imu_ekf.yaml";
  size_t n_events = 20;
  int positional = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--no-crop") {
      opts.crop_to_overlap = false;
    } else if (a == "--events" && i + 1 < argc) {
      n_events = std::strtoul(argv[++i], nullptr, 10);
    } else if (a == "-h" || a == "--help") {
      return usage(argv[0]);
    } else if (positional == 0) {
      opts.dir = a;
      ++positional;
    } else if (positional == 1) {
      params = a;
      ++positional;
    } else {
      return usage(argv[0]);
    }
  }
  if (positional == 0) return usage(argv[0]);

  try {
    opts.drone_names = flycrane::load_drone_names(params);
    const io::Dataset ds = io::load_dataset(opts);

    std::cout << std::fixed << std::setprecision(4);
    io::print_report(ds, std::cout);

    std::cout << "\nfirst " << std::min(n_events, ds.timeline.size()) << " events  [t rel. t0, s]\n";
    for (size_t i = 0; i < ds.timeline.size() && i < n_events; ++i) print_event(ds, ds.timeline[i]);

    const bool uwb_ok = check_uwb_against_odom(ds);
    check_imu(ds);
    return uwb_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
