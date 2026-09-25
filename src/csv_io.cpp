#include "flycrane_ekf/csv_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace flycrane::io {

namespace fs = std::filesystem;

namespace {

// Column schemas -- must match the *_COLS lists in bag_converter.py exactly
const std::vector<std::string> kImuCols = {"t_ns", "ax", "ay", "az", "gx", "gy", "gz"};
const std::vector<std::string> kOdomCols = {"t_ns", "px", "py", "pz", "qx", "qy", "qz", "qw",
                                            "vx",   "vy", "vz", "wx", "wy", "wz"};
const std::vector<std::string> kMotorCols = {"t_ns", "w0", "w1", "w2", "w3"};
const std::vector<std::string> kCableCols = {"t_ns", "sx", "sy", "sz", "tension"};
const std::vector<std::string> kUwbCols = {"t_ns", "host", "peer", "range"};

// A cable direction must be a unit vector.  The converter writes (0,0,0,0) when
// the observed force is ~zero; those rows carry no direction and are dropped.
constexpr double kUnitNormTol = 1e-3;

[[noreturn]] void fail(const fs::path& file, size_t line, const std::string& what) {
  std::ostringstream os;
  os << file.string();
  if (line > 0) os << ':' << line;
  os << ": " << what;
  throw std::runtime_error(os.str());
}

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
  return s;
}

std::string slurp(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, 0, "cannot open file");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Create Table Struct
struct RawTable {
  fs::path file;
  std::vector<std::string> header;
  std::vector<int64_t> t_ns;
  std::vector<double> vals;
  size_t nvals = 0;  // columns per row, excluding t_ns

  size_t rows() const { return t_ns.size(); }
  double at(size_t row, size_t col) const { return vals[row * nvals + col]; }
  // Blank lines are rejected, so data row r always sits on file line r + 2.
  static size_t line_of(size_t row) { return row + 2; }
};

bool starts_number(char c) {
  return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.';
}

// Strict reader for the converter's CSV contract: LF endings, a header row,
// then purely numeric rows with a fixed field count.  Anything else throws
// with the offending file:line rather than being skipped silently.
//
// strtoll/strtod are used instead of std::from_chars because Apple's libc++
// lacks floating-point from_chars.  strtod is correctly rounded, so %.17g
// values round-trip bit-exactly (assuming the default "C" numeric locale).
RawTable read_numeric_csv(const fs::path& path) {
  RawTable t;
  t.file = path;
  const std::string buf = slurp(path);
  const char* c = buf.c_str();  // NUL-terminated, so strto* cannot run past `end`
  const char* const end = c + buf.size();

  // ---- header
  const char* eol = std::find(c, end, '\n');
  const std::string hdr(c, eol);
  if (hdr.empty()) fail(path, 1, "missing header row");
  if (hdr.back() == '\r') fail(path, 1, "CRLF line endings; expected LF only");
  for (size_t start = 0;;) {
    const size_t comma = hdr.find(',', start);
    t.header.push_back(hdr.substr(start, comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (t.header.front() != "t_ns") fail(path, 1, "first column must be t_ns, got '" + t.header.front() + "'");
  t.nvals = t.header.size() - 1;
  c = (eol == end) ? end : eol + 1;

  const size_t approx_rows = static_cast<size_t>(std::count(c, end, '\n')) + 1;
  t.t_ns.reserve(approx_rows);
  t.vals.reserve(approx_rows * t.nvals);

  for (size_t line = 2; c < end; ++line) {
    char* next = nullptr;

    if (!starts_number(*c)) fail(path, line, *c == '\n' ? "blank line" : "row does not start with a number");
    errno = 0;
    const long long stamp = std::strtoll(c, &next, 10);
    if (next == c || errno == ERANGE) fail(path, line, "t_ns is not a valid int64");
    if (*next != ',' && *next != '\n' && next != end) fail(path, line, "t_ns must be an integer");
    t.t_ns.push_back(static_cast<int64_t>(stamp));
    c = next;

    for (size_t k = 0; k < t.nvals; ++k) {
      const std::string& col = t.header[k + 1];
      if (c >= end || *c != ',') {
        fail(path, line, "expected " + std::to_string(t.header.size()) + " fields, row ended before '" + col + "'");
      }
      ++c;
      if (c >= end || !starts_number(*c)) fail(path, line, "field '" + col + "' is not a number");
      const double v = std::strtod(c, &next);
      if (next == c) fail(path, line, "field '" + col + "' is not a number");
      if (!std::isfinite(v)) fail(path, line, "field '" + col + "' is not finite");
      t.vals.push_back(v);
      c = next;
    }

    if (c < end) {
      if (*c == '\r') fail(path, line, "CRLF line endings; expected LF only");
      if (*c != '\n') {
        fail(path, line, "unexpected characters after " + std::to_string(t.header.size()) + " fields");
      }
      ++c;
    }
  }
  return t;
}

void require_columns(const RawTable& t, const std::vector<std::string>& expected) {
  if (t.header != expected) {
    fail(t.file, 1, "columns are [" + join(t.header) + "], expected [" + join(expected) + "]");
  }
}

// Convert tables to measurement vectors
Eigen::Vector3d vec3(const RawTable& t, size_t r, size_t c0) {
  return Eigen::Vector3d(t.at(r, c0), t.at(r, c0 + 1), t.at(r, c0 + 2));
}

std::vector<ImuSample> to_imu(const RawTable& t, StreamReport&) {
  require_columns(t, kImuCols);
  std::vector<ImuSample> out;
  out.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    out.push_back({t.t_ns[r], vec3(t, r, 0), vec3(t, r, 3)});
  }
  return out;
}

std::vector<PoseSample> to_pose(const RawTable& t, StreamReport&) {
  require_columns(t, kOdomCols);
  std::vector<PoseSample> out;
  out.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    // CSV order is (qx, qy, qz, qw); Eigen's constructor takes (w, x, y, z)
    Eigen::Quaterniond q(t.at(r, 6), t.at(r, 3), t.at(r, 4), t.at(r, 5));
    const double n = q.norm();
    if (std::abs(n - 1.0) > 0.1) {
      fail(t.file, RawTable::line_of(r), "quaternion norm " + std::to_string(n) + " is far from 1");
    }
    q.coeffs() /= n;
    out.push_back({t.t_ns[r], vec3(t, r, 0), q, vec3(t, r, 7), vec3(t, r, 10)});
  }
  return out;
}

std::vector<MotorSample> to_motors(const RawTable& t, StreamReport&) {
  require_columns(t, kMotorCols);
  std::vector<MotorSample> out;
  out.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    out.push_back({t.t_ns[r], Eigen::Vector4d(t.at(r, 0), t.at(r, 1), t.at(r, 2), t.at(r, 3))});
  }
  return out;
}

std::vector<CableSample> to_cable(const RawTable& t, StreamReport& rep) {
  require_columns(t, kCableCols);
  std::vector<CableSample> out;
  out.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    const Eigen::Vector3d s = vec3(t, r, 0);
    const double n = s.norm();
    if (std::abs(n - 1.0) > kUnitNormTol) {  // check for small forces(cannot give direction)
      ++rep.dropped;
      continue;
    }
    const double tension = t.at(r, 3);
    out.push_back({t.t_ns[r], s / n, tension, tension > 0.0});
  }
  return out;
}

// convert the stream to UwbRange Objects
std::vector<UwbRange> to_uwb(const RawTable& t, StreamReport& rep, size_t n_drones) {
  require_columns(t, kUwbCols);
  auto drone_id = [&](size_t r, size_t col) -> uint8_t {
    const double v = t.at(r, col);
    if (v != std::floor(v) || v < 0.0 || v >= static_cast<double>(n_drones)) {
      std::ostringstream os;
      os << t.header[col + 1] << " = " << v << " is not a drone index in [0, " << n_drones
         << ") -- does the params YAML list every drone the converter used?";
      fail(t.file, RawTable::line_of(r), os.str());
    }
    return static_cast<uint8_t>(v);
  };

  std::vector<UwbRange> out;
  out.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    const uint8_t host = drone_id(r, 0);
    const uint8_t peer = drone_id(r, 1);
    if (host == peer) fail(t.file, RawTable::line_of(r), "host == peer");
    const double range = t.at(r, 2);
    if (range <= 0.0) {
      ++rep.dropped;
      continue;
    }
    out.push_back({t.t_ns[r], host, peer, range});
  }
  return out;
}

// Per-stream sorting and removal of duplicates
template <class T>
int64_t sample_key(const T& s) { return s.t_ns; }
std::tuple<int64_t, uint8_t, uint8_t> sample_key(const UwbRange& s) { return {s.t_ns, s.host, s.peer}; }

template <class T>
void sort_and_dedup(std::vector<T>& v, StreamReport& rep) {
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i].t_ns < v[i - 1].t_ns) ++rep.out_of_order;
  }
  std::stable_sort(v.begin(), v.end(),
                   [](const T& a, const T& b) { return sample_key(a) < sample_key(b); });
  // unique() keeps the first of each run, i.e. the earliest-recorded sample.
  const auto last = std::unique(v.begin(), v.end(),
                                [](const T& a, const T& b) { return sample_key(a) == sample_key(b); });
  rep.duplicates = static_cast<size_t>(std::distance(last, v.end()));
  v.erase(last, v.end());

  rep.kept = v.size();
  if (!v.empty()) {
    rep.t_min = v.front().t_ns;
    rep.t_max = v.back().t_ns;
  }
}

// Read/convert/sort one file and record what happened
template <class Convert>
auto load_stream(Dataset& ds, const fs::path& file, bool required, Convert convert) {
  StreamReport rep;
  rep.file = file.filename().string();
  decltype(convert(std::declval<const RawTable&>(), rep)) out;

  if (!fs::exists(file)) {
    if (required) fail(file, 0, "required file is missing");
    ds.warnings.push_back("optional stream not found: " + rep.file);
    ds.report.push_back(rep);
    return out;
  }

  rep.present = true;
  const RawTable raw = read_numeric_csv(file);
  rep.rows_read = raw.rows();
  out = convert(raw, rep);
  sort_and_dedup(out, rep);
  if (required && out.empty()) fail(file, 0, "required stream has no valid rows");
  if (out.size() > std::numeric_limits<uint32_t>::max()) fail(file, 0, "too many rows for a 32-bit event index");

  ds.report.push_back(rep);
  return out;
}

// Double check the names of the files
void validate_names(const std::vector<std::string>& names) {
  if (names.empty()) throw std::runtime_error("load_dataset: drone_names is empty");
  if (names.size() > std::numeric_limits<uint8_t>::max()) {
    throw std::runtime_error("load_dataset: at most 255 drones are supported");
  }
  std::unordered_set<std::string> seen;
  for (const auto& n : names) {
    if (n.empty()) throw std::runtime_error("load_dataset: empty drone name");
    if (!seen.insert(n).second) throw std::runtime_error("load_dataset: duplicate drone name '" + n + "'");
  }
}

// Check the manifest generated by the csv converter 
void check_manifest(const fs::path& dir, const std::vector<std::string>& names, Dataset& ds) {
  const fs::path path = dir / "manifest.json";
  if (!fs::exists(path)) {
    ds.warnings.push_back("manifest.json not found -- drone order NOT cross-checked against the converter "
                          "(a wrong order silently mis-assigns every UWB range)");
    return;
  }
  const std::string text = slurp(path);
  for (size_t i = 0; i < names.size(); ++i) {
    const std::string needle = "\"" + std::to_string(i) + "\": \"" + names[i] + "\"";
    if (text.find(needle) == std::string::npos) {
      fail(path, 0, "drone index " + std::to_string(i) + " is not '" + names[i] +
                        "' in drone_index_to_name. The drones list in the params YAML must match "
                        "the --drones order given to bag_converter.py");
    }
  }
}

template <class T>
void widen(int64_t& lo, int64_t& hi, const std::vector<T>& v) {
  if (v.empty()) return;
  lo = std::min(lo, v.front().t_ns);
  hi = std::max(hi, v.back().t_ns);
}

template <class T>
void narrow(int64_t& lo, int64_t& hi, const std::vector<T>& v) {
  lo = std::max(lo, v.front().t_ns);
  hi = std::min(hi, v.back().t_ns);
}

template <class T>
void append_events(std::vector<Event>& tl, Sensor type, uint8_t drone, const std::vector<T>& v,
                   int64_t lo, int64_t hi) {
  for (size_t i = 0; i < v.size(); ++i) {
    const int64_t t = v[i].t_ns;
    if (t >= lo && t <= hi) tl.push_back({t, type, drone, static_cast<uint32_t>(i)});
  }
}

// MoCap events from one drone's odom stream(~600Hz), downsampled to at rate_hz
void append_mocap_events(std::vector<Event>& tl, uint8_t drone, const std::vector<PoseSample>& odom,
                         double rate_hz, int64_t lo, int64_t hi) {
  const int64_t period_ns = rate_hz > 0.0 ? static_cast<int64_t>(std::llround(1e9 / rate_hz)) : 0;
  int64_t next_due = lo;
  for (size_t i = 0; i < odom.size(); ++i) {
    const int64_t t = odom[i].t_ns;
    if (t < lo || t > hi || t < next_due) continue;
    tl.push_back({t, Sensor::Mocap, drone, static_cast<uint32_t>(i)});
    next_due += period_ns;
    if (next_due <= t) next_due = t + period_ns;  // catch up after a gap in the odom stream
  }
}

}  // namespace

// load the full dataset
Dataset load_dataset(const LoadOptions& opts) {
  validate_names(opts.drone_names);
  if (!fs::is_directory(opts.dir)) {
    throw std::runtime_error("load_dataset: not a directory: " + opts.dir.string());
  }
  if (opts.mocap_drone >= static_cast<int>(opts.drone_names.size())) {
    throw std::runtime_error("load_dataset: mocap_drone " + std::to_string(opts.mocap_drone) +
                             " is not a drone index (-1 disables MoCap)");
  }

  Dataset ds;
  if (opts.check_manifest) check_manifest(opts.dir, opts.drone_names, ds);

  // per-drone streams
  const size_t n = opts.drone_names.size();
  ds.drones.resize(n);
  for (size_t i = 0; i < n; ++i) {
    DroneStreams& d = ds.drones[i];
    d.name = opts.drone_names[i];
    auto file = [&](const char* stem) { return opts.dir / (std::string(stem) + "_" + d.name + ".csv"); };

    d.imu = load_stream(ds, file("imu"), true, to_imu);
    d.odom = load_stream(ds, file("odom"), true, to_pose);
    if (opts.use_motors) d.motors = load_stream(ds, file("motors"), false, to_motors);
    if (opts.use_cable) d.cable = load_stream(ds, file("cable"), false, to_cable);
  }

  // UWB stream
  if (opts.use_uwb) {
    ds.uwb = load_stream(ds, opts.dir / "uwb.csv", false,
                         [n](const RawTable& t, StreamReport& rep) { return to_uwb(t, rep, n); });
  }

  // payload ground truth
  ds.payload = load_stream(ds, opts.dir / "payload.csv", false, to_pose);

  // fix time window
  int64_t lo = std::numeric_limits<int64_t>::max();
  int64_t hi = std::numeric_limits<int64_t>::min();
  if (opts.crop_to_overlap) {
    // Double check that all sensors and inputs are present
    std::swap(lo, hi);
    for (const auto& d : ds.drones) {
      narrow(lo, hi, d.imu);
      narrow(lo, hi, d.odom);
    }
    if (!ds.uwb.empty()) narrow(lo, hi, ds.uwb);
    if (!ds.payload.empty()) narrow(lo, hi, ds.payload);
    if (hi < lo) {
      throw std::runtime_error("load_dataset: IMU, odom" + std::string(ds.uwb.empty() ? "" : ", UWB") +
                               std::string(ds.payload.empty() ? "" : ", payload") + " streams do not overlap in time");
    }
  } else {
    for (const auto& d : ds.drones) {
      widen(lo, hi, d.imu);
      widen(lo, hi, d.motors);
      widen(lo, hi, d.cable);
    }
    widen(lo, hi, ds.uwb);
  }
  ds.t_first_ns = lo;
  ds.t_last_ns = hi;

  // Create merged timeline
  size_t total = ds.uwb.size();
  for (const auto& d : ds.drones) total += d.imu.size() + d.motors.size() + d.cable.size();
  if (opts.mocap_drone >= 0) total += ds.drones[opts.mocap_drone].odom.size();
  ds.timeline.reserve(total);
  for (size_t i = 0; i < n; ++i) {
    const auto id = static_cast<uint8_t>(i);
    const DroneStreams& d = ds.drones[i];
    append_events(ds.timeline, Sensor::Imu, id, d.imu, lo, hi);
    append_events(ds.timeline, Sensor::Motors, id, d.motors, lo, hi);
    append_events(ds.timeline, Sensor::Cable, id, d.cable, lo, hi);
  }
  for (size_t k = 0; k < ds.uwb.size(); ++k) {
    const UwbRange& u = ds.uwb[k];
    if (u.t_ns >= lo && u.t_ns <= hi) ds.timeline.push_back({u.t_ns, Sensor::Uwb, u.host, static_cast<uint32_t>(k)});
  }
  if (opts.mocap_drone >= 0) {
    const auto id = static_cast<uint8_t>(opts.mocap_drone);
    append_mocap_events(ds.timeline, id, ds.drones[id].odom, opts.mocap_rate_hz, lo, hi);
  }
  // Fix the order to make it deterministic(inputs go before updates)
  std::sort(ds.timeline.begin(), ds.timeline.end(), [](const Event& a, const Event& b) {
    return std::tie(a.t_ns, a.type, a.drone, a.idx) < std::tie(b.t_ns, b.type, b.drone, b.idx);
  });

  return ds;
}

void print_report(const Dataset& ds, std::ostream& os) {
  const auto flags = os.flags();
  const auto prec = os.precision();

  os << "drones:";
  for (size_t i = 0; i < ds.drones.size(); ++i) os << "  " << i << '=' << ds.drones[i].name;
  os << "\n\n";

  os << std::left << std::setw(30) << "file" << std::right << std::setw(9) << "rows" << std::setw(9) << "kept"
     << std::setw(9) << "dropped" << std::setw(8) << "unord" << std::setw(8) << "dup" << std::setw(21)
     << "span [s, rel. t0]" << '\n';
  os << std::fixed << std::setprecision(3);
  for (const auto& r : ds.report) {
    os << std::left << std::setw(30) << r.file << std::right;
    if (!r.present) {
      os << std::setw(9) << "--" << "  (not found)\n";
      continue;
    }
    os << std::setw(9) << r.rows_read << std::setw(9) << r.kept << std::setw(9) << r.dropped << std::setw(8)
       << r.out_of_order << std::setw(8) << r.duplicates;
    if (r.kept > 0) os << std::setw(10) << ds.rel_s(r.t_min) << " .. " << std::setw(7) << ds.rel_s(r.t_max);
    os << '\n';
  }

  size_t count[5] = {0, 0, 0, 0, 0};
  for (const auto& e : ds.timeline) ++count[static_cast<size_t>(e.type)];
  os << "\ntimeline: " << ds.timeline.size() << " events over " << ds.rel_s(ds.t_last_ns) << " s  (t0 = "
     << ds.t_first_ns << " ns)\n  ";
  for (Sensor s : {Sensor::Imu, Sensor::Motors, Sensor::Cable, Sensor::Uwb, Sensor::Mocap}) {
    os << to_string(s) << ' ' << count[static_cast<size_t>(s)] << "   ";
  }
  os << '\n';

  if (!ds.warnings.empty()) {
    os << "\nwarnings:\n";
    for (const auto& w : ds.warnings) os << "  ! " << w << '\n';
  }

  os.flags(flags);
  os.precision(prec);
}

std::optional<PoseSample> interpolate(const std::vector<PoseSample>& track, int64_t t_ns) {
  if (track.empty() || t_ns < track.front().t_ns || t_ns > track.back().t_ns) return std::nullopt;

  // First sample strictly after t_ns; since t_ns >= front, it is never begin().
  const auto next = std::upper_bound(track.begin(), track.end(), t_ns,
                                     [](int64_t t, const PoseSample& s) { return t < s.t_ns; });
  if (next == track.end()) return track.back();  // t_ns == back().t_ns

  const PoseSample& a = *std::prev(next);
  const PoseSample& b = *next;
  // Integer difference first: exact, and small enough to convert to double.
  const double u = static_cast<double>(t_ns - a.t_ns) / static_cast<double>(b.t_ns - a.t_ns);

  PoseSample out;
  out.t_ns = t_ns;
  out.p = a.p + u * (b.p - a.p);
  out.v = a.v + u * (b.v - a.v);
  out.w = a.w + u * (b.w - a.w);
  out.q = a.q.slerp(u, b.q).normalized();  // Eigen's slerp takes the shortest arc
  return out;
}

}  // namespace flycrane::io
