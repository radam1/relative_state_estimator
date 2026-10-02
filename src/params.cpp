#include "flycrane_ekf/params.hpp"

#include <algorithm>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace flycrane {

namespace {

// load params(as yaml node) from file
YAML::Node load_file(const std::filesystem::path& yaml_file) {
  try {
    return YAML::LoadFile(yaml_file.string());
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(yaml_file.string() + ": " + e.what());
  }
}

// Keying the sections by file, path, and yaml node
struct Section {
  YAML::Node node;
  std::string file;
  std::string path; 

  std::string key_path(const std::string& key) const { return path.empty() ? key : path + "." + key; }

  [[noreturn]] void fail(const std::string& key, const std::string& what) const {
    throw std::runtime_error(file + ": " + key_path(key) + ": " + what);
  }

  YAML::Node get(const std::string& key) const {
    const YAML::Node n = node[key];
    if (!n) fail(key, "missing");
    return n;
  }

  Section sub(const std::string& key) const {
    const YAML::Node n = get(key);
    if (!n.IsMap()) fail(key, "expected a mapping");
    return {n, file, key_path(key)};
  }

  double number(const std::string& key) const {
    const YAML::Node n = get(key);
    try {
      if (n.IsScalar()) return n.as<double>();
    } catch (const YAML::Exception&) {
    }
    fail(key, "expected a number");
  }

  double positive(const std::string& key) const {
    const double v = number(key);
    if (!(v > 0.0)) fail(key, "must be > 0");
    return v;
  }

  double non_negative(const std::string& key) const {
    const double v = number(key);
    if (!(v >= 0.0)) fail(key, "must be >= 0");
    return v;
  }

  int integer(const std::string& key) const {
    const YAML::Node n = get(key);
    try {
      if (n.IsScalar()) return n.as<int>();
    } catch (const YAML::Exception&) {
    }
    fail(key, "expected an integer");
  }

  bool flag(const std::string& key) const {
    const YAML::Node n = get(key);
    try {
      if (n.IsScalar()) return n.as<bool>();
    } catch (const YAML::Exception&) {
    }
    fail(key, "expected true or false");
  }

  std::string text(const std::string& key) const {
    const YAML::Node n = get(key);
    if (!n.IsScalar()) fail(key, "expected a string");
    const std::string v = n.as<std::string>();
    if (v.empty()) fail(key, "must not be empty");
    return v;
  }

  Eigen::Vector3d vec3(const std::string& key) const {
    const YAML::Node n = get(key);
    if (!n.IsSequence() || n.size() != 3) fail(key, "expected [x, y, z]");
    Eigen::Vector3d v;
    for (int k = 0; k < 3; ++k) {
      try {
        v[k] = n[k].as<double>();
      } catch (const YAML::Exception&) {
        fail(key, "expected [x, y, z]");
      }
    }
    return v;
  }

  Eigen::Vector3d positive_vec3(const std::string& key) const {
    const Eigen::Vector3d v = vec3(key);
    if (!(v.minCoeff() > 0.0)) fail(key, "every entry must be > 0");
    return v;
  }
};

std::vector<std::string> read_drone_names(const Section& root) {
  const YAML::Node drones = root.get("drones");
  if (!drones.IsSequence() || drones.size() == 0) {
    root.fail("drones", "expected a non-empty list [ns0, ns1, ...]");
  }
  std::vector<std::string> names;
  for (const auto& d : drones) {
    if (!d.IsScalar()) root.fail("drones", "every entry must be a name");
    const std::string name = d.as<std::string>();
    if (std::find(names.begin(), names.end(), name) != names.end()) {
      root.fail("drones", "duplicate name '" + name + "'");
    }
    names.push_back(name);
  }
  return names;
}

}  // namespace

UwbImuEkfParams load_params(const std::filesystem::path& yaml_file) {
  const Section root{load_file(yaml_file), yaml_file.string(), ""};
  UwbImuEkfParams p;
  p.drone_names = read_drone_names(root);

  const Section physics = root.sub("physics");
  p.physics.gravity = physics.positive("gravity");
  p.physics.cable_stiffness = physics.positive("cable_stiffness");
  p.physics.cable_damping = physics.non_negative("cable_damping");
  p.physics.allow_slack = physics.flag("allow_slack");

  const Section payload = root.sub("payload");
  p.payload.mass = payload.positive("mass");
  p.payload.inertia = payload.positive_vec3("inertia");

  // Geometry is keyed by namespace, so reordering `drones` cannot silently
  // give a drone another drone's cable.
  const Section geometry = root.sub("drone_geometry");
  for (const std::string& name : p.drone_names) {
    const Section g = geometry.sub(name);
    DroneParams d;
    d.name = name;
    d.mass = g.positive("mass");
    d.inertia = g.positive_vec3("inertia");
    d.cable_length = g.positive("cable_length");
    d.attach_point_payload = g.vec3("attach_point_payload");
    d.hook_offset = g.vec3("hook_offset");
    d.uwb_antenna = g.vec3("uwb_antenna");
    p.drones.push_back(d);
  }

  const Section filter = root.sub("filter");
  p.filter.estimate_imu_bias = filter.flag("estimate_imu_bias");
  p.filter.anchor_drone = filter.integer("anchor_drone");
  if (p.filter.anchor_drone < 0 || p.filter.anchor_drone >= static_cast<int>(p.drones.size())) {
    filter.fail("anchor_drone", "must be a drone index in [0, " + std::to_string(p.drones.size()) + ")");
  }
  p.filter.max_integration_step = filter.positive("max_integration_step");
  p.filter.use_mocap = filter.flag("use_mocap");
  p.filter.use_uwb = filter.flag("use_uwb");
  p.filter.use_cable = filter.flag("use_cable");
  p.filter.mocap_rate_hz = filter.positive("mocap_rate_hz");
  p.filter.log_rate_hz = filter.positive("log_rate_hz");
  p.filter.init_pose_offset = filter.non_negative("init_pose_offset");
  p.filter.init_ang_offset = filter.non_negative("init_ang_offset");

  // Initial sigmas must be > 0 so P0 is positive definite.
  const Section init = root.sub("initial_sigma");
  p.initial_sigma.payload_pos = init.positive("payload_pos");
  p.initial_sigma.payload_vel = init.positive("payload_vel");
  p.initial_sigma.payload_att = init.positive("payload_att");
  p.initial_sigma.payload_rate = init.positive("payload_rate");
  p.initial_sigma.drone_pos = init.positive("drone_pos");
  p.initial_sigma.drone_vel = init.positive("drone_vel");
  p.initial_sigma.drone_att = init.positive("drone_att");
  p.initial_sigma.gyro_bias = init.positive("gyro_bias");
  p.initial_sigma.acc_bias = init.positive("acc_bias");

  const Section q = root.sub("process_noise");
  p.process_noise.acc = q.non_negative("acc");
  p.process_noise.gyro = q.non_negative("gyro");
  p.process_noise.acc_bias_walk = q.non_negative("acc_bias_walk");
  p.process_noise.gyro_bias_walk = q.non_negative("gyro_bias_walk");
  p.process_noise.payload_force = q.non_negative("payload_force");
  p.process_noise.payload_torque = q.non_negative("payload_torque");

  // Measurement sigmas must be > 0 so the innovation covariance is invertible.
  const Section r = root.sub("measurement_noise");
  p.measurement_noise.uwb = r.positive("uwb");
  p.measurement_noise.cable_dir = r.positive("cable_dir");
  p.measurement_noise.mocap_pos = r.positive("mocap_pos");
  p.measurement_noise.mocap_att = r.positive("mocap_att");

  const Section plot = root.sub("plot");
  p.plot.enabled = plot.flag("enabled");
  p.plot.show = plot.flag("show");
  p.plot.output_folder = plot.text("output_folder");

  const Section diagnostics = root.sub("diagnostics");
  p.diagnostics.pose_convergence_tol = diagnostics.positive("pose_convergence_tol");
  p.diagnostics.ang_convergence_tol = diagnostics.positive("ang_convergence_tol"); 
  return p;
}

std::vector<std::string> load_drone_names(const std::filesystem::path& yaml_file) {
  return read_drone_names(Section{load_file(yaml_file), yaml_file.string(), ""});
}

}  // namespace flycrane
