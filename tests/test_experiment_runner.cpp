// Checks for src/experiment_runner.cpp and the TestConfig overrides it uses:
//   1. parsing: params/experiment_setup.json loads in file order with one seed
//      per run; an integer run_seeds is a base seed (s, s+1, ...); every kind
//      of malformed bag throws an error naming its key path
//   2. inputs: every missing data folder and param file is reported, and
//      run_experiment() throws before any run starts (no results.csv)
//   3. overrides: load_test_params leaves the YAML alone without overrides and
//      applies the seed / save_plots overrides when given
//   4. end to end, on a small synthetic dataset written to a temp folder:
//      one results.csv row per run in file order, equal seeds give identical
//      metrics and different seeds do not, a broken bag is recorded as failed
//      without stopping the other runs, save_plots decides whether plots are
//      written to run<k>_seed<s>/, and the vertical synthetic cables give
//      ~0 deg splay.
//
//   test_experiment_runner [repo root]      (default: .)
//
// Exit code 0 if every check passes.

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "flycrane_ekf/experiment_runner.hpp"
#include "flycrane_ekf/params.hpp"
#include "flycrane_ekf/run_ekf_test.hpp"

namespace {

namespace fs = std::filesystem;
using flycrane::BagSettings;
using flycrane::ExperimentPaths;
using flycrane::ExperimentRunner;
using flycrane::ExperimentSettings;
using flycrane::UwbImuEkfParams;

int g_failures = 0;

void report(const std::string& name, bool ok, const std::string& detail = "") {
  std::cout << (ok ? "  pass  " : "  FAIL  ") << std::left << std::setw(44) << name << std::right << detail << '\n';
  if (!ok) ++g_failures;
}

void write_file(const fs::path& file, const std::string& text) {
  fs::create_directories(file.parent_path());
  std::ofstream out(file);
  out << text;
}

// Run fn and check that it throws an error containing `expected`.
void expect_throw(const std::string& name, const std::function<void()>& fn, const std::string& expected) {
  try {
    fn();
    report(name, false, "did not throw");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    report(name, msg.find(expected) != std::string::npos, "\"" + msg + "\"");
  }
}

// One bag of a JSON, with the fields as raw JSON text.
std::string bag_json(const std::string& name, const std::string& runs, const std::string& seeds,
                     const std::string& extra = "") {
  return "\"" + name + "\": {\"param_file\": \"p.yaml\", \"runs\": " + runs + ", \"run_seeds\": " + seeds +
         ", \"save_plots\": false" + extra + "}";
}

// ---- CSV reading (handles quoted fields)
std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields(1);
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); i++) {
    const char c = line[i];
    if (quoted) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        fields.back() += '"';
        i++;
      } else if (c == '"') {
        quoted = false;
      } else {
        fields.back() += c;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.emplace_back();
    } else {
      fields.back() += c;
    }
  }
  return fields;
}

struct Csv {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;

  int col(const std::string& name) const {
    for (std::size_t i = 0; i < header.size(); i++) {
      if (header[i] == name) return static_cast<int>(i);
    }
    return -1;
  }
  std::string at(std::size_t row, const std::string& name) const {
    const int c = col(name);
    return (c < 0 || row >= rows.size() || c >= static_cast<int>(rows[row].size())) ? "<missing>" : rows[row][c];
  }
};

Csv read_csv(const fs::path& file) {
  Csv csv;
  std::ifstream in(file);
  std::string line;
  if (std::getline(in, line)) csv.header = split_csv_line(line);
  while (std::getline(in, line)) {
    if (!line.empty()) csv.rows.push_back(split_csv_line(line));
  }
  return csv;
}

// =============================================================================
// 1. parsing
// =============================================================================
void check_parsing(const fs::path& repo, const fs::path& tmp) {
  std::cout << "\nparsing\n";
  const ExperimentPaths paths{"data_root", "params_dir", "out"};

  // the real experiment file
  const std::vector<ExperimentSettings> plan =
      flycrane::load_experiment_plan((repo / "params" / "experiment_setup.json").string(), paths);
  std::vector<std::string> names;
  bool bags_ok = true;
  for (const ExperimentSettings& e : plan) {
    names.push_back(e.name);
    if (e.bags.size() != 3) bags_ok = false;
    for (std::size_t b = 0; b < e.bags.size(); b++) {
      const BagSettings& bag = e.bags[b];
      bags_ok = bags_ok && bag.bag_name == e.name + "_test" + std::to_string(b + 1) && bag.runs == 5 &&
                bag.seeds == std::vector<int>({1, 2, 3, 4, 5}) && !bag.save_plots &&
                bag.param_file == "uwb_imu_ekf.yaml";
    }
  }
  const std::vector<std::string> expected_names = {"fscale3", "fscale4", "fscale5", "fscale6",
                                                   "fscale7", "fscale8", "fscale9"};
  report("experiment_setup.json: experiments in order", names == expected_names);
  report("experiment_setup.json: 3 bags x 5 runs each", bags_ok);

  const BagSettings& first = plan.front().bags.front();
  report("data folder = <data_root>/<bag>", fs::path(first.data_dir) == fs::path("data_root") / "fscale3_test1",
         first.data_dir);
  report("param file = <params_dir>/<param_file>",
         fs::path(first.config_file) == fs::path("params_dir") / "uwb_imu_ekf.yaml", first.config_file);

  // bags keep file order, also when not alphabetical
  const fs::path order_json = tmp / "order.json";
  write_file(order_json, "{\"z\": {" + bag_json("b2", "1", "[1]") + ", " + bag_json("b10", "1", "[1]") + ", " +
                             bag_json("a1", "1", "[1]") + "}, \"a\": {" + bag_json("x", "1", "[1]") + "}}");
  const auto order = flycrane::load_experiment_plan(order_json.string(), paths);
  const bool order_ok = order.size() == 2 && order[0].name == "z" && order[1].name == "a" &&
                        order[0].bags.size() == 3 && order[0].bags[0].bag_name == "b2" &&
                        order[0].bags[1].bag_name == "b10" && order[0].bags[2].bag_name == "a1";
  report("experiments and bags keep file order", order_ok);

  // integer run_seeds is a base seed
  const fs::path base_json = tmp / "base_seed.json";
  write_file(base_json, "{\"e\": {" + bag_json("b", "4", "10") + "}}");
  const auto base = flycrane::load_experiment_plan(base_json.string(), paths);
  report("integer run_seeds -> s, s+1, ...", base[0].bags[0].seeds == std::vector<int>({10, 11, 12, 13}));

  const fs::path one_json = tmp / "one_run.json";
  write_file(one_json, "{\"e\": {" + bag_json("b", "1", "[-3]") + "}}");
  report("single run, negative seed", flycrane::load_experiment_plan(one_json.string(), paths)[0].bags[0].seeds ==
                                          std::vector<int>({-3}));

  // malformed files: each error names the offending key
  struct BadCase {
    std::string name, json, expected;
  };
  const std::vector<BadCase> bad = {
      {"too few seeds", "{\"e\": {" + bag_json("b", "5", "[1,2,3,4]") + "}}", "e.b.run_seeds: expected 5 seeds"},
      {"too many seeds", "{\"e\": {" + bag_json("b", "2", "[1,2,3]") + "}}", "e.b.run_seeds: expected 2 seeds"},
      {"non-integer seed", "{\"e\": {" + bag_json("b", "2", "[1, \"a\"]") + "}}", "e.b.run_seeds[1]: expected an integer"},
      {"fractional base seed", "{\"e\": {" + bag_json("b", "2", "1.5") + "}}", "e.b.run_seeds: expected an integer"},
      {"seeds as object", "{\"e\": {" + bag_json("b", "2", "{}") + "}}", "e.b.run_seeds: expected a list"},
      {"runs = 0", "{\"e\": {" + bag_json("b", "0", "[]") + "}}", "e.b.runs: must be >= 1"},
      {"fractional runs", "{\"e\": {" + bag_json("b", "2.5", "1") + "}}", "e.b.runs: expected an integer"},
      {"save_plots not a bool", "{\"e\": {\"b\": {\"param_file\": \"p.yaml\", \"runs\": 1, \"run_seeds\": 1, "
                                "\"save_plots\": \"maybe\"}}}", "e.b.save_plots: expected true or false"},
      {"missing param_file", "{\"e\": {\"b\": {\"runs\": 1, \"run_seeds\": 1, \"save_plots\": false}}}",
       "e.b.param_file: missing"},
      {"missing run_seeds", "{\"e\": {\"b\": {\"param_file\": \"p.yaml\", \"runs\": 1, \"save_plots\": false}}}",
       "e.b.run_seeds: missing"},
      {"unknown key (typo)", "{\"e\": {" + bag_json("b", "1", "1", ", \"run_seed\": 1") + "}}",
       "e.b.run_seed: unknown key"},
      {"bag is not an object", "{\"e\": {\"b\": 5}}", "e.b: expected an object"},
      {"experiment is not an object", "{\"e\": [1, 2]}", "e: expected a non-empty object of bags"},
      {"experiment without bags", "{\"e\": {}}", "e: expected a non-empty object of bags"},
      {"top level is a list", "[1, 2]", "expected a non-empty object of experiments"},
      {"bag name with a path", "{\"e\": {" + bag_json("../b", "1", "1") + "}}", "'../b' is not a valid name"},
      {"malformed JSON", "{\"e\": {\"b\": {", "bad.json"},
  };
  for (const BadCase& c : bad) {
    const fs::path file = tmp / "bad.json";
    write_file(file, c.json);
    expect_throw(c.name, [&] { flycrane::load_experiment_plan(file.string(), paths); }, c.expected);
  }
  expect_throw("JSON file not found", [&] { flycrane::load_experiment_plan((tmp / "nope.json").string(), paths); },
               "nope.json");

  // run folder names
  BagSettings bag;
  bag.bag_name = "bag";
  bag.seeds = {4, 9};
  report("run folder run<k>_seed<s>",
         fs::path(flycrane::run_output_dir("root", "exp", bag, 1)) == fs::path("root") / "exp" / "bag" / "run2_seed9");
}

// =============================================================================
// 2. missing inputs
// =============================================================================
void check_inputs(const fs::path& repo, const fs::path& tmp) {
  std::cout << "\nmissing inputs\n";
  fs::create_directories(tmp / "data" / "present_bag");
  const fs::path json = tmp / "inputs.json";
  write_file(json, "{\"e\": {"
                   "\"present_bag\": {\"param_file\": \"uwb_imu_ekf.yaml\", \"runs\": 1, \"run_seeds\": 1, \"save_plots\": false}, "
                   "\"absent_bag\": {\"param_file\": \"uwb_imu_ekf.yaml\", \"runs\": 1, \"run_seeds\": 1, \"save_plots\": false}, "
                   "\"bad_params\": {\"param_file\": \"absent.yaml\", \"runs\": 1, \"run_seeds\": 1, \"save_plots\": false}}}");
  const ExperimentPaths paths{(tmp / "data").string(), (repo / "params").string(), (tmp / "inputs_out").string()};

  // loading only checks the format, so missing folders are fine here
  ExperimentRunner runner(json.string(), paths);
  report("plan with missing folders still loads", runner.experiments().size() == 1 && runner.total_runs() == 3);

  const std::vector<std::string> problems = flycrane::check_experiment_inputs(runner.experiments());
  const auto mentions = [&](const std::string& s) {
    for (const std::string& p : problems) {
      if (p.find(s) != std::string::npos) return true;
    }
    return false;
  };
  report("every missing input reported", problems.size() == 3, std::to_string(problems.size()) + " problem(s)");
  report("  missing data folder named", mentions("e.absent_bag: data folder not found"));
  report("  missing param file named", mentions("e.bad_params: param file not found"));
  report("  present bag not reported", !mentions("e.present_bag"));

  expect_throw("run_experiment stops before any run", [&] { runner.run_experiment(); }, "no runs were started");
  report("  no results.csv written", !fs::exists(runner.results_csv()), runner.results_csv());
}

// =============================================================================
// 3. TestConfig overrides
// =============================================================================
void check_overrides(const fs::path& params_file) {
  std::cout << "\nTestConfig overrides\n";
  flycrane::TestConfig config;
  config.config_path = params_file.string();
  const UwbImuEkfParams yaml = flycrane::load_params(params_file);
  const UwbImuEkfParams plain = flycrane::load_test_params(config);
  report("no overrides: YAML seed settings kept", plain.filter.random_offset_seed == yaml.filter.random_offset_seed &&
                                                      plain.filter.generator_seed == yaml.filter.generator_seed);
  report("no overrides: YAML save_plots kept", plain.plot.save_plots == yaml.plot.save_plots);

  config.seed = 42;
  config.save_plots = !yaml.plot.save_plots;
  const UwbImuEkfParams overridden = flycrane::load_test_params(config);
  report("seed override: fixed seed 42", !overridden.filter.random_offset_seed && overridden.filter.generator_seed == 42);
  report("save_plots override", overridden.plot.save_plots == !yaml.plot.save_plots);
}

// =============================================================================
// 4. end to end
// =============================================================================
// The test params: the repo YAML with a nonzero initial offset (so the seed
// matters), random_offset_seed on (so the experiment's seed must override it)
// and save_plots on (so the experiment's save_plots must override it).
fs::path write_test_params(const fs::path& repo, const fs::path& dir) {
  YAML::Node node = YAML::LoadFile((repo / "params" / "uwb_imu_ekf.yaml").string());
  node["filter"]["init_pose_offset"] = 0.3;
  node["filter"]["init_ang_offset"] = 5.0;
  node["filter"]["random_offset_seed"] = true;
  node["plot"]["enabled"] = true;
  node["plot"]["save_plots"] = true;
  node["plot"]["show"] = false;
  YAML::Emitter out;
  out << node;
  const fs::path file = dir / "test_params.yaml";
  write_file(file, out.c_str());
  return file;
}

// A static hover of 2 s in the format of bag_converter.py: payload at the
// origin, every drone straight above its attach point with its cable stretched
// to carry an equal share of the payload weight, so all cables are vertical.
void write_synthetic_dataset(const UwbImuEkfParams& p, const fs::path& dir) {
  constexpr int64_t kT0 = 1700000000000000000;
  constexpr int64_t kDuration = 2000000000;
  const std::size_t n = p.drones.size();
  const double tension = p.payload.mass * p.physics.gravity / static_cast<double>(n);
  const double stretch = tension / p.physics.cable_stiffness;

  std::vector<Eigen::Vector3d> pos(n);
  for (std::size_t i = 0; i < n; i++) {
    const flycrane::DroneParams& d = p.drones[i];
    pos[i] = d.attach_point_payload + Eigen::Vector3d(0, 0, d.cable_length + stretch) - d.hook_offset;
  }

  const auto num = [](double v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
  };
  const auto pose_row = [&](int64_t t, const Eigen::Vector3d& x) {
    return std::to_string(t) + "," + num(x.x()) + "," + num(x.y()) + "," + num(x.z()) + ",0,0,0,1,0,0,0,0,0,0\n";
  };
  const std::string pose_header = "t_ns,px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz\n";

  std::string payload = pose_header;
  for (int64_t t = kT0; t <= kT0 + kDuration; t += 10000000) payload += pose_row(t, Eigen::Vector3d::Zero());
  write_file(dir / "payload.csv", payload);

  for (std::size_t i = 0; i < n; i++) {
    const std::string name = p.drones[i].name;
    std::string odom = pose_header, imu = "t_ns,ax,ay,az,gx,gy,gz\n", cable = "t_ns,sx,sy,sz,tension\n";
    for (int64_t t = kT0; t <= kT0 + kDuration; t += 10000000) odom += pose_row(t, pos[i]);
    for (int64_t t = kT0; t <= kT0 + kDuration; t += 5000000) {
      imu += std::to_string(t) + ",0,0," + num(p.physics.gravity) + ",0,0,0\n";
    }
    for (int64_t t = kT0; t <= kT0 + kDuration; t += 20000000) cable += std::to_string(t) + ",0,0,-1," + num(tension) + "\n";
    write_file(dir / ("odom_" + name + ".csv"), odom);
    write_file(dir / ("imu_" + name + ".csv"), imu);
    write_file(dir / ("cable_" + name + ".csv"), cable);
  }

  std::string uwb = "t_ns,host,peer,range\n";
  for (int64_t t = kT0; t <= kT0 + kDuration; t += 50000000) {
    for (std::size_t a = 0; a < n; a++) {
      for (std::size_t b = a + 1; b < n; b++) {
        const double range = ((pos[a] + p.drones[a].uwb_antenna) - (pos[b] + p.drones[b].uwb_antenna)).norm();
        uwb += std::to_string(t) + "," + std::to_string(a) + "," + std::to_string(b) + "," + num(range) + "\n";
      }
    }
  }
  write_file(dir / "uwb.csv", uwb);

  std::string manifest = "{\n  \"drone_index_to_name\": {\n";
  for (std::size_t i = 0; i < n; i++) {
    manifest += "    \"" + std::to_string(i) + "\": \"" + p.drones[i].name + "\"" + (i + 1 < n ? "," : "") + "\n";
  }
  write_file(dir / "manifest.json", manifest + "  }\n}\n");
}

bool finite_number(const std::string& s) {
  if (s.empty()) return false;
  try {
    return std::isfinite(std::stod(s));
  } catch (const std::exception&) {
    return false;
  }
}

void check_end_to_end(const fs::path& repo, const fs::path& tmp) {
  std::cout << "\nend to end (synthetic dataset)\n";
  const fs::path params_dir = tmp / "e2e_params";
  const fs::path params_file = write_test_params(repo, params_dir);
  const UwbImuEkfParams params = flycrane::load_params(params_file);
  write_synthetic_dataset(params, tmp / "e2e_data" / "synth_bag");
  fs::create_directories(tmp / "e2e_data" / "broken_bag");  // exists, but has no csv files

  const fs::path json = tmp / "e2e.json";
  write_file(json, "{\n"
                   "  \"synthetic\": {\n"
                   "    \"broken_bag\": {\"param_file\": \"test_params.yaml\", \"runs\": 1, \"run_seeds\": 1, \"save_plots\": false},\n"
                   "    \"synth_bag\": {\"param_file\": \"test_params.yaml\", \"runs\": 3, \"run_seeds\": [7, 7, 8], \"save_plots\": false}\n"
                   "  },\n"
                   "  \"plotted\": {\n"
                   "    \"synth_bag\": {\"param_file\": \"test_params.yaml\", \"runs\": 1, \"run_seeds\": 3, \"save_plots\": true}\n"
                   "  }\n"
                   "}\n");
  const ExperimentPaths paths{(tmp / "e2e_data").string(), params_dir.string(), (tmp / "e2e_results").string()};
  ExperimentRunner runner(json.string(), paths);
  const int failed = runner.run_experiment();
  std::cout << "\n";
  report("one failed run (the broken bag)", failed == 1, std::to_string(failed) + " failed");
  report("results in <output_dir>/<json name>/results.csv",
         fs::path(runner.results_csv()) == tmp / "e2e_results" / "e2e" / "results.csv");

  const Csv csv = read_csv(runner.results_csv());
  report("one row per run", csv.rows.size() == 5, std::to_string(csv.rows.size()) + " rows");
  bool widths_ok = true;
  for (const auto& row : csv.rows) widths_ok = widths_ok && row.size() == csv.header.size();
  report("every row as wide as the header", widths_ok, std::to_string(csv.header.size()) + " columns");
  if (csv.rows.size() != 5 || !widths_ok) return;

  // columns: the requested metrics for every drone and the payload
  std::vector<std::string> metric_cols;
  for (const std::string& body : {"Drone1", "Drone2", "Drone3", "Payload"}) {
    for (const std::string& m : {"_pos_rms_m", "_vel_rms_mps", "_att_rms_deg"}) metric_cols.push_back(body + m);
  }
  for (const std::string& drone : {"Drone1", "Drone2", "Drone3"}) {
    for (const std::string& m : {"_splay_mean_deg", "_splay_min_deg", "_splay_below_tol_pct",
                                 "_thrust_cable_mean_deg", "_thrust_cable_min_deg"}) {
      metric_cols.push_back(drone + m);
    }
  }
  bool columns_ok = true;
  for (const std::string& c : metric_cols) columns_ok = columns_ok && csv.col(c) >= 0;
  report("pos/vel/att RMS + cable columns present", columns_ok);
  if (!columns_ok) return;

  // order and identification
  const std::vector<std::string> expected_ids = {"synthetic/broken_bag/1/1", "synthetic/synth_bag/1/7",
                                                 "synthetic/synth_bag/2/7", "synthetic/synth_bag/3/8",
                                                 "plotted/synth_bag/1/3"};
  bool ids_ok = true;
  for (std::size_t r = 0; r < 5; r++) {
    ids_ok = ids_ok && csv.at(r, "experiment") + "/" + csv.at(r, "bag") + "/" + csv.at(r, "run") + "/" +
                               csv.at(r, "seed") == expected_ids[r];
  }
  report("rows in file order, with run and seed", ids_ok);
  report("drone names recorded", csv.at(1, "drone_names") == "falcon1;falcon2;falcon3", csv.at(1, "drone_names"));

  // the failed run
  bool failed_empty = true;
  for (const std::string& c : metric_cols) failed_empty = failed_empty && csv.at(0, c).empty();
  report("broken bag: status failed with its error",
         csv.at(0, "status") == "failed" && csv.at(0, "error").find("required file is missing") != std::string::npos,
         csv.at(0, "error"));
  report("broken bag: metrics left empty", failed_empty);

  // the good runs
  bool good_ok = true;
  for (std::size_t r = 1; r < 5; r++) {
    good_ok = good_ok && csv.at(r, "status") == "ok" && csv.at(r, "error").empty();
    for (const std::string& c : metric_cols) good_ok = good_ok && finite_number(csv.at(r, c));
  }
  report("runs after the failure: ok, every metric set", good_ok);

  bool same_seed_same = true;
  bool other_seed_differs = false;
  for (const std::string& c : metric_cols) {
    same_seed_same = same_seed_same && csv.at(1, c) == csv.at(2, c);
    if (csv.at(1, c) != csv.at(3, c)) other_seed_differs = true;
  }
  report("same seed -> identical metrics", same_seed_same);
  report("different seed -> different metrics", other_seed_differs,
         "Drone1_pos_rms_m " + csv.at(1, "Drone1_pos_rms_m") + " vs " + csv.at(3, "Drone1_pos_rms_m"));

  // the synthetic cables hang straight down from upright drones
  bool vertical = true;
  for (const std::string& drone : {"Drone1", "Drone2", "Drone3"}) {
    vertical = vertical && std::stod(csv.at(1, drone + "_splay_mean_deg")) < 0.1 &&
               std::stod(csv.at(1, drone + "_thrust_cable_mean_deg")) < 0.1 &&
               std::stod(csv.at(1, drone + "_splay_below_tol_pct")) == 100.0;
  }
  report("vertical cables: splay ~0, all near-vertical", vertical,
         "Drone1 splay " + csv.at(1, "Drone1_splay_mean_deg") + " deg");

  // save_plots from the JSON, not the YAML
  const fs::path root = runner.output_root();
  report("save_plots false: no plot folder", !fs::exists(root / "synthetic" / "synth_bag" / "run1_seed7"));
  report("save_plots true: plots in run1_seed3/",
         fs::exists(root / "plotted" / "synth_bag" / "run1_seed3" / "Drone1" / "overall_error.png") &&
             fs::exists(root / "plotted" / "synth_bag" / "run1_seed3" / "Payload" / "position.png"));
}

}  // namespace

int main(int argc, char** argv) {
  const fs::path repo = argc > 1 ? argv[1] : ".";
  const fs::path tmp = fs::temp_directory_path() / ("flycrane_experiment_test_" + std::to_string(std::random_device{}()));
  fs::create_directories(tmp);
  std::cout << "repo: " << repo.string() << "\ntemp: " << tmp.string() << "\n";

  try {
    check_parsing(repo, tmp);
    check_inputs(repo, tmp);
    check_overrides(write_test_params(repo, tmp / "override_params"));
    check_end_to_end(repo, tmp);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }

  std::cout << '\n' << (g_failures == 0 ? "ALL CHECKS PASSED" : std::to_string(g_failures) + " CHECK(S) FAILED") << '\n';
  if (g_failures == 0) {
    fs::remove_all(tmp);
  } else {
    std::cout << "test files kept in " << tmp.string() << '\n';
  }
  return g_failures == 0 ? 0 : 1;
}
