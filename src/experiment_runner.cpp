/*
experiment_runner.cpp

Loads in an experiment JSON file and runs the experiments with the designated
settings (see experiment_runner.hpp for the JSON layout and the outputs).

The JSON is read with yaml-cpp: every JSON document is also valid YAML, and
yaml-cpp keeps the keys of an object in file order.
*/

#include "flycrane_ekf/experiment_runner.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "flycrane_ekf/params.hpp"
#include "flycrane_ekf/plotter.hpp"

namespace flycrane {

namespace fs = std::filesystem;

namespace {

// Keys for JSON
const std::set<std::string> kBagKeys = {"param_file", "runs", "run_seeds", "save_plots"};

// A node of the JSON and its key path, for error messages (as Section in params.cpp).
struct JsonNode {
    YAML::Node node;
    std::string file;
    std::string path;

    std::string key_path(const std::string& key) const { return path.empty() ? key : path + "." + key; }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error(file + ": " + (path.empty() ? "" : path + ": ") + what);
    }

    JsonNode get(const std::string& key) const {
        const YAML::Node n = node[key];
        if (!n) JsonNode{node, file, key_path(key)}.fail("missing");
        return {n, file, key_path(key)};
    }

    int integer() const {
        try {
            if (node.IsScalar()) return node.as<int>();
        } catch (const YAML::Exception&) {
        }
        fail("expected an integer");
    }

    bool flag() const {
        try {
            if (node.IsScalar()) return node.as<bool>();
        } catch (const YAML::Exception&) {
        }
        fail("expected true or false");
    }

    std::string text() const {
        if (!node.IsScalar() || node.as<std::string>().empty()) fail("expected a non-empty string");
        return node.as<std::string>();
    }
};

// Experiment and bag names become folder names.
void check_name(const JsonNode& parent, const std::string& name) {
    if (name.empty() || name == "." || name == ".." || name.find_first_of("/\\") != std::string::npos) {
        parent.fail("'" + name + "' is not a valid name (it is used as a folder name)");
    }
}

BagSettings parse_bag(const JsonNode& bag_node, const std::string& bag_name, const ExperimentPaths& paths) {
    if (!bag_node.node.IsMap()) bag_node.fail("expected an object with " + std::string("param_file, runs, run_seeds, save_plots"));
    for (const auto& kv : bag_node.node) {
        const std::string key = kv.first.as<std::string>();
        if (kBagKeys.count(key) == 0) JsonNode{kv.second, bag_node.file, bag_node.key_path(key)}.fail("unknown key");
    }

    BagSettings bag;
    bag.bag_name = bag_name;
    bag.data_dir = (fs::path(paths.data_root) / bag_name).string();
    bag.param_file = bag_node.get("param_file").text();
    bag.config_file = (fs::path(paths.params_dir) / bag.param_file).string();
    bag.save_plots = bag_node.get("save_plots").flag();

    const JsonNode runs = bag_node.get("runs");
    bag.runs = runs.integer();
    if (bag.runs < 1) runs.fail("must be >= 1");

    // A list with one seed per run, or one integer: the base seed of run1 incremented for every following run.
    const JsonNode seeds = bag_node.get("run_seeds");
    if (seeds.node.IsSequence()) {
        if (seeds.node.size() != static_cast<std::size_t>(bag.runs)) {
            seeds.fail("expected " + std::to_string(bag.runs) + " seeds (one per run), got " +
                       std::to_string(seeds.node.size()));
        }
        for (std::size_t i = 0; i < seeds.node.size(); i++) {
            bag.seeds.push_back(JsonNode{seeds.node[i], seeds.file, seeds.path + "[" + std::to_string(i) + "]"}.integer());
        }
    } else if (seeds.node.IsScalar()) {
        const int base = seeds.integer();
        for (int k = 0; k < bag.runs; k++) bag.seeds.push_back(base + k);
    } else {
        seeds.fail("expected a list of seeds (one per run) or one integer base seed");
    }
    return bag;
}

// Write the Results CSV (Pretty much all data)
std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + "\"";
}

// Empty for NaN (no ground truth), so the column stays numeric in pandas etc.
std::string csv_number(double v) {
    if (!std::isfinite(v)) return "";
    std::ostringstream os;
    os.precision(9);
    os << v;
    return os.str();
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); i++) s += (i ? sep : "") + v[i];
    return s;
}

// One row per run 
class ResultsCsv {
    public:
        ResultsCsv(const std::string& file, std::size_t n_drones) : n_drones_(n_drones) {
            fs::create_directories(fs::path(file).parent_path());
            out_.open(file, std::ios::trunc);
            if (!out_) throw std::runtime_error("cannot write the results CSV: " + file);

            std::vector<std::string> header = {"experiment", "bag", "run", "seed", "param_file", "drone_names",
                                               "status", "error"};
            std::vector<std::string> bodies;
            for (std::size_t i = 0; i < n_drones_; i++) bodies.push_back("Drone" + std::to_string(i + 1));
            for (const std::string& b : bodies) add_body_columns(header, b);
            add_body_columns(header, "Payload");
            header.push_back("splay_vertical_tol_deg");
            for (const std::string& b : bodies) {
                for (const char* c : {"_splay_mean_deg", "_splay_min_deg", "_splay_below_tol_pct",
                                      "_thrust_cable_mean_deg", "_thrust_cable_min_deg"}) {
                    header.push_back(b + c);
                }
            }
            n_columns_ = header.size();
            write(header);
        }

        // metrics is null for a failed run
        void add_run(const std::string& experiment, const BagSettings& bag, std::size_t run,
                     const UwbImuEkfParams& params, const RunMetrics* metrics, const std::string& error) {
            std::vector<std::string> row = {experiment, bag.bag_name, std::to_string(run + 1),
                                            std::to_string(bag.seeds[run]), bag.param_file,
                                            join(params.drone_names, ";"),
                                            metrics ? "ok" : "failed", error};
            // drones, padded to n_drones_, then the payload
            for (std::size_t i = 0; i < n_drones_; i++) {
                const BodyMetrics* b = (metrics && i + 1 < metrics->bodies.size()) ? &metrics->bodies[i] : nullptr;
                add_body_values(row, b);
            }
            add_body_values(row, metrics ? &metrics->bodies.back() : nullptr);
            row.push_back(csv_number(params.diagnostics.splay_vertical_tol));
            for (std::size_t i = 0; i < n_drones_; i++) {
                const CableMetrics* c = (metrics && i < metrics->cables.size()) ? &metrics->cables[i] : nullptr;
                for (double v : {c ? c->splay_mean_deg : NAN, c ? c->splay_min_deg : NAN,
                                 c ? c->splay_below_tol_pct : NAN, c ? c->thrust_cable_mean_deg : NAN,
                                 c ? c->thrust_cable_min_deg : NAN}) {
                    row.push_back(csv_number(v));
                }
            }
            if (row.size() != n_columns_) throw std::logic_error("ResultsCsv: row and header sizes differ");
            write(row);
        }

    private:
        static void add_body_columns(std::vector<std::string>& header, const std::string& body) {
            header.push_back(body + "_pos_rms_m");
            header.push_back(body + "_vel_rms_mps");
            header.push_back(body + "_att_rms_deg");
        }

        static void add_body_values(std::vector<std::string>& row, const BodyMetrics* b) {
            row.push_back(csv_number(b ? b->pos_rms_m : NAN));
            row.push_back(csv_number(b ? b->vel_rms_mps : NAN));
            row.push_back(csv_number(b ? b->att_rms_deg : NAN));
        }

        // Flushed per row, so the finished runs are kept if a long batch is stopped.
        void write(const std::vector<std::string>& fields) {
            for (std::size_t i = 0; i < fields.size(); i++) out_ << (i ? "," : "") << csv_field(fields[i]);
            out_ << "\n";
            out_.flush();
        }

        std::ofstream out_;
        std::size_t n_drones_;
        std::size_t n_columns_ = 0;
};

}  // namespace

// helper functions

// load all of the experiments from the file. 
std::vector<ExperimentSettings> load_experiment_plan(const std::string& experiment_json, const ExperimentPaths& paths) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(experiment_json);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(experiment_json + ": " + e.what());
    }
    const JsonNode top{root, experiment_json, ""};
    if (!root.IsMap() || root.size() == 0) top.fail("expected a non-empty object of experiments");

    std::vector<ExperimentSettings> experiments;
    for (const auto& exp_kv : root) {
        ExperimentSettings experiment;
        experiment.name = exp_kv.first.as<std::string>();
        check_name(top, experiment.name);
        const JsonNode exp_node{exp_kv.second, experiment_json, experiment.name};
        if (!exp_node.node.IsMap() || exp_node.node.size() == 0) exp_node.fail("expected a non-empty object of bags");

        for (const auto& bag_kv : exp_node.node) {
            const std::string bag_name = bag_kv.first.as<std::string>();
            check_name(exp_node, bag_name);
            experiment.bags.push_back(parse_bag({bag_kv.second, experiment_json, exp_node.key_path(bag_name)},
                                                bag_name, paths));
        }
        experiments.push_back(std::move(experiment));
    }
    return experiments;
}

// double check that all the csv paths are valid
std::vector<std::string> check_experiment_inputs(const std::vector<ExperimentSettings>& experiments) {
    std::vector<std::string> problems;
    std::map<std::string, std::string> param_errors;  // param file -> load error ("" if it loads)
    for (const ExperimentSettings& experiment : experiments) {
        for (const BagSettings& bag : experiment.bags) {
            const std::string where = experiment.name + "." + bag.bag_name + ": ";
            if (!fs::is_directory(bag.data_dir)) problems.push_back(where + "data folder not found: " + bag.data_dir);

            if (!fs::is_regular_file(bag.config_file)) {
                problems.push_back(where + "param file not found: " + bag.config_file);
                continue;
            }
            if (param_errors.count(bag.config_file) == 0) {
                try {
                    load_params(bag.config_file);
                    param_errors[bag.config_file] = "";
                } catch (const std::exception& e) {
                    param_errors[bag.config_file] = e.what();
                }
            }
            if (!param_errors[bag.config_file].empty()) problems.push_back(where + param_errors[bag.config_file]);
        }
    }
    return problems;
}

std::string run_output_dir(const std::string& output_root, const std::string& experiment, const BagSettings& bag,
                           std::size_t run) {
    const std::string run_folder = "run" + std::to_string(run + 1) + "_seed" + std::to_string(bag.seeds.at(run));
    return (fs::path(output_root) / experiment / bag.bag_name / run_folder).string();
}

// Now define the actual experimentrunner
ExperimentRunner::ExperimentRunner(std::string experiment_json, ExperimentPaths paths)
    : experiment_json_(std::move(experiment_json)),
      paths_(std::move(paths)),
      output_root_((fs::path(paths_.output_dir) / fs::path(experiment_json_).stem()).string()),
      experiments_(readExperimentJson()) {}

std::vector<ExperimentSettings> ExperimentRunner::readExperimentJson() const {
    return load_experiment_plan(experiment_json_, paths_);
}

std::size_t ExperimentRunner::total_runs() const {
    std::size_t n = 0;
    for (const ExperimentSettings& experiment : experiments_) {
        for (const BagSettings& bag : experiment.bags) n += static_cast<std::size_t>(bag.runs);
    }
    return n;
}

std::string ExperimentRunner::results_csv() const {
    return (fs::path(output_root_) / "results.csv").string();
}

void ExperimentRunner::print_plan(std::ostream& os) const {
    os << "experiment plan: " << experiment_json_ << "\n";
    for (const ExperimentSettings& experiment : experiments_) {
        os << "  " << experiment.name << "\n";
        for (const BagSettings& bag : experiment.bags) {
            std::vector<std::string> seeds;
            for (int s : bag.seeds) seeds.push_back(std::to_string(s));
            os << "    " << bag.bag_name << ": " << bag.runs << " run(s), seeds [" << join(seeds, ", ") << "], params "
               << bag.config_file << ", plots " << (bag.save_plots ? "saved" : "not saved")
               << (fs::is_directory(bag.data_dir) ? "" : "   [data folder missing: " + bag.data_dir + "]") << "\n";
        }
    }
    os << total_runs() << " run(s) in total; results -> " << results_csv() << "\n";
}

int ExperimentRunner::run_experiment() {
    // Throw file errors before any run starts
    const std::vector<std::string> problems = check_experiment_inputs(experiments_);
    if (!problems.empty()) {
        throw std::runtime_error("experiment inputs missing, no runs were started:\n  " + join(problems, "\n  "));
    }

    // Every param file loads
    std::map<std::string, UwbImuEkfParams> bag_params;
    std::size_t max_drones = 0;
    for (const ExperimentSettings& experiment : experiments_) {
        for (const BagSettings& bag : experiment.bags) {
            if (bag_params.count(bag.config_file) == 0) bag_params[bag.config_file] = load_params(bag.config_file);
            max_drones = std::max(max_drones, bag_params[bag.config_file].drones.size());
        }
    }
    ResultsCsv csv(results_csv(), max_drones);

    const std::size_t n_total = total_runs();
    std::size_t n_done = 0;
    int n_failed = 0;
    for (const ExperimentSettings& experiment : experiments_) {
        for (const BagSettings& bag : experiment.bags) {
            for (std::size_t run = 0; run < static_cast<std::size_t>(bag.runs); run++) {
                n_done++;
                std::cout << "\n==== [" << n_done << "/" << n_total << "] " << experiment.name << " / " << bag.bag_name
                          << " / run " << run + 1 << " (seed " << bag.seeds[run] << ") ====\n";
                const UwbImuEkfParams& params = bag_params.at(bag.config_file);
                try {
                    const RunMetrics metrics = run_single(experiment.name, bag, run);
                    csv.add_run(experiment.name, bag, run, params, &metrics, "");
                } catch (const std::exception& e) {
                    n_failed++;
                    std::cerr << "run failed: " << e.what() << "\n";
                    csv.add_run(experiment.name, bag, run, params, nullptr, e.what());
                }
            }
        }
    }

    std::cout << "\nexperiment finished: " << n_total - static_cast<std::size_t>(n_failed) << "/" << n_total
              << " runs ok, results in " << results_csv() << "\n";
    return n_failed;
}

RunMetrics ExperimentRunner::run_single(const std::string& experiment, const BagSettings& bag, std::size_t run) const {
    const std::string output_dir = run_output_dir(output_root_, experiment, bag, run);

    TestConfig config;
    config.data_path = bag.data_dir;
    config.config_path = bag.config_file;
    config.output_dir = output_dir;
    config.seed = bag.seeds[run];
    config.save_plots = bag.save_plots;

    // A fresh runner per run: a TestRunner accumulates its log and can only run once.
    TestRunner test_runner(config);
    const std::vector<LogEntry> log = test_runner.run_test();

    if (test_runner.params().plot.enabled) {
        // Never open plot windows during a batch: plt::show() would block every run.
        UwbImuEkfParams plot_params = test_runner.params();
        plot_params.plot.show = false;
        TestPlotter plotter(plot_params, output_dir);
        plotter.plot_test(log);
    }
    return compute_run_metrics(test_runner.params(), log);
}

} // flycrane namespace
