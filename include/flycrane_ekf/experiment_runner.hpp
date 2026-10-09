#pragma once
/*
experiment_runner.hpp

Loads in an experiment JSON file and runs the experiments with the designated settings. 
For an example, 
*/

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "flycrane_ekf/run_ekf_test.hpp"
#include "flycrane_ekf/run_metrics.hpp"

namespace flycrane {

// One bag of an experiment and how to run it.
struct BagSettings {
    std::string bag_name;      // key in the JSON, e.g. "fscale3_test1"
    std::string data_dir;      // <data_root>/<bag_name>
    std::string param_file;    // as written in the JSON
    std::string config_file;   // <params_dir>/<param_file>
    int runs = 0;
    std::vector<int> seeds;    // one per run
    bool save_plots = false;
};

// One value of the tested variable (e.g. "fscale3") and its bags.
struct ExperimentSettings {
    std::string name;
    std::vector<BagSettings> bags;
};

// Where the inputs are looked up and the outputs are written.
struct ExperimentPaths {
    std::string data_root = "test_data/csv";
    std::string params_dir = "params";
    std::string output_dir = "results";
};

// Parse and validate the experiment JSON
std::vector<ExperimentSettings> load_experiment_plan(const std::string& experiment_json, const ExperimentPaths& paths);

// Ensure all csv folders are valid(return empty vector if no errors)
std::vector<std::string> check_experiment_inputs(const std::vector<ExperimentSettings>& experiments);

// Plot folder for all experiments
std::string run_output_dir(const std::string& output_root, const std::string& experiment, const BagSettings& bag,
                           std::size_t run);

class ExperimentRunner {
    public:
        // Loads and validates the experiment JSON (throws on a malformed file).
        explicit ExperimentRunner(std::string experiment_json, ExperimentPaths paths = {});

        // Print every bag and its runs first
        void print_plan(std::ostream& os) const;

        // Run the actual experiment
        int run_experiment();

        const std::vector<ExperimentSettings>& experiments() const { return experiments_; }
        std::size_t total_runs() const;
        // <output_dir>/<json file name without extension>
        const std::string& output_root() const { return output_root_; }
        std::string results_csv() const;

    private:
        std::vector<ExperimentSettings> readExperimentJson() const;
        // Run one run of a bag (0-based) and return its metrics.  Throws on failure.
        RunMetrics run_single(const std::string& experiment, const BagSettings& bag, std::size_t run) const;

        std::string experiment_json_;
        ExperimentPaths paths_;
        std::string output_root_;
        std::vector<ExperimentSettings> experiments_;
};

} // flycrane namespace
