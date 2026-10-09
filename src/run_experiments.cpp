/*
run_experiments.cpp

Runs every run of an experiment JSON (see experiment_runner.hpp) and writes
the results CSV.

    run_experiments <experiment.json> [--data-root DIR] [--params-dir DIR] [--output-dir DIR] [--dry-run]

    --data-root    folder with the csv folders of the bags   (default test_data/csv)
    --params-dir   folder with the param files               (default params)
    --output-dir   results go to <output-dir>/<json name>/   (default results)
    --dry-run      only print the plan and any missing inputs
*/

#include "flycrane_ekf/experiment_runner.hpp"

#include <exception>
#include <iostream>
#include <string>

using namespace flycrane;

namespace {

const char* kUsage =
    "usage: run_experiments <experiment.json> [--data-root DIR] [--params-dir DIR] [--output-dir DIR] [--dry-run]";

struct Args {
    std::string experiment_json;
    ExperimentPaths paths;
    bool dry_run = false;
    bool success = true;
};

Args handle_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--dry-run") {
            args.dry_run = true;
        } else if (arg == "--data-root" && has_value) {
            args.paths.data_root = argv[++i];
        } else if (arg == "--params-dir" && has_value) {
            args.paths.params_dir = argv[++i];
        } else if (arg == "--output-dir" && has_value) {
            args.paths.output_dir = argv[++i];
        } else if (arg.rfind("--", 0) != 0 && args.experiment_json.empty()) {
            args.experiment_json = arg;
        } else {
            std::cout << "Unexpected argument: " << arg << "\n";
            args.success = false;
        }
    }
    if (args.experiment_json.empty()) {
        std::cout << "No experiment JSON specified!\n";
        args.success = false;
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = handle_args(argc, argv);
    if (!args.success) {
        std::cout << kUsage << "\n";
        return 1;
    }

    try {
        ExperimentRunner runner(args.experiment_json, args.paths);
        runner.print_plan(std::cout);

        if (args.dry_run) {
            const std::vector<std::string> problems = check_experiment_inputs(runner.experiments());
            for (const std::string& p : problems) std::cout << "missing: " << p << "\n";
            return problems.empty() ? 0 : 1;
        }
        return runner.run_experiment() == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
