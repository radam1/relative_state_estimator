#pragma once
// Test of the EKF in this repository: replays a dataset converted by
// bag_converter.py through the EKF and logs the estimate against ground truth
// then plots the results.
//
// TestRunner is implemented in src/test_runner.cpp so that both run_ekf_test
// (one dataset) and run_experiments (experiment_runner.hpp) can use it.

#include <iostream>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <random>

#include <Eigen/Core>

#include "flycrane_ekf/ekf.hpp"
#include "flycrane_ekf/params.hpp"
#include "flycrane_ekf/plotter.hpp"
#include "flycrane_ekf/csv_io.hpp"


namespace flycrane {

// Configuration struct for loading flycrane tests
struct TestConfig {
    std::string data_path;
    std::string config_path = "params/uwb_imu_ekf.yaml";
    std::string output_dir = "results";   // plots go to <output_dir>/<data folder name>/
    bool success = true;

    // Optional overrides of the params YAML (used by the experiment runner).
    // Left empty, the values in the YAML are used.
    std::optional<int> seed;          // sets random_offset_seed = false, generator_seed = seed
    std::optional<bool> save_plots;   // sets plot.save_plots
};

// Load the params YAML of a test and apply the overrides in the TestConfig.
UwbImuEkfParams load_test_params(const TestConfig& config);

// Accepted and rejected updates of one sensor, and the sum of their NIS.
struct UpdateStats {
    int accepted = 0;
    int rejected = 0;
    double nis_sum = 0.0;
};

class TestRunner {
    public:
        // Loads the params and the dataset.
        explicit TestRunner(const TestConfig& config);

        // Initialise the filter, replay the whole timeline, and return one
        // log entry per 1/log_rate_hz.
        std::vector<LogEntry> run_test();

        const UwbImuEkfParams& params() const { return params_; }

    private:
        std::string dataset_dir_;
        UwbImuEkfParams params_;  

        UwbImuEkf ekf_;

        io::LoadOptions load_options_;
        io::Dataset dataset_;
        size_t timeline_length_ = 0;

        std::vector<LogEntry> ekf_timeline_;

        UpdateStats uwb_stats_;
        UpdateStats cable_stats_;
        UpdateStats mocap_stats_;

        // For adding random offset
        std::normal_distribution<double> n_; 
        std::mt19937_64 rng_; 
        double init_pose_offset_; 
        double init_ang_offset_; 
        PoseSample add_random_offset_(const PoseSample& s); 

        // Other helper functions
        void step_ekf_for_event(const io::Event& event);
        void initialize_filter();
        PoseSample get_payload_pose(int64_t t_ns) const;
        LogEntry make_log_entry() const;
        void record_update(UpdateStats& stats, bool accepted);
        void print_summary() const;
};

} //namespace flycrane
