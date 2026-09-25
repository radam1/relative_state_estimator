/*
run_ekf_test.cpp

This file will use the EKF implementation in ekf.cpp to test the state estimator
on example rosbag data, converted into csv using bag_converter.py and read into
the cpp code using csv_io.cpp.

    run_ekf_test <csv_dir> [params.yaml]
*/

// includes
#include "flycrane_ekf/run_ekf_test.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <random>

#include <Eigen/Dense>

// print a line to the console
void print(const std::string& msg){
    std::cout<<msg<<"\n";
}

// define the TestRunner class
namespace flycrane {

TestRunner::TestRunner(const TestConfig& config)
    : dataset_dir_(config.data_path),
      params_(load_params(config.config_path)),
      ekf_(params_),
      n_(0.0, 1.0), 
      rng_(std::random_device{}()),
      init_pose_offset_(params_.filter.init_pose_offset),
      init_ang_offset_(params_.filter.init_ang_offset) {
            // configure dataset load options
            load_options_.dir = dataset_dir_;
            load_options_.drone_names = params_.drone_names;
            load_options_.use_motors = false;    // the IMU-input model does not use motor speeds
            load_options_.use_cable = params_.filter.use_cable;
            load_options_.use_uwb = params_.filter.use_uwb;
            if (params_.filter.use_mocap){
                // the anchor drone's odom is also put on the timeline as Sensor::Mocap events
                load_options_.mocap_drone = params_.filter.anchor_drone;
                load_options_.mocap_rate_hz = params_.filter.mocap_rate_hz;
            }

            // Load in dataset
            dataset_ = io::load_dataset(load_options_);
            io::print_report(dataset_, std::cout);
            timeline_length_ = dataset_.timeline.size();

            // Create State Estimate Timeline: one entry per log period
            const double duration_s = dataset_.rel_s(dataset_.t_last_ns);
            ekf_timeline_.reserve(static_cast<size_t>(duration_s * params_.filter.log_rate_hz) + 1);
}

std::vector<LogEntry> TestRunner::run_test(){
    initialize_filter();

    // iterate through the timeline and run the ekf for each event
    const int64_t log_period_ns = static_cast<int64_t>(std::llround(1e9 / params_.filter.log_rate_hz));
    int64_t next_log_ns = dataset_.t_first_ns;

    for (size_t i=0; i<timeline_length_; i++){
        const io::Event& timeline_event_i = dataset_.timeline[i];
        step_ekf_for_event(timeline_event_i);

        // log at most once per log period
        if (timeline_event_i.t_ns >= next_log_ns){
            ekf_timeline_.push_back(make_log_entry());
            next_log_ns += log_period_ns;
            // after a gap in the data, restart the log grid from here
            if (next_log_ns <= timeline_event_i.t_ns) next_log_ns = timeline_event_i.t_ns + log_period_ns;
        }
    }

    print_summary();
    return ekf_timeline_;
}

// Start the filter at the beginning of the timeline
void TestRunner::initialize_filter(){
    const int64_t t0 = dataset_.t_first_ns;
    const size_t n_drones = dataset_.drones.size();

    std::vector<PoseSample> drone_poses(n_drones);
    for (size_t i=0; i<n_drones; i++){
        const std::optional<PoseSample> truth = io::interpolate(dataset_.ground_truth(i), t0);
        if (!truth){
            throw std::runtime_error("initialize_filter: no ground truth for " + dataset_.drones[i].name +
                                     " at the start of the timeline");
        }
        drone_poses[i] = add_random_offset_(*truth);
    }
    // need to make sure that the initial payload pose can be calculated
    const PoseSample payload_pose = get_payload_pose(t0);
    ekf_.initialize(t0, add_random_offset_(payload_pose), drone_poses);

    print("\nfilter initialised at t = 0 s, payload at [" + std::to_string(payload_pose.p.x()) + ", " +
          std::to_string(payload_pose.p.y()) + ", " + std::to_string(payload_pose.p.z()) + "] m");
}

// Adds yaml-specified random offset to a PoseSample class
PoseSample TestRunner::add_random_offset_(const PoseSample& s) {
  // copy the original PoseSample 
  PoseSample s_transformed = s; 

  // generate random unit vectors for pose and ang 
  Eigen::Vector3d v_pose;
  do { v_pose << n_(rng_), n_(rng_), n_(rng_); } while (v_pose.norm() < 1e-12);
  v_pose = v_pose.normalized();

  Eigen::Vector3d v_ang;
  do { v_ang << n_(rng_), n_(rng_), n_(rng_); } while (v_ang.norm() < 1e-12);
  v_ang = v_ang.normalized();

  // Translation: exactly init_pose_offset_ meters in random direction
  s_transformed.p += init_pose_offset_ * v_pose;

  // Rotation: exactly init_ang_offset_ degrees about random axis
  const double ang_rad = init_ang_offset_ * M_PI / 180.0;
  const Eigen::Quaterniond dq(Eigen::AngleAxisd(ang_rad, v_ang));
  s_transformed.q = (dq * s.q).normalized();

  return s_transformed;
}

// Payload ground truth from csv at t_ns.
PoseSample TestRunner::get_payload_pose(int64_t t_ns) const {
    const std::optional<PoseSample> truth = io::interpolate(dataset_.payload, t_ns);
    if (!truth){
        throw std::runtime_error("get_payload_pose: no payload ground truth (payload.csv) at t = " +
                                 std::to_string(dataset_.rel_s(t_ns)) + " s");
    }
    return *truth;
}

LogEntry TestRunner::make_log_entry() const {
    EkfSnapshot snapshot = ekf_.report_current_readings();

    // ground truth of every drone at the time of the estimate
    const size_t n_drones = dataset_.drones.size();
    std::vector<PoseSample> ground_truth(n_drones);
    bool truth_valid = true;
    for (size_t i=0; i<n_drones; i++){
        const std::optional<PoseSample> truth = io::interpolate(dataset_.ground_truth(i), snapshot.t_ns);
        if (truth) ground_truth[i] = *truth;
        else truth_valid = false;
    }

    // ground truth of the payload at the time of the estimate
    const std::optional<PoseSample> payload_truth = io::interpolate(dataset_.payload, snapshot.t_ns);

    LogEntry entry;
    entry.t_s = dataset_.rel_s(snapshot.t_ns);
    entry.estimate = snapshot;
    entry.drone_truth = ground_truth;
    entry.drone_truth_valid = truth_valid;
    if (payload_truth) entry.payload_truth = *payload_truth;
    entry.payload_truth_valid = payload_truth.has_value();
    return entry;
}

void TestRunner::step_ekf_for_event(const io::Event& event){
    Sensor type = event.type;
    switch (type) {
        case Sensor::Imu: {
            // for imu measurements, only run the prediction
            ImuSample imu_meas = dataset_.drones[event.drone].imu[event.idx];
            ekf_.set_imu(event.drone, imu_meas);
            break;
        }

        case Sensor::Motors:
            // for motor measurements, don't run anything
            break;

        case Sensor::Cable: {
            // for Cable measurements, run prediction and update
            CableSample cable_meas = dataset_.drones[event.drone].cable[event.idx];
            record_update(cable_stats_, ekf_.update_cable(event.drone, cable_meas));
            break;
        }

        case Sensor::Uwb: {
            // for UWB measurements, run prediction and update
            UwbRange uwb_meas = dataset_.uwb[event.idx];
            record_update(uwb_stats_, ekf_.update_uwb(uwb_meas));
            break;
        }

        case Sensor::Mocap: {
            // only handle MoCap for the anchor drone
            if (event.drone == params_.filter.anchor_drone){
                PoseSample mocap_measurement = dataset_.drones[event.drone].odom[event.idx];
                record_update(mocap_stats_, ekf_.update_mocap(mocap_measurement));
            }
            break;
        }

        default:
            // Something has gone wrong, so throw an error and stop the application.
            throw std::logic_error("There is some error in the step_ekf_for_event() method: unknown sensor type");
    }
}

// Count an update and, if it was applied, its NIS.
void TestRunner::record_update(UpdateStats& stats, bool accepted){
    if (accepted){
        stats.accepted++;
        stats.nis_sum += ekf_.last_nis();
    } else {
        stats.rejected++;
    }
}

// Print the summary for a full run
void TestRunner::print_summary() const {
    std::cout << "\nupdates      accepted  rejected  mean NIS  (expect)\n";
    const auto row = [](const std::string& name, const UpdateStats& s, const std::string& expect){
        const double mean_nis = s.accepted > 0 ? s.nis_sum / s.accepted : 0.0;
        std::cout << "  " << std::left << std::setw(10) << name << std::right << std::setw(9) << s.accepted
                  << std::setw(10) << s.rejected << std::setw(10) << std::fixed << std::setprecision(2) << mean_nis
                  << "  (" << expect << ")\n";
    };
    row("uwb", uwb_stats_, "1");
    row("cable", cable_stats_, "2");
    row("mocap", mocap_stats_, "6");
    std::cout << "log: " << ekf_timeline_.size() << " entries\n";
}

}//flycrane namespace

// now use the flycrane namespace from previous files
using namespace flycrane;

// Test the configuration
TestConfig handle_args(int argc, char** argv){
    TestConfig config;
    // parse inputs
    switch (argc){
        case 1:
            // warn the user that they have not put in the right arguments
            print("No data directory or yaml config specified!");
            print("usage: run_ekf_test <csv_dir> [params.yaml]");
            // exit the code
            config.success = false;
            break;
        case 2: {
            // check if argument is .yaml or directory
            const std::string arg = argv[1];
            if (arg.find(".yaml") != std::string::npos){
                print("Only argument is a YAML file. Please add bag path");
                config.success = false;
            } else{
                print("Trying to find data directory: "+arg+"\nProceeding with default config file");
                config.data_path = arg;
            }
            break;
        }
        case 3:
            // for now, assume that arg1 is the data and arg2 is the yaml file.
            config.data_path = argv[1];
            config.config_path= argv[2];
            break;
        default:
            // warn the user that they have not put in the right arguments
            print("Too many input arguments(ONLY PUT DATA DIR AND CONFIG FILE)!");
            config.success = false;
    }

    // ensure that the config directory exists
    if (config.success && !std::filesystem::is_directory(config.data_path)){
        print("Data directory not found: " + config.data_path);
        config.success = false;
    }
    return config;
}

// Form the output directory for the plots from the yaml-specified plot.output_folder
std::string plot_directory(const TestConfig& config, const std::string& output_folder){
    return (std::filesystem::path(config.output_dir) / output_folder).string();
}


// main() function to handle overall arg->test->plot flow
int main(int argc, char** argv){
    // Handle the arguments:
    TestConfig test_config = handle_args(argc, argv);

    if (test_config.success == false){
        print("Loading the test config failed! Exiting...");
        return 1;
    }

    // Loading the params or the dataset throws on bad input; report it and stop.
    try {
        // instantiate the TestRunner and Plotter
        TestRunner test_runner(test_config);
        TestPlotter plotter(test_runner.params(), plot_directory(test_config, test_runner.params().plot.output_folder));

        // Run the test and plot the results
        std::vector<LogEntry> estimator_data = test_runner.run_test();

        // Now plot the data with the plotter
        if (test_runner.params().plot.enabled){
            plotter.plot_test(estimator_data);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
