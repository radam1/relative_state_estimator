/*
run_ekf_test.cpp

This file will use the EKF implementation in ekf.cpp to test the state estimator
on example rosbag data, converted into csv using bag_converter.py and read into
the cpp code using csv_io.cpp.  The TestRunner itself is in test_runner.cpp.

    run_ekf_test <csv_dir> [params.yaml]
*/

// includes
#include "flycrane_ekf/run_ekf_test.hpp"

#include <exception>
#include <filesystem>

// print a line to the console
void print(const std::string& msg){
    std::cout<<msg<<"\n";
}

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
