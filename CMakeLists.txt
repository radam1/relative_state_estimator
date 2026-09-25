cmake_minimum_required(VERSION 3.14)
project(flycrane_ekf_standalone VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Declare Dependencies 
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Use an installed yaml-cpp if there is one, otherwise build it from source.
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  include(FetchContent)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(yaml-cpp)
endif()
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)       
endif()

# form library
add_library(flycrane_ekf_standalone
  src/csv_io.cpp
  src/params.cpp
  src/dynamics.cpp
  src/ekf.cpp
)
target_include_directories(flycrane_ekf_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(flycrane_ekf_standalone
  PUBLIC Eigen3::Eigen
  PRIVATE ${YAML_CPP_TARGET}
)
target_compile_options(flycrane_ekf_standalone PRIVATE -Wall -Wextra -Wpedantic)

# tools from claude
add_executable(dataset_info tools/dataset_info.cpp)
target_link_libraries(dataset_info PRIVATE flycrane_ekf_standalone)
target_compile_options(dataset_info PRIVATE -Wall -Wextra -Wpedantic)

# Install MatPlotLib-cpp for the plotter
find_package(Python3 COMPONENTS Interpreter Development)
set(HAVE_MATPLOTLIB FALSE)
if(Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
                  RESULT_VARIABLE MATPLOTLIB_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
  if(MATPLOTLIB_IMPORT_RESULT EQUAL 0)
    set(HAVE_MATPLOTLIB TRUE)
  endif()
endif()

# If matplotlib is installed, add the ekf teset 
if(HAVE_MATPLOTLIB)
  add_executable(run_ekf_test src/run_ekf_test.cpp src/plotter.cpp)
  target_include_directories(run_ekf_test SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplotlib-cpp)
  target_link_libraries(run_ekf_test PRIVATE flycrane_ekf_standalone Python3::Python)
  target_compile_options(run_ekf_test PRIVATE -Wall -Wextra -Wpedantic)
else()
  message(WARNING "run_ekf_test is NOT built: no Python with matplotlib found "
                  "(Python3_EXECUTABLE = '${Python3_EXECUTABLE}').  Re-run cmake with "
                  "-DPython3_EXECUTABLE=/path/to/python, e.g. /opt/anaconda3/bin/python")
endif()

# unit testing
enable_testing()
add_executable(test_dynamics tests/test_dynamics.cpp)
target_link_libraries(test_dynamics PRIVATE flycrane_ekf_standalone)
target_compile_options(test_dynamics PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_dynamics
  COMMAND test_dynamics ${CMAKE_CURRENT_SOURCE_DIR}/params/uwb_imu_ekf.yaml)
