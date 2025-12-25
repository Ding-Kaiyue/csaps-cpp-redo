cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(csaps
    VERSION 0.1.0
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

# ROS 2 / ament support
find_package(ament_cmake QUIET)
find_package(Eigen3 3.1 REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

set(CSAPS_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty
)

if (BUILD_TESTING)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()

# Generate and install CMake config, version and target files
include(CMakePackageConfigHelpers)
set(CONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}")

# 创建Config版本文件
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/csapsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# 使用configure_file生成Config文件（与hardware_driver一致）
configure_file(
    "cmake/csapsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/csapsConfig.cmake"
    @ONLY
)

# 安装生成的config文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/csapsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/csapsConfigVersion.cmake"
    DESTINATION ${CONFIG_INSTALL_DIR}
)

# 安装targets文件
install(EXPORT csapsTargets
  NAMESPACE csaps::
  FILE csapsTargets.cmake
  DESTINATION ${CONFIG_INSTALL_DIR}
)

# ament support for ROS 2
if(ament_cmake_FOUND)
  ament_package()
endif()
