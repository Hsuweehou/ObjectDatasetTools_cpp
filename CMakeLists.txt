cmake_minimum_required(VERSION 3.16)
project(object_dataset_tools_cpp LANGUAGES CXX)

# 与 FindOpenGL（GLVND）一致；避免依赖第三方 Open3DConfig.cmake 内的 CMP0072 OLD
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# 本文件位于 algorithm/ObjectDatasetTools_cpp：仓库根为 ${CMAKE_CURRENT_SOURCE_DIR}/../..
get_filename_component(ODT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# 与仓库根 CMakeLists 一致：3rdParty 与仓库目录并列（…/project_template_scanner/../3rdParty）
if(NOT DEFINED THIRD_PARTY_LIBRARY_DIR)
    if(DEFINED ENV{THIRD_PARTY_LIBRARY_DIR})
        set(THIRD_PARTY_LIBRARY_DIR $ENV{THIRD_PARTY_LIBRARY_DIR} CACHE PATH "3rd party root")
    else()
        set(THIRD_PARTY_LIBRARY_DIR "${ODT_REPO_ROOT}/../3rdParty" CACHE PATH "3rd party root")
    endif()
endif()

if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR ${THIRD_PARTY_LIBRARY_DIR}/OpenCV4.X_GPU/x64/vc16/lib)
endif()
find_package(OpenCV REQUIRED)

set(Open3D_DIR ${THIRD_PARTY_LIBRARY_DIR}/open3d/cmake)
find_package(Open3D REQUIRED PATHS ${Open3D_DIR})

set(Eigen3_DIR ${THIRD_PARTY_LIBRARY_DIR}/Eigen/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)

# glog 依赖 gflags（与仓库根 CMakeLists 一致）
if(NOT DEFINED gflags_DIR)
    set(gflags_DIR ${THIRD_PARTY_LIBRARY_DIR}/gflags/lib/cmake/gflags)
endif()
find_package(gflags REQUIRED)
if(NOT DEFINED glog_DIR)
    set(glog_DIR ${THIRD_PARTY_LIBRARY_DIR}/glog/lib/cmake/glog)
endif()
find_package(glog REQUIRED)

# AprilTag：需要时在目标上链接 algorithm/apriltag_processor（其还依赖 apriltag、glog、Open3D、my_pointcloud_processor 等）

get_filename_component(_ODT_AIR_SDK_DEFAULT "${ODT_REPO_ROOT}/SDK_C++" ABSOLUTE)
if(NOT DEFINED ODT_AIR_SDK_ROOT)
    set(ODT_AIR_SDK_ROOT "${_ODT_AIR_SDK_DEFAULT}" CACHE PATH "AIRScanner SDK 根目录（含 include/、lib/cmake/AIRScanner）")
endif()
if(NOT DEFINED AIRScanner_DIR)
    set(AIRScanner_DIR "${ODT_AIR_SDK_ROOT}/lib/cmake/AIRScanner")
endif()
find_package(AIRScanner REQUIRED PATHS "${AIRScanner_DIR}")

set(nlohmann_json_INCLUDE_DIRS ${THIRD_PARTY_LIBRARY_DIR}/nlohmann/include)

set(NUMCPP_INCLUDE_DIR "${THIRD_PARTY_LIBRARY_DIR}/NumCpp_2.16.1/include" CACHE PATH "NumCpp header-only include root")

add_library(odt_core STATIC
    src/async_progress.cpp
    src/camera_utils.cpp
    src/odt_registration.cpp
    src/pose_visualization.cpp
    src/register_scene_fusion.cpp
)
target_include_directories(odt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_INCLUDE_DIRS}
)
target_link_libraries(odt_core PUBLIC
    ${OpenCV_LIBS}
    Open3D::Open3D
    Eigen3::Eigen
)

# 原各 main 中的逻辑（Run*），无 main；record_air 使用 odt::AirRecordScanner（AIRScanner SDK，不经 scanner_a）
add_library(odt_tools STATIC
    src/air_record_scanner.cpp
    src/record_air.cpp
    src/compute_gt_poses.cpp
    src/register_scene.cpp
    src/create_label_files.cpp
)
target_include_directories(odt_tools PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "${ODT_REPO_ROOT}/include"
    "${NUMCPP_INCLUDE_DIR}"
)
target_link_libraries(odt_tools PUBLIC
    odt_core
    AIRScanner::AIRScanner
    glog::glog
    ${OpenCV_LIBS}
    Open3D::Open3D
)
if(WIN32)
    target_compile_definitions(odt_tools PRIVATE GLOG_USE_GLOG_EXPORT)
endif()

add_executable(object_dataset_tools test/main.cpp)
target_link_libraries(object_dataset_tools PRIVATE odt_tools)

# Windows：构建后自动将运行时 DLL 复制到 exe 所在目录（与根工程 scanner_orbbec_demo 行为一致）
if(WIN32)
    if(DEFINED ORBBEC_SDK_ROOT)
        set(_ODT_ORBBEC_ROOT "${ORBBEC_SDK_ROOT}")
    else()
        set(_ODT_ORBBEC_ROOT "${ODT_REPO_ROOT}/scanner_SDKs/Orbbec_SDK")
    endif()
    set(_ODT_ORBBEC_DLL "${_ODT_ORBBEC_ROOT}/bin/OrbbecSDK.dll")
    set(_ODT_ORBBEC_EXT "${_ODT_ORBBEC_ROOT}/bin/extensions")

    if(EXISTS "${_ODT_ORBBEC_DLL}")
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ODT_ORBBEC_DLL}"
                $<TARGET_FILE_DIR:object_dataset_tools>
            COMMENT "[object_dataset_tools] Copy OrbbecSDK.dll")
    endif()
    if(EXISTS "${_ODT_ORBBEC_EXT}")
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${_ODT_ORBBEC_EXT}"
                $<TARGET_FILE_DIR:object_dataset_tools>/extensions
            COMMENT "[object_dataset_tools] Copy Orbbec extensions")
    endif()

    if(TARGET AIRScanner::AIRScanner)
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:AIRScanner::AIRScanner>
                $<TARGET_FILE_DIR:object_dataset_tools>
            COMMENT "[object_dataset_tools] Copy AIRScanner.dll")
    endif()
    set(_ODT_ONPLUS_BIN "${ODT_REPO_ROOT}/scanner_SDKs/Onplus_SDK_V1.0.2.8/C_C++/Bin/x64")
    if(EXISTS "${_ODT_ONPLUS_BIN}/ABHAMCAMSDK.dll")
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ODT_ONPLUS_BIN}/ABHAMCAMSDK.dll" $<TARGET_FILE_DIR:object_dataset_tools>
            COMMENT "[object_dataset_tools] Copy ABHAMCAMSDK.dll")
    endif()
    if(EXISTS "${_ODT_ONPLUS_BIN}/libssh2.dll")
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ODT_ONPLUS_BIN}/libssh2.dll" $<TARGET_FILE_DIR:object_dataset_tools>
            COMMENT "[object_dataset_tools] Copy libssh2.dll (Onplus)")
    endif()

    if(OpenCV_DIR)
        set(_odt_opencv_bin "${OpenCV_DIR}/../bin")
        if(NOT EXISTS "${_odt_opencv_bin}")
            set(_odt_opencv_bin "${OpenCV_DIR}/bin")
        endif()
        if(EXISTS "${_odt_opencv_bin}")
            file(GLOB_RECURSE _odt_opencv_dlls "${_odt_opencv_bin}/*.dll")
            foreach(_dll IN LISTS _odt_opencv_dlls)
                add_custom_command(
                    TARGET object_dataset_tools POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}"
                        $<TARGET_FILE_DIR:object_dataset_tools>
                    COMMENT "[object_dataset_tools] Copy OpenCV DLL")
            endforeach()
            if(EXISTS "${_odt_opencv_bin}/opencv4")
                add_custom_command(
                    TARGET object_dataset_tools POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${_odt_opencv_bin}/opencv4"
                        $<TARGET_FILE_DIR:object_dataset_tools>/opencv4
                    COMMENT "[object_dataset_tools] Copy OpenCV opencv4 data")
            endif()
        endif()
    endif()

    if(TARGET glog::glog)
        add_custom_command(
            TARGET object_dataset_tools POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:glog::glog>
                $<TARGET_FILE_DIR:object_dataset_tools>
            COMMENT "[object_dataset_tools] Copy glog.dll")
    endif()

    # Open3D 预编译包常见布局：${Open3D_DIR}/../../bin 或 ../bin
    if(Open3D_DIR)
        set(_odt_o3d_bin "${Open3D_DIR}/../../bin")
        if(NOT EXISTS "${_odt_o3d_bin}")
            set(_odt_o3d_bin "${Open3D_DIR}/../bin")
        endif()
        if(EXISTS "${_odt_o3d_bin}")
            file(GLOB _odt_o3d_dlls "${_odt_o3d_bin}/*.dll")
            foreach(_dll IN LISTS _odt_o3d_dlls)
                add_custom_command(
                    TARGET object_dataset_tools POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}"
                        $<TARGET_FILE_DIR:object_dataset_tools>
                    COMMENT "[object_dataset_tools] Copy Open3D DLL")
            endforeach()
        endif()
    endif()
endif()
