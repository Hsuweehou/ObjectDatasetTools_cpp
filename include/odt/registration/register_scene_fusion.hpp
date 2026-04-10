#pragma once

#include "odt/utils/camera_utils.hpp"

#include <open3d/geometry/PointCloud.h>

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace odt {

/// 从 LINEMOD 序列读取 JPEGImages/{idx}.jpg 与 depth/{idx}.png，深度+彩色融合为点云（与 Python load_pcds 一致）。
std::shared_ptr<open3d::geometry::PointCloud> LoadSequenceRgbdPointCloud(
    const std::string& sequence_path,
    int file_index,
    const CameraIntrinsics& intr);

bool WritePointCloudPlyBinary(const std::string& path,
                              const open3d::geometry::PointCloud& pcd);

bool WritePointCloudPlyBinary(const std::string& path,
                              const Eigen::MatrixXd& pts,
                              const Eigen::MatrixXd& cols);

/// 与 register_scene.py::post_process 一致：多帧 KDTree 近邻合并与投票。
void FusePointCloudsKdTreeMerge(
    const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& originals,
    double voxel_r,
    double inlier_r,
    Eigen::MatrixXd& pts,
    Eigen::MatrixXd& cols,
    std::vector<int>& vote,
    const std::function<void()>& on_frame = {});

}  // namespace odt
