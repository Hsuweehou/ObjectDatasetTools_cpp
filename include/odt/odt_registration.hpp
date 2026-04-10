#pragma once

#include <Eigen/Core>
#include <memory>
#include <optional>
#include <utility>

#include <opencv2/core.hpp>
#include <open3d/geometry/PointCloud.h>

#include "odt/registration_params.hpp"

namespace odt {

std::optional<Eigen::Matrix4d> MatchRansacStyle(const Eigen::MatrixXd& p,
                                                const Eigen::MatrixXd& p_prime,
                                                double tol = 0.01);

std::shared_ptr<open3d::geometry::PointCloud> BuildPcdFromRgbd(
    const cv::Mat& cad_bgr,
    const cv::Mat& depth_xyz,
    double voxel_size,
    bool estimate_normals);

std::pair<Eigen::Matrix4d, Eigen::Matrix6d> RunIcp(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    double voxel_size,
    double max_corr_coarse,
    double max_corr_fine,
    IcpMethod method);

std::optional<Eigen::Matrix4d> FeatureRegistrationRgb(
    const cv::Mat& cad_src_bgr,
    const cv::Mat& depth_xyz_src,
    const cv::Mat& cad_dst_bgr,
    const cv::Mat& depth_xyz_dst,
    int min_match_count);

/// diag_src_frame / diag_dst_frame：>=0 时写入诊断日志（对应 compute_gt_poses 的帧下标）
std::optional<Eigen::Matrix4d> MarkerRegistration(
    const cv::Mat& cad_src_bgr,
    const cv::Mat& depth_xyz_src,
    const cv::Mat& cad_dst_bgr,
    const cv::Mat& depth_xyz_dst,
    int diag_src_frame = -1,
    int diag_dst_frame = -1);

}  // namespace odt
