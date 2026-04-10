#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <open3d/Open3D.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include <open3d/pipelines/registration/GlobalOptimizationConvergenceCriteria.h>

#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "odt/object_dataset_tools.hpp"
#include "odt/io/npy_io.hpp"
#include "odt/registration/odt_registration.hpp"
#include "odt/registration/pose_visualization.hpp"
#include "odt/registration/registration_params.hpp"
#include "odt/utils/async_progress.hpp"
#include "odt/utils/camera_utils.hpp"
#include "odt/utils/linemod_dataset.hpp"

namespace odt {

int ObjectDatasetTools::computeGtPoses(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ... compute_gt_poses <sequence_dir>\n"
                  << "  sequence_dir: e.g. LINEMOD/sugar (contains JPEGImages, "
                     "depth, intrinsics.json)\n";
        return 1;
    }
    std::string seq = argv[1];
    while (!seq.empty() && (seq.back() == '/' || seq.back() == '\\')) {
        seq.pop_back();
    }
    std::replace(seq.begin(), seq.end(), '\\', '/');

    odt::CameraIntrinsics intr;
    if (!odt::LoadIntrinsicsJson(seq + "/intrinsics.json", intr)) {
        std::cerr << "Failed to read intrinsics.json\n";
        return 1;
    }

    const int label_interval = odt::GetRegistrationParams().label_interval;
    const int total_jpg = CountJpegImagesInDir(seq + "/JPEGImages");
    if (total_jpg < 1) {
        std::cerr << "No JPEGImages found.\n";
        return 1;
    }
    const int n_pcds = total_jpg / label_interval;
    if (n_pcds < 1) {
        std::cerr << "Not enough frames for LABEL_INTERVAL.\n";
        return 1;
    }

    const double voxel_size = odt::GetRegistrationParams().voxel_size;
    const double max_corr_coarse = voxel_size * 15.0;
    const double max_corr_fine = voxel_size * 1.5;
    const int n_neigh = odt::GetRegistrationParams().k_neighbors;
    const int step = std::max(1, n_pcds / std::max(1, n_neigh));

    int64_t pair_total = 0;
    for (int s = 0; s < n_pcds; ++s) {
        for (int t = s + 1; t < n_pcds; t += step) {
            ++pair_total;
        }
    }

    const std::string tag_pose_dir = seq + "/tag_pose";
    std::filesystem::create_directories(tag_pose_dir);

    std::cout << "[odt_gt] phase pairs_begin " << pair_total << "\n" << std::flush;

    odt::AsyncProgress progress;
    progress.begin_step("compute_gt_poses: 构建位姿图", pair_total);

    int64_t pair_done = 0;

    using namespace open3d;
    pipelines::registration::PoseGraph pose_graph;
    Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(odometry);

    std::vector<std::shared_ptr<geometry::PointCloud>> pcds(
        static_cast<size_t>(n_pcds));

    for (int source_id = 0; source_id < n_pcds; ++source_id) {
        if (source_id > 0) {
            pcds[static_cast<size_t>(source_id - 1)].reset();
        }
        cv::Mat color_src, xyz_src;
        if (!LoadLinemodRgbdPair(seq, source_id, label_interval, intr, color_src,
                                 xyz_src)) {
            continue;
        }
        for (int target_id = source_id + 1; target_id < n_pcds;
             target_id += step) {
            progress.advance(1);
            ++pair_done;
            std::cout << "[odt_gt] pairs " << pair_done << " " << pair_total << "\n"
                      << std::flush;
            cv::Mat color_dst, xyz_dst;
            if (!LoadLinemodRgbdPair(seq, target_id, label_interval, intr,
                                     color_dst, xyz_dst)) {
                continue;
            }
            std::optional<Eigen::Matrix4d> res = odt::MarkerRegistration(
                color_src, xyz_src, color_dst, xyz_dst, source_id, target_id);
            if (!res.has_value() && target_id != source_id + 1) {
                continue;
            }
            if (!pcds[static_cast<size_t>(source_id)]) {
                pcds[static_cast<size_t>(source_id)] =
                    odt::BuildPcdFromRgbd(color_src, xyz_src, voxel_size, true);
            }
            if (!pcds[static_cast<size_t>(target_id)]) {
                pcds[static_cast<size_t>(target_id)] =
                    odt::BuildPcdFromRgbd(color_dst, xyz_dst, voxel_size, true);
            }
            Eigen::Matrix4d transformation_icp = Eigen::Matrix4d::Identity();
            Eigen::Matrix6d information_icp = Eigen::Matrix6d::Identity();
            if (!res.has_value()) {
                auto icp = odt::RunIcp(*pcds[static_cast<size_t>(source_id)],
                                       *pcds[static_cast<size_t>(target_id)],
                                       voxel_size, max_corr_coarse,
                                       max_corr_fine,
                                       odt::GetRegistrationParams().icp_method);
                transformation_icp = icp.first;
                information_icp = icp.second;
            } else {
                transformation_icp = *res;
                information_icp =
                    pipelines::registration::GetInformationMatrixFromPointClouds(
                        *pcds[static_cast<size_t>(source_id)],
                        *pcds[static_cast<size_t>(target_id)], max_corr_fine,
                        transformation_icp);
                const int fid_src = source_id * label_interval;
                const int fid_dst = target_id * label_interval;
                cv::Mat vis_src = color_src.clone();
                cv::Mat vis_dst = color_dst.clone();
                if (odt::DrawArucoTagAxesOnImage(vis_src, intr,
                                                 odt::GetRegistrationParams()
                                                     .aruco_marker_length_meters)) {
                    const std::string path_src =
                        tag_pose_dir + "/" + std::to_string(fid_src) + ".jpg";
                    cv::imwrite(path_src, vis_src);
                    std::cout << "[odt_gt] image " << path_src << "\n" << std::flush;
                }
                if (odt::DrawArucoTagAxesOnImage(vis_dst, intr,
                                                 odt::GetRegistrationParams()
                                                     .aruco_marker_length_meters)) {
                    const std::string path_dst =
                        tag_pose_dir + "/" + std::to_string(fid_dst) + ".jpg";
                    cv::imwrite(path_dst, vis_dst);
                    std::cout << "[odt_gt] image " << path_dst << "\n" << std::flush;
                }
            }

            if (target_id == source_id + 1) {
                odometry = transformation_icp * odometry;
                pose_graph.nodes_.push_back(pipelines::registration::PoseGraphNode(
                    odometry.inverse()));
                pose_graph.edges_.emplace_back(
                    source_id, target_id, transformation_icp, information_icp,
                    false);
            } else {
                pose_graph.edges_.emplace_back(
                    source_id, target_id, transformation_icp, information_icp,
                    true);
            }
        }
    }

    progress.end_step();
    std::cout << "[odt_gt] phase global_begin\n" << std::flush;
    progress.begin_step("compute_gt_poses: 全局优化", 0);
    pipelines::registration::GlobalOptimizationOption opt(
        max_corr_fine, 0.25, 1.0, 0);
    pipelines::registration::GlobalOptimizationConvergenceCriteria go_crit(
        odt::GetRegistrationParams().global_opt_max_iterations, 1e-6, 1e-6, 1e-6,
        1e-6, odt::GetRegistrationParams().global_opt_max_lm_iterations);
    pipelines::registration::GlobalOptimization(
        pose_graph, pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
        go_crit, opt);
    progress.end_step();
    std::cout << "[odt_gt] phase global_end\n" << std::flush;

    const int num_nodes = static_cast<int>(pose_graph.nodes_.size());
    if (num_nodes != n_pcds) {
        std::cerr << "warning: pose graph has " << num_nodes
                  << " nodes but frame count is " << n_pcds
                  << " (some RGB-D loads failed). Output uses node count.\n";
    }

    const int num_annotations = num_nodes;
    std::cout << "[odt_gt] phase save_begin " << num_annotations << "\n" << std::flush;
    progress.begin_step("compute_gt_poses: 保存 transforms.npy",
                        static_cast<int64_t>(num_annotations));
    nc::NdArray<double> arr(static_cast<int>(num_annotations * 4), 4);
    for (int i = 0; i < num_annotations; ++i) {
        const Eigen::Matrix4d& P = pose_graph.nodes_[static_cast<size_t>(i)].pose_;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                arr(i * 4 + r, c) = P(r, c);
            }
        }
        progress.advance(1);
        std::cout << "[odt_gt] save " << (i + 1) << " " << num_annotations << "\n"
                  << std::flush;
    }
    progress.end_step();
    nc::reshape(arr, num_annotations, 16);
    const std::string out_npy = seq + "/transforms.npy";
    progress.begin_step("compute_gt_poses: 写入磁盘", 0);
    if (!odt::save_ndarray_npy(out_npy, arr, static_cast<std::size_t>(num_annotations),
                               4, 4)) {
        progress.end_step();
        std::cerr << "Failed to write transforms.npy\n";
        return 1;
    }
    progress.end_step();
    std::cout << "Saved " << out_npy << "\n";
    return 0;
}

}  // namespace odt
