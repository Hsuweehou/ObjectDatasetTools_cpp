#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <open3d/Open3D.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "odt/object_dataset_tools.hpp"
#include "odt/io/npy_io.hpp"
#include "odt/registration/registration_params.hpp"
#include "odt/utils/async_progress.hpp"
#include "odt/utils/camera_utils.hpp"
#include "odt/utils/label_projection.hpp"
#include "odt/utils/linemod_dataset.hpp"

namespace odt {

int ObjectDatasetTools::createLabelFiles(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ... create_label_files <path|all>\n";
        return 1;
    }

    std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::string> folders;
    const std::string arg1 = argv[1];
    if (arg1 == "all") {
        folders = GlobLinemodFolders(cwd.string());
        if (folders.empty()) {
            std::cerr << "No LINEMOD/*/\n";
            return 1;
        }
    } else {
        folders.push_back(NormalizeLinemodFolderPath(arg1));
    }

    const int label_interval = odt::GetRegistrationParams().label_interval;
    odt::AsyncProgress progress;

    for (size_t classlabel = 0; classlabel < folders.size(); ++classlabel) {
        const std::string folder = folders[classlabel];
        std::cout << ObjectNameFromFolder(folder) << " is assigned class label "
                  << classlabel << ".\n";

        odt::CameraIntrinsics intr;
        if (!odt::LoadIntrinsicsJson(folder + "intrinsics.json", intr)) {
            std::cerr << "intrinsics missing\n";
            continue;
        }
        const Eigen::Matrix3d K = BuildK(intr);

        std::filesystem::create_directories(folder + "labels");
        std::filesystem::create_directories(folder + "mask");
        std::filesystem::create_directories(folder + "transforms");
        std::filesystem::create_directories(folder + "labelled_img");

        nc::NdArray<double> transforms;
        std::size_t         n_pose = 0, n1 = 0, n2 = 0;
        if (!odt::load_ndarray_npy(folder + "transforms.npy", transforms, n_pose, n1,
                                   n2) ||
            n1 != 4 || n2 != 4) {
            std::cerr << "transforms not computed, run compute_gt_poses first\n";
            continue;
        }

        progress.begin_step("create_label_files: 加载点云与包围盒", 0);
        open3d::geometry::PointCloud pcd;
        if (!open3d::io::ReadPointCloud(folder + "registeredScene.ply", pcd) ||
            !pcd.HasPoints()) {
            progress.end_step();
            std::cerr << "registeredScene.ply missing or empty\n";
            continue;
        }
        auto pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(pcd);

        open3d::geometry::OrientedBoundingBox obb =
            open3d::geometry::OrientedBoundingBox::CreateFromPoints(pcd_ptr->points_);

        Eigen::Matrix4d Tform = Eigen::Matrix4d::Identity();
        Tform.block<3, 3>(0, 0) = obb.R_.transpose();
        Tform.block<3, 1>(0, 3) = -obb.R_.transpose() * obb.center_;
        pcd_ptr->Transform(Tform);

        const std::string obj_name = ObjectNameFromFolder(folder);
        const std::string export_path = folder + obj_name + ".ply";
        open3d::io::WritePointCloud(export_path, *pcd_ptr, true);

        open3d::geometry::AxisAlignedBoundingBox aabb =
            pcd_ptr->GetAxisAlignedBoundingBox();
        const Eigen::Vector3d mn = aabb.min_bound_;
        const Eigen::Vector3d mx = aabb.max_bound_;
        const double min_x = mn(0), min_y = mn(1), min_z = mn(2);
        const double max_x = mx(0), max_y = mx(1), max_z = mx(2);

        Eigen::MatrixXd box_pts(8, 3);
        box_pts << min_x, min_y, min_z, min_x, min_y, max_z, min_x, max_y, min_z,
            min_x, max_y, max_z, max_x, min_y, min_z, max_x, min_y, max_z, max_x,
            max_y, min_z, max_x, max_y, max_z;

        Eigen::Vector3d centroid(0, 0, 0);
        for (const auto& v : pcd_ptr->points_) {
            centroid += v;
        }
        centroid /= static_cast<double>(std::max<size_t>(1, pcd_ptr->points_.size()));

        Eigen::MatrixXd points_with_center(9, 3);
        points_with_center.row(0) = centroid.transpose();
        points_with_center.bottomRows(8) = box_pts;

        const Eigen::Matrix4d invTform = Tform.inverse();
        Eigen::MatrixXd points_original =
            TransformPoints4x4(points_with_center, invTform);

        progress.end_step();
        progress.begin_step("create_label_files: 逐帧标签",
                            static_cast<int64_t>(n_pose));

        for (size_t i = 0; i < n_pose; ++i) {
            progress.set_current(static_cast<int64_t>(i));
            Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    Ti(r, c) = transforms(static_cast<int>(i), r * 4 + c);
                }
            }
            const Eigen::Matrix4d transform = Ti.inverse();

            auto pcd_copy = std::make_shared<open3d::geometry::PointCloud>(*pcd_ptr);
            const Eigen::Matrix4d T_mesh = transform * invTform;
            pcd_copy->Transform(T_mesh);

            const std::string img_path =
                folder + "JPEGImages/" +
                std::to_string(static_cast<int>(i) * label_interval) + ".jpg";
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                continue;
            }

            Eigen::MatrixXd transformed =
                TransformPoints4x4(points_original, transform);
            Eigen::MatrixXd corners_proj = ProjectPoints(transformed, K);
            Eigen::MatrixXd corners = corners_proj.transpose();
            const Eigen::MatrixXd corners_px = corners;
            const double iw = static_cast<double>(intr.width);
            const double ih = static_cast<double>(intr.height);
            for (int r = 0; r < corners.rows(); ++r) {
                corners(r, 0) /= iw;
                corners(r, 1) /= ih;
            }

            const std::string tf_path =
                folder + "transforms/" +
                std::to_string(static_cast<int>(i) * label_interval) + ".npy";
            {
                nc::NdArray<double> tfm(4, 4);
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        tfm(r, c) = T_mesh(r, c);
                    }
                }
                odt::save_ndarray_npy(tf_path, tfm, 4, 4);
            }

            const size_t ns = pcd_copy->points_.size();
            Eigen::MatrixXd sp(10000, 3);
            if (ns == 0) {
                continue;
            }
            std::mt19937 rng(static_cast<unsigned>(i + classlabel * 10007));
            std::uniform_int_distribution<size_t> dist(0, ns - 1);
            for (int j = 0; j < 10000; ++j) {
                sp.row(j) = pcd_copy->points_[dist(rng)].transpose();
            }
            Eigen::MatrixXd masks_proj = ProjectPoints(sp, K);
            Eigen::MatrixXd masks = masks_proj.transpose();

            double min_px = masks.col(0).minCoeff();
            double min_py = masks.col(1).minCoeff();
            double max_px = masks.col(0).maxCoeff();
            double max_py = masks.col(1).maxCoeff();

            cv::Mat image_mask =
                cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
            for (int r = 0; r < masks.rows(); ++r) {
                const int px = static_cast<int>(masks(r, 0));
                const int py = static_cast<int>(masks(r, 1));
                cv::circle(image_mask, cv::Point(px, py), 5, 255, -1);
            }
            cv::Mat thresh;
            cv::threshold(image_mask, thresh, 30, 255, cv::THRESH_BINARY);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(thresh.clone(), contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                continue;
            }
            size_t best = 0;
            double best_a = 0;
            for (size_t k = 0; k < contours.size(); ++k) {
                const double a = std::abs(cv::contourArea(contours[k]));
                if (a > best_a) {
                    best_a = a;
                    best = k;
                }
            }
            image_mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
            cv::drawContours(image_mask, contours, static_cast<int>(best),
                             cv::Scalar(255), -1);

            const std::string mask_path =
                folder + "mask/" +
                std::to_string(static_cast<int>(i) * label_interval) + ".png";
            cv::imwrite(mask_path, image_mask);

            const std::string label_path =
                folder + "labels/" +
                std::to_string(static_cast<int>(i) * label_interval) + ".txt";
            std::ofstream lf(label_path);
            lf << StrHead8Int(static_cast<int>(classlabel)) << " ";
            for (int r = 0; r < corners.rows(); ++r) {
                for (int c = 0; c < corners.cols(); ++c) {
                    lf << StrHead8(corners(r, c)) << " ";
                }
            }
            lf << StrHead8((max_px - min_px) / iw) << " ";
            lf << StrHead8((max_py - min_py) / ih);
            lf.close();

            cv::Mat labelled = img.clone();
            DrawProjectedBox8(labelled, corners_px);
            const std::string labelled_path =
                folder + "labelled_img/" +
                std::to_string(static_cast<int>(i) * label_interval) + ".jpg";
            cv::imwrite(labelled_path, labelled);

            const int pct = static_cast<int>(
                100 * (static_cast<int>(i) + 1) /
                static_cast<int>(std::max<size_t>(1, n_pose)));
            std::cout << "[odt_clf] pct " << pct << "\n" << std::flush;
            std::string mp = mask_path;
            std::string lp = labelled_path;
            std::replace(mp.begin(), mp.end(), '\\', '/');
            std::replace(lp.begin(), lp.end(), '\\', '/');
            std::cout << "[odt_clf] pair\t" << mp << "\t" << lp << "\n" << std::flush;
        }
        progress.end_step();
    }
    return 0;
}

}  // namespace odt
