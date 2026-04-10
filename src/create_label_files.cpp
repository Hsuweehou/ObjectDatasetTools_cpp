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
#include <sstream>
#include <string>
#include <vector>

#include "odt/async_progress.hpp"
#include "odt/camera_utils.hpp"
#include "odt/npy_io.hpp"
#include "odt/registration_params.hpp"

namespace {

std::string NormalizeFolder(const std::string& arg) {
    std::string s = arg;
    std::replace(s.begin(), s.end(), '\\', '/');
    if (!s.empty() && s.back() != '/') {
        s += '/';
    }
    return s;
}

std::string ObjectNameFromFolder(const std::string& folder_norm) {
    const std::string pref = "LINEMOD/";
    if (folder_norm.size() > pref.size() &&
        folder_norm.compare(0, pref.size(), pref) == 0) {
        std::string rest = folder_norm.substr(pref.size());
        while (!rest.empty() && (rest.back() == '/' || rest.back() == '\\')) {
            rest.pop_back();
        }
        return rest;
    }
    std::filesystem::path p(folder_norm);
    std::string fn = p.filename().string();
    if (fn.empty()) {
        p = p.parent_path();
        fn = p.filename().string();
    }
    return fn;
}

std::vector<std::string> GlobLinemodFolders(const std::string& repo_root) {
    std::vector<std::string> out;
    const std::filesystem::path root = std::filesystem::path(repo_root) / "LINEMOD";
    if (!std::filesystem::exists(root)) {
        return out;
    }
    for (auto& p : std::filesystem::directory_iterator(root)) {
        if (p.is_directory()) {
            std::string s = p.path().generic_string();
            if (!s.empty() && s.back() != '/') {
                s += '/';
            }
            out.push_back(NormalizeFolder(s));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

Eigen::Matrix3d BuildK(const odt::CameraIntrinsics& intr) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = intr.fx;
    K(1, 1) = intr.fy;
    K(0, 2) = intr.ppx;
    K(1, 2) = intr.ppy;
    K(2, 2) = 1.0;
    return K;
}

Eigen::MatrixXd ProjectPoints(const Eigen::MatrixXd& points_xyz,
                              const Eigen::Matrix3d& K) {
    const int n = static_cast<int>(points_xyz.rows());
    Eigen::MatrixXd P(3, n);
    for (int i = 0; i < n; ++i) {
        P(0, i) = points_xyz(i, 0);
        P(1, i) = points_xyz(i, 1);
        P(2, i) = points_xyz(i, 2);
    }
    Eigen::MatrixXd cp = K * P;
    Eigen::MatrixXd out(2, n);
    for (int i = 0; i < n; ++i) {
        const double z = cp(2, i);
        if (std::abs(z) < 1e-12) {
            out(0, i) = 0;
            out(1, i) = 0;
        } else {
            out(0, i) = cp(0, i) / z;
            out(1, i) = cp(1, i) / z;
        }
    }
    return out;
}

Eigen::MatrixXd TransformPoints4x4(const Eigen::MatrixXd& pts,
                                   const Eigen::Matrix4d& T) {
    const int n = static_cast<int>(pts.rows());
    Eigen::MatrixXd out(n, 3);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector4d h(pts(i, 0), pts(i, 1), pts(i, 2), 1.0);
        Eigen::Vector4d r = T * h;
        out(i, 0) = r(0);
        out(i, 1) = r(1);
        out(i, 2) = r(2);
    }
    return out;
}

std::string StrHead8(double x) {
    std::ostringstream o;
    o << x;
    std::string s = o.str();
    if (s.size() > 8) {
        s.resize(8);
    }
    return s;
}

std::string StrHead8Int(int v) {
    std::ostringstream o;
    o << v;
    std::string s = o.str();
    if (s.size() > 8) {
        s.resize(8);
    }
    return s;
}

// box_pts 顺序：0 min min min, 1 min min max, 2 min max min, 3 min max max,
// 4 max min min, 5 max min max, 6 max max min, 7 max max max（与主循环中一致）
void DrawProjectedBox8(cv::Mat& img, const Eigen::MatrixXd& corners_px) {
    if (corners_px.rows() < 9 || img.empty()) {
        return;
    }
    static const int kEdges[12][2] = {
        {0, 2}, {2, 6}, {6, 4}, {4, 0},  // z = min_z 面
        {1, 3}, {3, 7}, {7, 5}, {5, 1},  // z = max_z 面
        {0, 1}, {2, 3}, {4, 5}, {6, 7},  // 竖边
    };
    const cv::Scalar color(0, 255, 0);
    const int thickness = 2;
    auto pt = [&](int corner_idx) -> cv::Point {
        const int r = 1 + corner_idx;
        return cv::Point(static_cast<int>(std::lround(corners_px(r, 0))),
                         static_cast<int>(std::lround(corners_px(r, 1))));
    };
    for (const auto& e : kEdges) {
        cv::line(img, pt(e[0]), pt(e[1]), color, thickness, cv::LINE_AA);
    }
    cv::circle(img,
               cv::Point(static_cast<int>(std::lround(corners_px(0, 0))),
                         static_cast<int>(std::lround(corners_px(0, 1)))),
               4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
}

}  // namespace

namespace odt {

int RunCreateLabelFiles(int argc, char** argv) {
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
        folders.push_back(NormalizeFolder(arg1));
    }

    const int label_interval = odt::kLabelInterval;
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
        if (!odt::load_ndarray_npy(folder + "transforms.npy", transforms, n_pose, n1, n2) ||
            n1 != 4 || n2 != 4) {
            std::cerr << "transforms not computed, run compute_gt_poses first\n";
            continue;
        }

        progress.begin_step("create_label_files: 加载点云与包围盒", 0);
        // Python Ply / trimesh：registeredScene 多为仅顶点、无面的点云 PLY
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
        }
        progress.end_step();
    }
    return 0;
}

}  // namespace odt
