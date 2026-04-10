#pragma once

#include <Eigen/Core>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>

#include "odt/utils/camera_utils.hpp"

namespace odt {

inline Eigen::Matrix3d BuildK(const CameraIntrinsics& intr) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = intr.fx;
    K(1, 1) = intr.fy;
    K(0, 2) = intr.ppx;
    K(1, 2) = intr.ppy;
    K(2, 2) = 1.0;
    return K;
}

inline Eigen::MatrixXd ProjectPoints(const Eigen::MatrixXd& points_xyz,
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

inline Eigen::MatrixXd TransformPoints4x4(const Eigen::MatrixXd& pts,
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

inline std::string StrHead8(double x) {
    std::ostringstream o;
    o << x;
    std::string s = o.str();
    if (s.size() > 8) {
        s.resize(8);
    }
    return s;
}

inline std::string StrHead8Int(int v) {
    std::ostringstream o;
    o << v;
    std::string s = o.str();
    if (s.size() > 8) {
        s.resize(8);
    }
    return s;
}

/// box_pts 顺序：0 min min min … 7 max max max（与 create_label_files 中 box_pts 一致）
inline void DrawProjectedBox8(cv::Mat& img, const Eigen::MatrixXd& corners_px) {
    if (corners_px.rows() < 9 || img.empty()) {
        return;
    }
    static const int kEdges[12][2] = {
        {0, 2}, {2, 6}, {6, 4}, {4, 0},  // z = min_z 面
        {1, 3}, {3, 7}, {7, 5}, {5, 1},  // z = max_z 面
        {0, 1}, {2, 3}, {4, 5}, {6, 7},  // 竖边
    };
    const cv::Scalar color(0, 255, 0);
    const int        thickness = 2;
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

}  // namespace odt
