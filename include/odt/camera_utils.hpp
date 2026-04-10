#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace odt {

struct CameraIntrinsics {
    double fx = 0;
    double fy = 0;
    double ppx = 0;
    double ppy = 0;
    int width = 0;
    int height = 0;
    /// z(米) = depth_u16 * depth_scale（与 Python intrinsics.json 一致）
    double depth_scale = 0.001;
};

/// 深度图与彩色图尺寸不一致时，将深度最近邻缩放到彩色尺寸再反投影，保证 (u,v) 与 BGR 像素一一对应（与采集端对齐到 color 一致）
inline void ResizeDepthUint16ToMatchColor(const cv::Mat& color_bgr,
                                          cv::Mat& depth_u16) {
    CV_Assert(color_bgr.channels() == 3);
    CV_Assert(depth_u16.type() == CV_16U && depth_u16.channels() == 1);
    if (color_bgr.size() == depth_u16.size()) {
        return;
    }
    cv::Mat resized;
    cv::resize(depth_u16, resized, color_bgr.size(), 0, 0, cv::INTER_NEAREST);
    depth_u16 = std::move(resized);
}

inline void DepthUint16ToPointCloud(const cv::Mat& depth_u16,
                                    const CameraIntrinsics& intr,
                                    cv::Mat& out_xyz) {
    CV_Assert(depth_u16.type() == CV_16U && depth_u16.channels() == 1);
    const int h = depth_u16.rows;
    const int w = depth_u16.cols;
    out_xyz.create(h, w, CV_32FC3);
    for (int v = 0; v < h; ++v) {
        const uint16_t* row_d = depth_u16.ptr<uint16_t>(v);
        cv::Vec3f* row_o = out_xyz.ptr<cv::Vec3f>(v);
        for (int u = 0; u < w; ++u) {
            const double z = static_cast<double>(row_d[u]) * intr.depth_scale;
            if (row_d[u] == 0 || z <= 0) {
                row_o[u] = cv::Vec3f(0, 0, 0);
                continue;
            }
            const double x = (static_cast<double>(u) - intr.ppx) / intr.fx * z;
            const double y = (static_cast<double>(v) - intr.ppy) / intr.fy * z;
            row_o[u] = cv::Vec3f(static_cast<float>(x), static_cast<float>(y),
                                 static_cast<float>(z));
        }
    }
}

inline void DepthFloatMmToPointCloud(const cv::Mat& depth_mm,
                                     const CameraIntrinsics& intr,
                                     cv::Mat& out_xyz) {
    CV_Assert(depth_mm.type() == CV_32F && depth_mm.channels() == 1);
    const int h = depth_mm.rows;
    const int w = depth_mm.cols;
    out_xyz.create(h, w, CV_32FC3);
    for (int v = 0; v < h; ++v) {
        const float* row_d = depth_mm.ptr<float>(v);
        cv::Vec3f* row_o = out_xyz.ptr<cv::Vec3f>(v);
        for (int u = 0; u < w; ++u) {
            const double z_m = static_cast<double>(row_d[u]) * 0.001;
            if (!(row_d[u] > 0) || z_m <= 0) {
                row_o[u] = cv::Vec3f(0, 0, 0);
                continue;
            }
            const double x =
                (static_cast<double>(u) - intr.ppx) / intr.fx * z_m;
            const double y =
                (static_cast<double>(v) - intr.ppy) / intr.fy * z_m;
            row_o[u] = cv::Vec3f(static_cast<float>(x), static_cast<float>(y),
                                 static_cast<float>(z_m));
        }
    }
}

bool LoadIntrinsicsJson(const std::string& path, CameraIntrinsics& out);

}  // namespace odt
