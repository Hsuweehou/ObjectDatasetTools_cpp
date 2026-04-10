#pragma once

#include <opencv2/core.hpp>

#include "odt/camera_utils.hpp"

namespace odt {

/// 检测 ArUco，用 PnP 估计 tag 位姿并绘制坐标轴；无检测则返回 false
bool DrawArucoTagAxesOnImage(cv::Mat& bgr,
                            const CameraIntrinsics& intr,
                            double marker_length_m);

}  // namespace odt
