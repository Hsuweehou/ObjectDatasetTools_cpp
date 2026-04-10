#pragma once

#include "odt/utils/air_record_scanner.hpp"
#include "odt/utils/camera_utils.hpp"

#include <AIRScanner.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <string>

namespace odt {

bool EqCi(const char* a, const char* b);

/// 深度(mm) + 彩色 BGR → 灰度着色点云，写入 `pointclouds/{index}.ply`
bool SaveFusionPointCloudGray(const std::string& pointcloud_dir,
                              int&                   pc_index,
                              const cv::Mat&         depth_mm_aligned,
                              const cv::Mat&         tex_bgr,
                              const CameraIntrinsics& intr);

bool ParsePositiveInt(const char* s, int& out);
bool ParsePort(const char* s, std::uint16_t& out);
void ParseTailNetworkParams(int argc, char** argv, int i0, AirScannerConfig& cfg);

void EnsureLinemodCaptureDirs(const std::string& base);

cv::Mat FloatDepthMmToMat(const float* depth, std::uint32_t dw, std::uint32_t dh);

/// 从 AIRFrameData 解析深度图宽高：优先 depth_width/height，其次 width/height，并与 depth_map_size、纹理尺寸交叉校验。
bool ResolveAirFrameDepthDims(const AIRFrameData& f, std::uint32_t& out_w, std::uint32_t& out_h);

/// 深度图（毫米，CV_32F）。若 depth_map 为空或无法解析尺寸则返回空 Mat。
cv::Mat AirFrameDepthMmToMat(const AIRFrameData& f);

/// 与 scanner_a SaveTextureMap 一致：灰度则转 BGR，RGB 则 RGB→BGR
cv::Mat AirFrameTextureToBgr(const AIRFrameData& f);

void WriteDepthPngMm(const cv::Mat& depth_mm_aligned, const std::string& path);

cv::Mat ResizeForPreview(const cv::Mat& bgr, int max_width);

cv::Mat MakeInteractiveIdleView(const cv::Mat& last_bgr);

void ApplyEnvScannerAddress(AirScannerConfig& cfg);

/// 与 scanner_a 一致：须先有参数 JSON 路径，再 AIR_ImportConfig
std::string ResolveScannerParamsPath(const AirScannerConfig& cfg);

}  // namespace odt
