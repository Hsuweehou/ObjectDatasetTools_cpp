#pragma once

namespace odt {

// 与 algorithm/ObjectDatasetTools/config/registrationParameters.py 一致
inline constexpr double kVoxelSize = 0.001;
inline constexpr double kVoxelR = 0.000324;//0.0002
inline constexpr int kKNeighbors = 10;
inline constexpr int kLabelInterval = 1;
/// 与 registrationParameters.RECONSTRUCTION_INTERVAL 一致（register_scene 抽帧合并）
inline constexpr int kReconstructionInterval = 9;
/// register_scene：将各步骤中间点云写入 <序列>/register_scene_steps/（便于 MeshLab/Open3D 查看）
inline constexpr bool kRegisterSceneExportStepPly = true;

enum class IcpMethod { kPointToPlane, kColoredIcp };

// 默认与 Python ICP_METHOD = "point-to-plane" 一致
inline constexpr IcpMethod kDefaultIcpMethod = IcpMethod::kPointToPlane;

/// ICP 单次最大迭代（粗配准 + 精配准各一次；默认 Open3D 为 30，略减可提速）
inline constexpr int kIcpMaxIterations = 20;
/// 位姿图全局优化迭代上限（默认 100）
inline constexpr int kGlobalOptMaxIterations = 50;
/// 全局优化中 Levenberg–Marquardt 内层迭代（默认 20）
inline constexpr int kGlobalOptMaxLmIterations = 12;

/// ArUco 物理边长（米），与 solvePnP / 可视化一致；需与实际打印 tag 尺寸一致
inline constexpr double kArucoMarkerLengthMeters = 0.056;

/// MarkerRegistration：角点深度在 (2*W+1)^2 邻域内取有效点，再对 x/y/z 分量分别取中值，抑制飞点
inline constexpr int kMarkerCornerDepthHalfWindow = 3;
/// 邻域内至少需要多少个有效深度点才采纳该角点
inline constexpr int kMarkerCornerMinValidSamples = 5;
/// 打印 MarkerRegistration 诊断（帧对、点数、RMSE、邻域失败次数等）
inline constexpr bool kMarkerRegistrationDiagnostics = true;
/// true：每个候选帧对都打诊断（输出量极大）；false：只对相邻帧 src->src+1 打印（与 ICP 回退相关）
inline constexpr bool kMarkerRegistrationDiagVerbose = false;
/// Marker 3D 对应点在 SVD+RANSAC 风格检验中的平均误差阈值（米）；远距离视角深度噪声大，10mm 常过不去
inline constexpr double kMarkerMatchRmseTolMeters = 0.01;

}  // namespace odt
