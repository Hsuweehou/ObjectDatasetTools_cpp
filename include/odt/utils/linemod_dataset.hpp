#pragma once

#include "odt/utils/camera_utils.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace odt {

/// 路径统一为正斜杠并以 `/` 结尾（LINEMOD 序列根目录）
std::string NormalizeLinemodFolderPath(const std::string& arg);

/// 枚举 LINEMOD 数据集下各物体序列目录（路径已 Normalize）。
/// 解析顺序：`repo_root/LINEMOD/`、`repo_root` 下名称为 LINEMOD（不区分大小写）的子目录、
/// `repo_root` 本身若名为 LINEMOD（不区分大小写）、或 `repo_root` 下已有典型物体子目录
///（含 intrinsics.json 或 JPEGImages/）则把 `repo_root` 当作数据集根。
std::vector<std::string> GlobLinemodFolders(const std::string& repo_root);

/// `register_scene all <workspace>`：去掉末尾的 LINEMOD/Linemod 路径段（与 Glob 规则一致）
std::string NormalizeRepoRootForLinemodWorkspace(const std::string& workspace_path);

/// 从 `.../LINEMOD/foo/` 或任意目录路径得到物体名 `foo`
std::string ObjectNameFromFolder(const std::string& folder_norm);

/// `jpeg_dir` 下 `.jpg` / `.JPG` 数量
int CountJpegImagesInDir(const std::string& jpeg_dir);

/// 读取 `JPEGImages/{idx}.jpg` 与 `depth/{idx}.png`，对齐深度并反投影为 xyz（米）
bool LoadLinemodRgbdPair(const std::string& seq_path,
                         int                 frame_index,
                         int                 label_interval,
                         const CameraIntrinsics& intr,
                         cv::Mat&            out_bgr,
                         cv::Mat&            out_xyz);

}  // namespace odt
