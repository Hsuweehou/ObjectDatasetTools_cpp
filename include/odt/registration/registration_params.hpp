#pragma once

#include <string>

namespace odt {

enum class IcpMethod { kPointToPlane, kColoredIcp };

/// 配准相关运行时参数，默认与 algorithm/ObjectDatasetTools/config/registrationParameters.py 及原 constexpr 一致；
/// 数值从 algorithm_params.json 的 `registration` 字段加载（见 config/algorithm_params.json）。
struct RegistrationParams {
    double voxel_size = 0.001;
    double voxel_r = 0.000324;
    int k_neighbors = 10;
    int label_interval = 1;
    int reconstruction_interval = 9;
    bool register_scene_export_step_ply = true;
    IcpMethod icp_method = IcpMethod::kPointToPlane;
    int icp_max_iterations = 20;
    int global_opt_max_iterations = 50;
    int global_opt_max_lm_iterations = 12;
    double aruco_marker_length_meters = 0.056;
    int marker_corner_depth_half_window = 3;
    int marker_corner_min_valid_samples = 5;
    bool marker_registration_diagnostics = true;
    bool marker_registration_diag_verbose = false;
    double marker_match_rmse_tol_meters = 0.01;
};

/// 从 JSON 读取并缓存；解析失败或文件不存在时使用内置默认值。
/// 查找顺序：SetRegistrationParamsJsonPath 覆盖路径 → 环境变量 ODT_ALGORITHM_PARAMS_JSON →
/// `./config/algorithm_params.json` → `../config/algorithm_params.json`。
const RegistrationParams& GetRegistrationParams();

/// 与当前缓存同次加载解析出的配置文件绝对路径；若未从任何 JSON 成功读取则为空。
const std::string& GetRegistrationParamsSourcePath();

/// 重新从磁盘加载 JSON 并更新缓存（未调用时首次 Get 仍会懒加载）。
void ReloadRegistrationParams();

/// 在首次加载前调用可强制指定 JSON 路径（用于测试或自定义部署）。
void SetRegistrationParamsJsonPath(std::string path);

}  // namespace odt
