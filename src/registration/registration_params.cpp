#include "odt/registration/registration_params.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace odt {

namespace {

std::mutex g_params_mutex;
std::string g_override_path;
RegistrationParams g_defaults_builtin() {
    return RegistrationParams{};
}

IcpMethod ParseIcpMethod(const std::string& s) {
    if (s == "colored_icp" || s == "colored-icp" || s == "ColoredICP") {
        return IcpMethod::kColoredIcp;
    }
    return IcpMethod::kPointToPlane;
}

void ApplyRegistrationJson(const nlohmann::json& r, RegistrationParams& p) {
    if (r.contains("voxel_size")) {
        p.voxel_size = r["voxel_size"].get<double>();
    }
    if (r.contains("voxel_r")) {
        p.voxel_r = r["voxel_r"].get<double>();
    }
    if (r.contains("k_neighbors")) {
        p.k_neighbors = r["k_neighbors"].get<int>();
    }
    if (r.contains("label_interval")) {
        p.label_interval = r["label_interval"].get<int>();
    }
    if (r.contains("reconstruction_interval")) {
        p.reconstruction_interval = r["reconstruction_interval"].get<int>();
    }
    if (r.contains("register_scene_export_step_ply")) {
        p.register_scene_export_step_ply =
            r["register_scene_export_step_ply"].get<bool>();
    }
    if (r.contains("icp_method")) {
        p.icp_method = ParseIcpMethod(r["icp_method"].get<std::string>());
    }
    if (r.contains("icp_max_iterations")) {
        p.icp_max_iterations = r["icp_max_iterations"].get<int>();
    }
    if (r.contains("global_opt_max_iterations")) {
        p.global_opt_max_iterations = r["global_opt_max_iterations"].get<int>();
    }
    if (r.contains("global_opt_max_lm_iterations")) {
        p.global_opt_max_lm_iterations = r["global_opt_max_lm_iterations"].get<int>();
    }
    if (r.contains("aruco_marker_length_meters")) {
        p.aruco_marker_length_meters = r["aruco_marker_length_meters"].get<double>();
    }
    if (r.contains("marker_corner_depth_half_window")) {
        p.marker_corner_depth_half_window =
            r["marker_corner_depth_half_window"].get<int>();
    }
    if (r.contains("marker_corner_min_valid_samples")) {
        p.marker_corner_min_valid_samples =
            r["marker_corner_min_valid_samples"].get<int>();
    }
    if (r.contains("marker_registration_diagnostics")) {
        p.marker_registration_diagnostics =
            r["marker_registration_diagnostics"].get<bool>();
    }
    if (r.contains("marker_registration_diag_verbose")) {
        p.marker_registration_diag_verbose =
            r["marker_registration_diag_verbose"].get<bool>();
    }
    if (r.contains("marker_match_rmse_tol_meters")) {
        p.marker_match_rmse_tol_meters =
            r["marker_match_rmse_tol_meters"].get<double>();
    }
}

bool TryReadJsonFile(const std::string& path, nlohmann::json* out) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return false;
    }
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    try {
        in >> *out;
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    return out->is_object();
}

std::string AbsolutePathForDisplay(const std::string& path) {
    std::error_code ec;
    const std::filesystem::path ap = std::filesystem::absolute(path, ec);
    if (ec) {
        return path;
    }
    return ap.string();
}

std::pair<RegistrationParams, std::string> LoadParamsWithSourcePath() {
    RegistrationParams p = g_defaults_builtin();
    std::vector<std::string> candidates;
    {
        std::lock_guard<std::mutex> lock(g_params_mutex);
        if (!g_override_path.empty()) {
            candidates.push_back(g_override_path);
        }
    }
    if (const char* env = std::getenv("ODT_ALGORITHM_PARAMS_JSON")) {
        candidates.emplace_back(env);
    }
    candidates.push_back("./config/algorithm_params.json");
    candidates.push_back("../config/algorithm_params.json");
    for (const auto& path : candidates) {
        nlohmann::json j;
        if (TryReadJsonFile(path, &j) && j.contains("registration") &&
            j["registration"].is_object()) {
            ApplyRegistrationJson(j["registration"], p);
            return {p, AbsolutePathForDisplay(path)};
        }
    }
    return {p, ""};
}

}  // namespace

void SetRegistrationParamsJsonPath(std::string path) {
    std::lock_guard<std::mutex> lock(g_params_mutex);
    g_override_path = std::move(path);
}

namespace {

std::mutex g_cache_mutex;
RegistrationParams g_cached_params;
std::string g_cached_source_path;
bool g_cache_initialized = false;

void RefreshCacheLocked() {
    auto loaded = LoadParamsWithSourcePath();
    g_cached_params = std::move(loaded.first);
    g_cached_source_path = std::move(loaded.second);
    g_cache_initialized = true;
}

}  // namespace

const RegistrationParams& GetRegistrationParams() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (!g_cache_initialized) {
        RefreshCacheLocked();
    }
    return g_cached_params;
}

const std::string& GetRegistrationParamsSourcePath() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (!g_cache_initialized) {
        RefreshCacheLocked();
    }
    return g_cached_source_path;
}

void ReloadRegistrationParams() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    RefreshCacheLocked();
}

}  // namespace odt
