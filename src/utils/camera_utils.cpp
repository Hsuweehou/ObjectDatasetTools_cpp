#include "odt/utils/camera_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

namespace odt {

bool LoadIntrinsicsJson(const std::string& path, CameraIntrinsics& out) {
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    nlohmann::json j;
    ifs >> j;
    out.fx = j.at("fx").get<double>();
    out.fy = j.at("fy").get<double>();
    out.ppx = j.at("ppx").get<double>();
    out.ppy = j.at("ppy").get<double>();
    out.height = j.at("height").get<int>();
    out.width = j.at("width").get<int>();
    if (j.contains("depth_scale")) {
        out.depth_scale = j.at("depth_scale").get<double>();
    } else {
        out.depth_scale = 0.001;
    }
    return true;
}

}  // namespace odt
