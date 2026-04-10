#include "odt/utils/linemod_dataset.hpp"
#include "odt/utils/camera_utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace odt {

namespace {

bool AsciiEqualsIgnoreCase(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const unsigned char ca = static_cast<unsigned char>(a[i]);
        const unsigned char cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) {
            return false;
        }
    }
    return true;
}

bool DirLooksLikeLinemodObjectSequence(const std::filesystem::path& d) {
    std::error_code ec;
    if (!std::filesystem::is_directory(d, ec)) {
        return false;
    }
    if (std::filesystem::is_regular_file(d / "intrinsics.json", ec)) {
        return true;
    }
    if (std::filesystem::is_directory(d / "JPEGImages", ec)) {
        return true;
    }
    return false;
}

/// 得到「其下的一级子目录即为各物体序列」的容器路径
std::filesystem::path ResolveLinemodContainer(const std::filesystem::path& repo_root) {
    std::error_code ec;
    if (!std::filesystem::is_directory(repo_root, ec)) {
        return {};
    }

    const auto exact = repo_root / "LINEMOD";
    if (std::filesystem::exists(exact, ec)) {
        return exact;
    }

    for (auto& p : std::filesystem::directory_iterator(repo_root)) {
        if (!p.is_directory()) {
            continue;
        }
        const std::string name = p.path().filename().string();
        if (AsciiEqualsIgnoreCase(name, "LINEMOD")) {
            return p.path();
        }
    }

    const std::string leaf = repo_root.filename().string();
    if (!leaf.empty() && AsciiEqualsIgnoreCase(leaf, "LINEMOD")) {
        return repo_root;
    }

    for (auto& p : std::filesystem::directory_iterator(repo_root)) {
        if (!p.is_directory()) {
            continue;
        }
        if (DirLooksLikeLinemodObjectSequence(p.path())) {
            return repo_root;
        }
    }

    return {};
}

}  // namespace

std::string NormalizeLinemodFolderPath(const std::string& arg) {
    std::string s = arg;
    std::replace(s.begin(), s.end(), '\\', '/');
    if (!s.empty() && s.back() != '/') {
        s += '/';
    }
    return s;
}

std::string NormalizeRepoRootForLinemodWorkspace(const std::string& workspace_path) {
    std::string s = workspace_path;
    std::replace(s.begin(), s.end(), '\\', '/');
    while (!s.empty() && s.back() == '/') {
        s.pop_back();
    }
    if (s.empty()) {
        return s;
    }
    std::filesystem::path p(s);
    const std::string leaf = p.filename().string();
    if (!leaf.empty() && AsciiEqualsIgnoreCase(leaf, "LINEMOD")) {
        p = p.parent_path();
        s = p.generic_string();
        std::replace(s.begin(), s.end(), '\\', '/');
        while (!s.empty() && s.back() == '/') {
            s.pop_back();
        }
    }
    return s;
}

std::vector<std::string> GlobLinemodFolders(const std::string& repo_root) {
    std::vector<std::string> out;
    const std::filesystem::path container = ResolveLinemodContainer(std::filesystem::path(repo_root));
    std::error_code             ec;
    if (container.empty() || !std::filesystem::is_directory(container, ec)) {
        return out;
    }
    for (auto& p : std::filesystem::directory_iterator(container)) {
        if (p.is_directory()) {
            std::string str = p.path().generic_string();
            out.push_back(NormalizeLinemodFolderPath(str));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
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
    std::string             fn = p.filename().string();
    if (fn.empty()) {
        p  = p.parent_path();
        fn = p.filename().string();
    }
    return fn;
}

int CountJpegImagesInDir(const std::string& jpeg_dir) {
    int n = 0;
    if (!std::filesystem::exists(jpeg_dir)) {
        return 0;
    }
    for (auto& p : std::filesystem::directory_iterator(jpeg_dir)) {
        const auto ext = p.path().extension().string();
        if (ext == ".jpg" || ext == ".JPG") {
            ++n;
        }
    }
    return n;
}

bool LoadLinemodRgbdPair(const std::string& seq_path,
                         int                 frame_index,
                         int                 label_interval,
                         const CameraIntrinsics& intr,
                         cv::Mat&            out_bgr,
                         cv::Mat&            out_xyz) {
    const int fid = frame_index * label_interval;
    const std::string img_path =
        seq_path + "/JPEGImages/" + std::to_string(fid) + ".jpg";
    const std::string depth_path =
        seq_path + "/depth/" + std::to_string(fid) + ".png";
    out_bgr = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat depth_u16 = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (out_bgr.empty() || depth_u16.empty() || depth_u16.type() != CV_16U) {
        return false;
    }
    ResizeDepthUint16ToMatchColor(out_bgr, depth_u16);
    DepthUint16ToPointCloud(depth_u16, intr, out_xyz);
    return true;
}

}  // namespace odt
