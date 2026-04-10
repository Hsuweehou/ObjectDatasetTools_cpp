#include "odt/utils/air_capture_helpers.hpp"
#include "odt/registration/odt_registration.hpp"
#include "odt/registration/register_scene_fusion.hpp"

#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace odt {

bool EqCi(const char* a, const char* b) {
    if (!a || !b) {
        return a == b;
    }
    while (*a && *b) {
        if (std::tolower(static_cast<unsigned char>(*a)) !=
            std::tolower(static_cast<unsigned char>(*b))) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

bool SaveFusionPointCloudGray(const std::string& pointcloud_dir,
                              int&                   pc_index,
                              const cv::Mat&         depth_mm_aligned,
                              const cv::Mat&         tex_bgr,
                              const CameraIntrinsics& intr) {
    if (tex_bgr.empty() || depth_mm_aligned.empty() ||
        tex_bgr.size() != depth_mm_aligned.size()) {
        return false;
    }
    if (!(intr.fx > 0.0) || !(intr.fy > 0.0)) {
        return false;
    }
    cv::Mat gray8;
    cv::cvtColor(tex_bgr, gray8, cv::COLOR_BGR2GRAY);
    cv::Mat gray_bgr;
    cv::cvtColor(gray8, gray_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat xyz;
    DepthFloatMmToPointCloud(depth_mm_aligned, intr, xyz);
    std::shared_ptr<open3d::geometry::PointCloud> pcd = BuildPcdFromRgbd(gray_bgr, xyz, 0.001, false);
    if (!pcd || pcd->points_.empty()) {
        pcd = BuildPcdFromRgbd(gray_bgr, xyz, 0.01, false);
    }
    if (!pcd || pcd->points_.empty()) {
        pcd = BuildPcdFromRgbd(gray_bgr, xyz, -1.0, false);
    }
    if (!pcd || pcd->points_.empty()) {
        return false;
    }
    const std::string path =
        pointcloud_dir + "/" + std::to_string(pc_index++) + ".ply";
    return WritePointCloudPlyBinary(path, *pcd);
}

bool ParsePositiveInt(const char* s, int& out) {
    if (!s || !*s) {
        return false;
    }
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > 86400) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool ParsePort(const char* s, std::uint16_t& out) {
    if (!s || !*s) {
        return false;
    }
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v == 0UL || v > 65535UL) {
        return false;
    }
    out = static_cast<std::uint16_t>(v);
    return true;
}

void ParseTailNetworkParams(int argc, char** argv, int i0, AirScannerConfig& cfg) {
    if (argc > i0 && argv[i0] && argv[i0][0]) {
        cfg.ip = argv[i0];
    }
    if (argc > i0 + 1) {
        std::uint16_t p = cfg.port;
        if (ParsePort(argv[i0 + 1], p)) {
            cfg.port = p;
        }
    }
    if (argc > i0 + 2 && argv[i0 + 2] && argv[i0 + 2][0]) {
        cfg.scanner_param_path = argv[i0 + 2];
    }
}

void EnsureLinemodCaptureDirs(const std::string& base) {
    std::filesystem::create_directories(base + "/JPEGImages");
    std::filesystem::create_directories(base + "/depth");
}

cv::Mat FloatDepthMmToMat(const float* depth, std::uint32_t dw, std::uint32_t dh) {
    if (!depth || dw == 0 || dh == 0) {
        return {};
    }
    cv::Mat m(static_cast<int>(dh), static_cast<int>(dw), CV_32F);
    std::memcpy(m.data, depth,
                static_cast<size_t>(dw) * static_cast<size_t>(dh) * sizeof(float));
    return m;
}

bool ResolveAirFrameDepthDims(const AIRFrameData& f, std::uint32_t& out_w, std::uint32_t& out_h) {
    if (!f.depth_map) {
        return false;
    }

    const auto size_ok = [&](std::uint32_t w, std::uint32_t h) -> bool {
        if (w == 0 || h == 0) {
            return false;
        }
        if (f.depth_map_size == 0) {
            return true;
        }
        return static_cast<size_t>(w) * static_cast<size_t>(h) == f.depth_map_size;
    };

    std::uint32_t cand_w = f.depth_width ? f.depth_width : f.width;
    std::uint32_t cand_h = f.depth_height ? f.depth_height : f.height;
    if (size_ok(cand_w, cand_h)) {
        out_w = cand_w;
        out_h = cand_h;
        return true;
    }

    const std::uint32_t tw = f.texture_width;
    const std::uint32_t th = f.texture_height;
    if (size_ok(tw, th)) {
        out_w = tw;
        out_h = th;
        return true;
    }

    if (f.depth_width > 0 && f.depth_height > 0 && size_ok(f.depth_width, f.depth_height)) {
        out_w = f.depth_width;
        out_h = f.depth_height;
        return true;
    }

    if (f.depth_map_size > 0) {
        const size_t n = f.depth_map_size;
        const size_t r = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
        if (r > 0 && r * r == n) {
            out_w = static_cast<std::uint32_t>(r);
            out_h = static_cast<std::uint32_t>(r);
            return true;
        }
    }

    return false;
}

cv::Mat AirFrameDepthMmToMat(const AIRFrameData& f) {
    std::uint32_t w = 0;
    std::uint32_t h = 0;
    if (!ResolveAirFrameDepthDims(f, w, h)) {
        return {};
    }
    return FloatDepthMmToMat(f.depth_map, w, h);
}

cv::Mat AirFrameTextureToBgr(const AIRFrameData& f) {
    const std::uint32_t tw = f.texture_width ? f.texture_width : f.width;
    const std::uint32_t th = f.texture_height ? f.texture_height : f.height;
    if (!f.texture_map || tw == 0 || th == 0) {
        return {};
    }
    const size_t expected_gray = static_cast<size_t>(tw) * static_cast<size_t>(th);
    const size_t expected_rgb  = expected_gray * 3u;

    if (f.texture_channel_count == 1u || f.texture_map_size == expected_gray) {
        cv::Mat gray(static_cast<int>(th), static_cast<int>(tw), CV_8UC1,
                     const_cast<std::uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (f.texture_channel_count == 3u || f.texture_map_size == expected_rgb) {
        cv::Mat rgb(static_cast<int>(th), static_cast<int>(tw), CV_8UC3,
                    const_cast<std::uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }

    if (f.texture_map_size == expected_gray) {
        cv::Mat gray(static_cast<int>(th), static_cast<int>(tw), CV_8UC1,
                     const_cast<std::uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (f.texture_map_size == expected_rgb) {
        cv::Mat rgb(static_cast<int>(th), static_cast<int>(tw), CV_8UC3,
                    const_cast<std::uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }

    return {};
}

void WriteDepthPngMm(const cv::Mat& depth_mm_aligned, const std::string& path) {
    CV_Assert(depth_mm_aligned.type() == CV_32F);
    cv::Mat u16(depth_mm_aligned.size(), CV_16U);
    for (int v = 0; v < depth_mm_aligned.rows; ++v) {
        const float* pr = depth_mm_aligned.ptr<float>(v);
        std::uint16_t* pw = u16.ptr<std::uint16_t>(v);
        for (int u = 0; u < depth_mm_aligned.cols; ++u) {
            float d = pr[u];
            if (!(d > 0) || !std::isfinite(d)) {
                pw[u] = 0;
            } else {
                const float c = std::min(d, 65535.f);
                pw[u]         = static_cast<std::uint16_t>(c + 0.5f);
            }
        }
    }
    cv::imwrite(path, u16);
}

cv::Mat ResizeForPreview(const cv::Mat& bgr, int max_width) {
    if (bgr.empty() || bgr.cols <= max_width) {
        return bgr;
    }
    cv::Mat out;
    const double scale = static_cast<double>(max_width) / static_cast<double>(bgr.cols);
    cv::resize(bgr, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

cv::Mat MakeInteractiveIdleView(const cv::Mat& last_bgr) {
    constexpr char kHint[] = "s: capture+save  f: capture+pcd  q: quit";
    if (!last_bgr.empty()) {
        cv::Mat v = last_bgr.clone();
        const int th = v.rows;
        cv::putText(v,
                    kHint,
                    cv::Point(10, std::max(24, std::min(th - 8, 28))),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 255, 0),
                    1,
                    cv::LINE_AA);
        return v;
    }
    cv::Mat v(480, 640, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::putText(v, kHint, cv::Point(20, 220), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::putText(v,
                "Each s/f triggers ONE new scan (several sec).",
                cv::Point(20, 260),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(200, 200, 200),
                1,
                cv::LINE_AA);
    return v;
}

void ApplyEnvScannerAddress(AirScannerConfig& cfg) {
    if (const char* ip = std::getenv("SCANNER_IP")) {
        if (ip[0]) {
            cfg.ip = ip;
        }
    }
    if (const char* ps = std::getenv("SCANNER_PORT")) {
        std::uint16_t p = cfg.port;
        if (ParsePort(ps, p)) {
            cfg.port = p;
        }
    }
    if (const char* pj = std::getenv("SCANNER_PARAMS_PATH")) {
        if (pj[0]) {
            cfg.scanner_param_path = pj;
        }
    }
}

std::string ResolveScannerParamsPath(const AirScannerConfig& cfg) {
    if (!cfg.scanner_param_path.empty()) {
        return cfg.scanner_param_path;
    }
    static const char* kCandidates[] = {
        "scanner_a/ScannerConfig/scanner0_params.json",
        "ScannerConfig/scanner0_params.json",
    };
    for (const char* p : kCandidates) {
        if (std::filesystem::exists(p)) {
            return std::string(p);
        }
    }
    return {};
}

}  // namespace odt
