#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "odt/air_record_scanner.hpp"
#include "odt/camera_utils.hpp"
#include "odt/odt_registration.hpp"
#include "odt/register_scene_fusion.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cctype>
#include <iostream>
#include <string>
#include <thread>

namespace {

bool EqCi(const char* a, const char* b) {
    if(!a || !b) {
        return a == b;
    }
    while(*a && *b) {
        if(std::tolower(static_cast<unsigned char>(*a)) != std::tolower(static_cast<unsigned char>(*b))) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

/// 深度(mm) + 彩色 BGR → 灰度着色点云，写入 pointclouds/{index}.ply
bool SaveFusionPointCloudGray(const std::string& pointcloud_dir,
                              int&                   pc_index,
                              const cv::Mat&         depth_mm_aligned,
                              const cv::Mat&         tex_bgr,
                              const odt::CameraIntrinsics& intr) {
    if(tex_bgr.empty() || depth_mm_aligned.empty() || tex_bgr.size() != depth_mm_aligned.size()) {
        return false;
    }
    cv::Mat gray8;
    cv::cvtColor(tex_bgr, gray8, cv::COLOR_BGR2GRAY);
    cv::Mat gray_bgr;
    cv::cvtColor(gray8, gray_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat xyz;
    odt::DepthFloatMmToPointCloud(depth_mm_aligned, intr, xyz);
    auto pcd = odt::BuildPcdFromRgbd(gray_bgr, xyz, 0.001, false);
    if(!pcd || pcd->points_.empty()) {
        return false;
    }
    const std::string path = pointcloud_dir + "/" + std::to_string(pc_index++) + ".ply";
    return odt::WritePointCloudPlyBinary(path, *pcd);
}


bool ParsePositiveInt(const char* s, int& out) {
    if(!s || !*s) {
        return false;
    }
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if(end == s || *end != '\0' || v <= 0 || v > 86400) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool ParsePort(const char* s, uint16_t& out) {
    if(!s || !*s) {
        return false;
    }
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if(end == s || *end != '\0' || v == 0UL || v > 65535UL) {
        return false;
    }
    out = static_cast<uint16_t>(v);
    return true;
}

void ParseTailNetworkParams(int argc, char** argv, int i0, odt::AirScannerConfig& cfg) {
    if(argc > i0 && argv[i0] && argv[i0][0]) {
        cfg.ip = argv[i0];
    }
    if(argc > i0 + 1) {
        uint16_t p = cfg.port;
        if(ParsePort(argv[i0 + 1], p)) {
            cfg.port = p;
        }
    }
    if(argc > i0 + 2 && argv[i0 + 2] && argv[i0 + 2][0]) {
        cfg.scanner_param_path = argv[i0 + 2];
    }
}

void EnsureDirs(const std::string& base) {
    std::filesystem::create_directories(base + "/JPEGImages");
    std::filesystem::create_directories(base + "/depth");
}

cv::Mat FloatDepthMmToMat(const float* depth, uint32_t dw, uint32_t dh) {
    if(!depth || dw == 0 || dh == 0) {
        return {};
    }
    cv::Mat m(static_cast<int>(dh), static_cast<int>(dw), CV_32F);
    std::memcpy(m.data, depth, static_cast<size_t>(dw) * static_cast<size_t>(dh) * sizeof(float));
    return m;
}

/// 与 scanner_a SaveTextureMap 一致：灰度则转 BGR，RGB 则 RGB→BGR
cv::Mat AirFrameTextureToBgr(const AIRFrameData& f) {
    const uint32_t tw = f.texture_width ? f.texture_width : f.width;
    const uint32_t th = f.texture_height ? f.texture_height : f.height;
    if(!f.texture_map || tw == 0 || th == 0) {
        return {};
    }
    const size_t expected_gray = static_cast<size_t>(tw) * static_cast<size_t>(th);
    const size_t expected_rgb    = expected_gray * 3u;

    if(f.texture_channel_count == 1u || f.texture_map_size == expected_gray) {
        cv::Mat gray(static_cast<int>(th), static_cast<int>(tw), CV_8UC1,
                     const_cast<uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if(f.texture_channel_count == 3u || f.texture_map_size == expected_rgb) {
        cv::Mat rgb(static_cast<int>(th), static_cast<int>(tw), CV_8UC3,
                    const_cast<uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }

    if(f.texture_map_size == expected_gray) {
        cv::Mat gray(static_cast<int>(th), static_cast<int>(tw), CV_8UC1,
                     const_cast<uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if(f.texture_map_size == expected_rgb) {
        cv::Mat rgb(static_cast<int>(th), static_cast<int>(tw), CV_8UC3,
                    const_cast<uint8_t*>(f.texture_map));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }

    return {};
}

void WriteDepthPngMm(const cv::Mat& depth_mm_aligned, const std::string& path) {
    CV_Assert(depth_mm_aligned.type() == CV_32F);
    cv::Mat u16(depth_mm_aligned.size(), CV_16U);
    for(int v = 0; v < depth_mm_aligned.rows; ++v) {
        const float* pr = depth_mm_aligned.ptr<float>(v);
        uint16_t*    pw = u16.ptr<uint16_t>(v);
        for(int u = 0; u < depth_mm_aligned.cols; ++u) {
            float d = pr[u];
            if(!(d > 0) || !std::isfinite(d)) {
                pw[u] = 0;
            } else {
                const float c = std::min(d, 65535.f);
                pw[u]         = static_cast<uint16_t>(c + 0.5f);
            }
        }
    }
    cv::imwrite(path, u16);
}

/// 预览窗口用，避免高分辨率整屏显示不全（写入文件仍用原始分辨率）
cv::Mat ResizeForPreview(const cv::Mat& bgr, int max_width) {
    if(bgr.empty() || bgr.cols <= max_width) {
        return bgr;
    }
    cv::Mat out;
    const double scale = static_cast<double>(max_width) / static_cast<double>(bgr.cols);
    cv::resize(bgr, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

/// interactive 空闲界面：无上一帧时灰底提示；有上一帧则叠字（避免画面永远不变）
cv::Mat MakeInteractiveIdleView(const cv::Mat& last_bgr) {
    constexpr char kHint[] = "s: capture+save  f: capture+pcd  q: quit";
    if(!last_bgr.empty()) {
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
    cv::putText(v, kHint, cv::Point(20, 220), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
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

void ApplyEnvScannerAddress(odt::AirScannerConfig& cfg) {
    if(const char* ip = std::getenv("SCANNER_IP")) {
        if(ip[0]) {
            cfg.ip = ip;
        }
    }
    if(const char* ps = std::getenv("SCANNER_PORT")) {
        uint16_t p = cfg.port;
        if(ParsePort(ps, p)) {
            cfg.port = p;
        }
    }
    if(const char* pj = std::getenv("SCANNER_PARAMS_PATH")) {
        if(pj[0]) {
            cfg.scanner_param_path = pj;
        }
    }
}

/// 与 scanner_a 一致：须先有参数 JSON 路径，再 AIR_ImportConfig，内参才有效
std::string ResolveScannerParamsPath(const odt::AirScannerConfig& cfg) {
    if(!cfg.scanner_param_path.empty()) {
        return cfg.scanner_param_path;
    }
    static const char* kCandidates[] = {
        "scanner_a/ScannerConfig/scanner0_params.json",
        "ScannerConfig/scanner0_params.json",
    };
    for(const char* p : kCandidates) {
        if(std::filesystem::exists(p)) {
            return std::string(p);
        }
    }
    return {};
}

}  // namespace

namespace odt {

int RunRecordAir(int argc, char** argv) {
    if(argc < 2) {
        std::cerr
            << "Usage:\n"
            << "  Timed (定时长): record_air <output_dir> [timed] <record_len_sec> <countdown_sec> [ip] [port] [scanner_params.json]\n"
            << "    省略关键字 timed 时与旧版相同：record_air <out> [record_len] [countdown] [ip] [port] [params]\n"
            << "  Interactive (按键): record_air <output_dir> interactive [ip] [port] [scanner_params.json]\n"
            << "    先按 s（保存图+深度）或 f（点云）：每次都会重新扫描一次再写入；q 退出\n"
            << "  环境变量：SCANNER_IP, SCANNER_PORT, SCANNER_PARAMS_PATH\n";
        return 1;
    }

    const bool interactive  = (argc >= 3 && EqCi(argv[2], "interactive"));
    const bool timed_keyword = (argc >= 3 && EqCi(argv[2], "timed"));

    int              record_len_sec = 40;
    int              countdown_sec  = 5;
    AirScannerConfig cfg;

    ApplyEnvScannerAddress(cfg);

    if(interactive) {
        if(argc > 6) {
            std::cerr << "Too many arguments (interactive mode).\n";
            return 1;
        }
        ParseTailNetworkParams(argc, argv, 3, cfg);
    } else if(timed_keyword) {
        if(argc < 5) {
            std::cerr << "Timed mode with keyword 'timed' needs: record_air <out> timed <record_len_sec> <countdown_sec> [ip] [port] [params]\n";
            return 1;
        }
        if(argc > 8) {
            std::cerr << "Too many arguments (timed mode).\n";
            return 1;
        }
        if(!ParsePositiveInt(argv[3], record_len_sec)) {
            std::cerr << "Invalid record_len_sec: " << (argv[3] ? argv[3] : "") << "\n";
            return 1;
        }
        if(!ParsePositiveInt(argv[4], countdown_sec)) {
            std::cerr << "Invalid countdown_sec: " << (argv[4] ? argv[4] : "") << "\n";
            return 1;
        }
        ParseTailNetworkParams(argc, argv, 5, cfg);
    } else {
        if(argc > 7) {
            std::cerr << "Too many arguments.\n";
            return 1;
        }
        if(argc >= 3) {
            if(!ParsePositiveInt(argv[2], record_len_sec)) {
                std::cerr << "Invalid record_len_sec: " << (argv[2] ? argv[2] : "") << "\n";
                return 1;
            }
        }
        if(argc >= 4) {
            if(!ParsePositiveInt(argv[3], countdown_sec)) {
                std::cerr << "Invalid countdown_sec: " << (argv[3] ? argv[3] : "") << "\n";
                return 1;
            }
        }
        ParseTailNetworkParams(argc, argv, 4, cfg);
    }

    std::string out = argv[1];
    while(!out.empty() && (out.back() == '/' || out.back() == '\\')) {
        out.pop_back();
    }
    std::replace(out.begin(), out.end(), '\\', '/');
    EnsureDirs(out);
    const std::string pointcloud_dir = out + "/pointclouds";
    if(interactive) {
        std::filesystem::create_directories(pointcloud_dir);
    }

    AirRecordScanner scanner(cfg);
    if(!scanner.Initialize()) {
        std::cerr << "AirRecordScanner::Initialize failed.\n";
        scanner.PrintLastError();
        return 1;
    }
    if(!scanner.Connect()) {
        std::cerr << "AirRecordScanner::Connect failed (target " << cfg.ip << ":" << cfg.port << ").\n";
        scanner.PrintLastError();
        return 1;
    }

    const std::string params_json = ResolveScannerParamsPath(cfg);
    if(params_json.empty()) {
        std::cerr
            << "No AIR scanner params JSON. Set SCANNER_PARAMS_PATH, pass 7th arg, or place e.g. scanner_a/ScannerConfig/scanner0_params.json (see scanner_a ImportConfig flow).\n";
        scanner.Disconnect();
        return 1;
    }
    if(!scanner.ImportConfig(params_json)) {
        std::cerr << "ImportConfig failed (path: " << params_json << ").\n";
        scanner.PrintLastError();
        scanner.Disconnect();
        return 1;
    }

    AIRScannerParams params{};
    if(!scanner.GetCameraParams(params)) {
        std::cerr << "GetCameraParams failed.\n";
        scanner.PrintLastError();
        scanner.Disconnect();
        return 1;
    }

    // 彩色图对齐用纹理内参；若仍为 0 则回退深度内参（部分固件/配置下纹理字段未填）
    AIRCameraIntrinsics rgb = params.intrinsics_texture;
    if((rgb.width == 0 || rgb.fx == 0.0) && params.intrinsics_depth.width > 0 && params.intrinsics_depth.fx != 0.0) {
        rgb = params.intrinsics_depth;
    }
    nlohmann::json             j;
    j["fx"]           = rgb.fx;
    j["fy"]           = rgb.fy;
    j["ppx"]          = rgb.cx;
    j["ppy"]          = rgb.cy;
    j["height"]       = rgb.height;
    j["width"]        = rgb.width;
    j["depth_scale"]  = 0.001;
    j["camera_model"] = "air_scanner";
    std::ofstream ofs(out + "/intrinsics.json");
    ofs << j.dump(4);
    ofs.close();

    CameraIntrinsics intr{};
    intr.fx          = rgb.fx;
    intr.fy          = rgb.fy;
    intr.ppx         = rgb.cx;
    intr.ppy         = rgb.cy;
    intr.width       = rgb.width;
    intr.height      = rgb.height;
    intr.depth_scale = 0.001;

    int frame_index = 0;
    int pc_index    = 0;

    if(interactive) {
        // 先选 s/f 再 Capture：每次按键都重新扫描，避免同一 SDK 缓冲被多次保存成相同内容。
        // 空闲界面显示上一帧或提示图，扫描结束后更新预览。
        constexpr int   kPreviewMaxWidth = 960;
        bool              session_done   = false;
        cv::Mat           last_bgr_for_idle;

        while(!session_done) {
            cv::Mat idle_view = MakeInteractiveIdleView(last_bgr_for_idle);
            cv::imshow("COLOR", ResizeForPreview(idle_view, kPreviewMaxWidth));
            cv::waitKey(1);

            const int cmd = cv::waitKey(0);
            if(cmd == 'q' || cmd == 'Q' || cmd == 27) {
                session_done = true;
                break;
            }
            if(cmd != 's' && cmd != 'S' && cmd != 'f' && cmd != 'F') {
                continue;
            }

            std::cout << "Scanning (please wait a few seconds)...\n" << std::flush;

            AIRFrameData frame{};
            if(!scanner.CaptureFrame(frame)) {
                std::cerr << "CaptureFrame failed.\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            const uint32_t dw = frame.depth_width ? frame.depth_width : frame.width;
            const uint32_t dh = frame.depth_height ? frame.depth_height : frame.height;
            cv::Mat        depth = FloatDepthMmToMat(frame.depth_map, dw, dh);
            cv::Mat        tex_bgr = AirFrameTextureToBgr(frame);

            if(tex_bgr.empty()) {
                scanner.ReleaseFrame(frame);
                std::cerr << "Empty texture, skipped.\n";
                continue;
            }

            cv::Mat depth_aligned;
            if(depth.empty()) {
                depth_aligned = cv::Mat(tex_bgr.size(), CV_32F, cv::Scalar(0));
            } else if(depth.size() != tex_bgr.size()) {
                cv::resize(depth, depth_aligned, tex_bgr.size(), 0, 0, cv::INTER_NEAREST);
            } else {
                depth_aligned = depth;
            }

            last_bgr_for_idle = tex_bgr.clone();
            cv::Mat preview_after = ResizeForPreview(tex_bgr.clone(), kPreviewMaxWidth);
            cv::imshow("COLOR", preview_after);
            cv::waitKey(1);

            if(cmd == 's' || cmd == 'S') {
                const std::string jpg = out + "/JPEGImages/" + std::to_string(frame_index) + ".jpg";
                const std::string png = out + "/depth/" + std::to_string(frame_index) + ".png";
                cv::imwrite(jpg, tex_bgr);
                WriteDepthPngMm(depth_aligned, png);
                std::cout << "Saved frame " << frame_index << " (jpg+png)\n" << std::flush;
                ++frame_index;
            } else if(cmd == 'f' || cmd == 'F') {
                if(SaveFusionPointCloudGray(pointcloud_dir, pc_index, depth_aligned, tex_bgr, intr)) {
                    std::cout << "Saved point cloud #" << (pc_index - 1) << " to " << pointcloud_dir << "\n" << std::flush;
                } else {
                    std::cerr << "Fusion point cloud failed (depth/color size or empty cloud).\n";
                }
            }

            scanner.ReleaseFrame(frame);
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();

        while(true) {
            AIRFrameData frame{};
            if(!scanner.CaptureFrame(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            const uint32_t dw = frame.depth_width ? frame.depth_width : frame.width;
            const uint32_t dh = frame.depth_height ? frame.depth_height : frame.height;
            cv::Mat        depth = FloatDepthMmToMat(frame.depth_map, dw, dh);
            cv::Mat        tex_bgr = AirFrameTextureToBgr(frame);

            if(tex_bgr.empty()) {
                scanner.ReleaseFrame(frame);
                continue;
            }

            cv::Mat depth_aligned;
            if(depth.empty()) {
                depth_aligned = cv::Mat(tex_bgr.size(), CV_32F, cv::Scalar(0));
            } else if(depth.size() != tex_bgr.size()) {
                cv::resize(depth, depth_aligned, tex_bgr.size(), 0, 0, cv::INTER_NEAREST);
            } else {
                depth_aligned = depth;
            }

            const auto elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            int cd = 0;
            if(elapsed < countdown_sec) {
                cd = countdown_sec - static_cast<int>(elapsed);
            } else if(elapsed < countdown_sec + record_len_sec) {
                const std::string jpg = out + "/JPEGImages/" + std::to_string(frame_index) + ".jpg";
                const std::string png = out + "/depth/" + std::to_string(frame_index) + ".png";
                cv::imwrite(jpg, tex_bgr);
                WriteDepthPngMm(depth_aligned, png);
                ++frame_index;
            }

            cv::Mat      show = tex_bgr.clone();
            const int    tw   = show.cols;
            const int    th   = show.rows;
            const double font_scale =
                std::max(1.5, static_cast<double>(std::min(tw, th)) / 200.0);
            const int thickness = std::max(2, static_cast<int>(font_scale * 1.5));
            auto draw_center_number = [&](const std::string& txt) {
                int              baseline = 0;
                const cv::Size ts =
                    cv::getTextSize(txt, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, thickness, &baseline);
                const cv::Point org((tw - ts.width) / 2, (th + ts.height) / 2);
                cv::putText(show, txt, org, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, cv::Scalar(0, 0, 255),
                            thickness, cv::LINE_AA);
            };
            if(elapsed < countdown_sec) {
                draw_center_number(std::to_string(cd));
            } else if(elapsed < countdown_sec + record_len_sec) {
                const int left = countdown_sec + record_len_sec - static_cast<int>(elapsed);
                draw_center_number(std::to_string(left));
            }

            constexpr int kPreviewMaxWidth = 960;
            cv::Mat         preview          = ResizeForPreview(show, kPreviewMaxWidth);
            cv::imshow("COLOR", preview);

            const int key = cv::waitKey(1);
            if(key == 'q' || key == 'Q') {
                scanner.ReleaseFrame(frame);
                break;
            }
            if(elapsed >= countdown_sec + record_len_sec) {
                scanner.ReleaseFrame(frame);
                break;
            }

            scanner.ReleaseFrame(frame);
        }
    }

    cv::destroyAllWindows();
    scanner.Disconnect();
    std::cout << "Saved " << frame_index << " frames to " << out;
    if(interactive) {
        std::cout << " (" << pc_index << " point clouds in " << pointcloud_dir << ")";
    }
    std::cout << "\n";
    return 0;
}

}  // namespace odt
