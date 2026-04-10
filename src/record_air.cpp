#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "odt/object_dataset_tools.hpp"
#include "odt/utils/air_capture_helpers.hpp"
#include "odt/utils/air_record_scanner.hpp"
#include "odt/utils/camera_utils.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace odt {

int ObjectDatasetTools::recordAir(int argc, char** argv) {
    if (argc < 2) {
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
    if (interactive) {
        if (argc > 6) {
            std::cerr << "Too many arguments (interactive mode).\n";
            return 1;
        }
        ParseTailNetworkParams(argc, argv, 3, cfg);
    } else if (timed_keyword) {
        if (argc < 5) {
            std::cerr << "Timed mode with keyword 'timed' needs: record_air <out> timed <record_len_sec> <countdown_sec> [ip] [port] [params]\n";
            return 1;
        }
        if (argc > 8) {
            std::cerr << "Too many arguments (timed mode).\n";
            return 1;
        }
        if (!ParsePositiveInt(argv[3], record_len_sec)) {
            std::cerr << "Invalid record_len_sec: " << (argv[3] ? argv[3] : "") << "\n";
            return 1;
        }
        if (!ParsePositiveInt(argv[4], countdown_sec)) {
            std::cerr << "Invalid countdown_sec: " << (argv[4] ? argv[4] : "") << "\n";
            return 1;
        }
        ParseTailNetworkParams(argc, argv, 5, cfg);
    } else {
        if (argc > 7) {
            std::cerr << "Too many arguments.\n";
            return 1;
        }
        if (argc >= 3) {
            if (!ParsePositiveInt(argv[2], record_len_sec)) {
                std::cerr << "Invalid record_len_sec: " << (argv[2] ? argv[2] : "") << "\n";
                return 1;
            }
        }
        if (argc >= 4) {
            if (!ParsePositiveInt(argv[3], countdown_sec)) {
                std::cerr << "Invalid countdown_sec: " << (argv[3] ? argv[3] : "") << "\n";
                return 1;
            }
        }
        ParseTailNetworkParams(argc, argv, 4, cfg);
    }

    std::string out = argv[1];
    while (!out.empty() && (out.back() == '/' || out.back() == '\\')) {
        out.pop_back();
    }
    std::replace(out.begin(), out.end(), '\\', '/');
    EnsureLinemodCaptureDirs(out);
    const std::string pointcloud_dir = out + "/pointclouds";
    if (interactive) {
        std::filesystem::create_directories(pointcloud_dir);
    }

    AirRecordScanner scanner(cfg);
    if (!scanner.Initialize()) {
        std::cerr << "AirRecordScanner::Initialize failed.\n";
        scanner.PrintLastError();
        return 1;
    }
    if (!scanner.Connect()) {
        std::cerr << "AirRecordScanner::Connect failed (target " << cfg.ip << ":" << cfg.port << ").\n";
        scanner.PrintLastError();
        return 1;
    }

    const std::string params_json = ResolveScannerParamsPath(cfg);
    if (params_json.empty()) {
        std::cerr
            << "No AIR scanner params JSON. Set SCANNER_PARAMS_PATH, pass 7th arg, or place e.g. scanner_a/ScannerConfig/scanner0_params.json (see scanner_a ImportConfig flow).\n";
        scanner.Disconnect();
        return 1;
    }
    if (!scanner.ImportConfig(params_json)) {
        std::cerr << "ImportConfig failed (path: " << params_json << ").\n";
        scanner.PrintLastError();
        scanner.Disconnect();
        return 1;
    }

    AIRScannerParams params{};
    if (!scanner.GetCameraParams(params)) {
        std::cerr << "GetCameraParams failed.\n";
        scanner.PrintLastError();
        scanner.Disconnect();
        return 1;
    }

    AIRCameraIntrinsics rgb = params.intrinsics_texture;
    if ((rgb.width == 0 || rgb.fx == 0.0) && params.intrinsics_depth.width > 0 &&
        params.intrinsics_depth.fx != 0.0) {
        rgb = params.intrinsics_depth;
    }
    nlohmann::json j;
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

    if (interactive) {
        constexpr int kPreviewMaxWidth = 960;
        bool          session_done     = false;
        cv::Mat       last_bgr_for_idle;

        while (!session_done) {
            cv::Mat idle_view = MakeInteractiveIdleView(last_bgr_for_idle);
            cv::imshow("COLOR", ResizeForPreview(idle_view, kPreviewMaxWidth));
            cv::waitKey(1);

            const int cmd = cv::waitKey(0);
            if (cmd == 'q' || cmd == 'Q' || cmd == 27) {
                session_done = true;
                break;
            }
            if (cmd != 's' && cmd != 'S' && cmd != 'f' && cmd != 'F') {
                continue;
            }

            std::cout << "Scanning (please wait a few seconds)...\n" << std::flush;

            AIRFrameData frame{};
            if (!scanner.CaptureFrame(frame)) {
                std::cerr << "CaptureFrame failed.\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            cv::Mat        depth = AirFrameDepthMmToMat(frame);
            cv::Mat        tex_bgr = AirFrameTextureToBgr(frame);

            if (tex_bgr.empty()) {
                scanner.ReleaseFrame(frame);
                std::cerr << "Empty texture, skipped.\n";
                continue;
            }

            cv::Mat depth_aligned;
            if (depth.empty()) {
                depth_aligned = cv::Mat(tex_bgr.size(), CV_32F, cv::Scalar(0));
            } else if (depth.size() != tex_bgr.size()) {
                cv::resize(depth, depth_aligned, tex_bgr.size(), 0, 0, cv::INTER_NEAREST);
            } else {
                depth_aligned = depth;
            }

            last_bgr_for_idle = tex_bgr.clone();
            cv::Mat preview_after = ResizeForPreview(tex_bgr.clone(), kPreviewMaxWidth);
            cv::imshow("COLOR", preview_after);
            cv::waitKey(1);

            if (cmd == 's' || cmd == 'S') {
                const std::string jpg = out + "/JPEGImages/" + std::to_string(frame_index) + ".jpg";
                const std::string png = out + "/depth/" + std::to_string(frame_index) + ".png";
                cv::imwrite(jpg, tex_bgr);
                WriteDepthPngMm(depth_aligned, png);
                std::cout << "Saved frame " << frame_index << " (jpg+png)\n" << std::flush;
                ++frame_index;
            } else if (cmd == 'f' || cmd == 'F') {
                if (SaveFusionPointCloudGray(pointcloud_dir, pc_index, depth_aligned,
                                             tex_bgr, intr)) {
                    std::cout << "Saved point cloud #" << (pc_index - 1) << " to "
                              << pointcloud_dir << "\n" << std::flush;
                } else {
                    std::cerr << "Fusion point cloud failed (depth/color size or empty cloud).\n";
                }
            }

            scanner.ReleaseFrame(frame);
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();

        while (true) {
            AIRFrameData frame{};
            if (!scanner.CaptureFrame(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            cv::Mat        depth = AirFrameDepthMmToMat(frame);
            cv::Mat        tex_bgr = AirFrameTextureToBgr(frame);

            if (tex_bgr.empty()) {
                scanner.ReleaseFrame(frame);
                continue;
            }

            cv::Mat depth_aligned;
            if (depth.empty()) {
                depth_aligned = cv::Mat(tex_bgr.size(), CV_32F, cv::Scalar(0));
            } else if (depth.size() != tex_bgr.size()) {
                cv::resize(depth, depth_aligned, tex_bgr.size(), 0, 0, cv::INTER_NEAREST);
            } else {
                depth_aligned = depth;
            }

            const auto elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            int cd = 0;
            if (elapsed < countdown_sec) {
                cd = countdown_sec - static_cast<int>(elapsed);
            } else if (elapsed < countdown_sec + record_len_sec) {
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
                cv::putText(show, txt, org, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale,
                            cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
            };
            if (elapsed < countdown_sec) {
                draw_center_number(std::to_string(cd));
            } else if (elapsed < countdown_sec + record_len_sec) {
                const int left = countdown_sec + record_len_sec - static_cast<int>(elapsed);
                draw_center_number(std::to_string(left));
            }

            constexpr int kPreviewMaxWidth = 960;
            cv::Mat         preview          = ResizeForPreview(show, kPreviewMaxWidth);
            cv::imshow("COLOR", preview);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                scanner.ReleaseFrame(frame);
                break;
            }
            if (elapsed >= countdown_sec + record_len_sec) {
                scanner.ReleaseFrame(frame);
                break;
            }

            scanner.ReleaseFrame(frame);
        }
    }

    cv::destroyAllWindows();
    scanner.Disconnect();
    std::cout << "Saved " << frame_index << " frames to " << out;
    if (interactive) {
        std::cout << " (" << pc_index << " point clouds in " << pointcloud_dir << ")";
    }
    std::cout << "\n";
    return 0;
}

}  // namespace odt
