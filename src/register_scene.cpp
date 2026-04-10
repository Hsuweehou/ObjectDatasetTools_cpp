#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <open3d/Open3D.h>

#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "odt/async_progress.hpp"
#include "odt/camera_utils.hpp"
#include "odt/npy_io.hpp"
#include "odt/register_scene_fusion.hpp"
#include "odt/registration_params.hpp"

namespace {

int CountJpg(const std::string& jpeg_dir) {
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

void WriteRegisterSceneStepManifest(const std::string& path,
                                    const std::string& seq_path,
                                    int recon_iv,
                                    int label_iv,
                                    double voxel_r,
                                    double inlier_r,
                                    int n_iter,
                                    int total_jpg) {
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "register_scene step export\n"
      << "sequence: " << seq_path << "\n"
      << "JPEGImages count: " << total_jpg << "\n"
      << "reconstruction_interval: " << recon_iv << "\n"
      << "label_interval: " << label_iv << "\n"
      << "voxel_r (merge): " << voxel_r << "\n"
      << "inlier_r (vote nn): " << inlier_r << "\n"
      << "loaded frames (n_iter): " << n_iter << "\n"
      << "\nPLY files:\n"
      << "  step01_frame_first_pre_transform.ply  first frame, camera frame (before T)\n"
      << "  step01_frame_last_pre_transform.ply   last frame, camera frame (before T)\n"
      << "  step01_frame_first.ply  first frame after T (world)\n"
      << "  step01_frame_last.ply   last frame after T (world)\n"
      << "  step02_merged_pre_vote.ply  KDTree merge (geometric union; same as registeredScene.ply)\n"
      << "  step03_vote_filtered.ply    optional: vote>0 only (stricter overlap; for comparison)\n"
      << "  (registeredScene.ply is the full union from step 2, not vote-filtered)\n";
}

std::vector<std::string> GlobLinemodFolders(const std::string& repo_root) {
    std::vector<std::string> out;
    const std::filesystem::path root = std::filesystem::path(repo_root) / "LINEMOD";
    if (!std::filesystem::exists(root)) {
        return out;
    }
    for (auto& p : std::filesystem::directory_iterator(root)) {
        if (p.is_directory()) {
            std::string s = p.path().generic_string();
            if (!s.empty() && s.back() != '/') {
                s += '/';
            }
            out.push_back(s);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool EndsWithSlash(const std::string& s) {
    return !s.empty() && (s.back() == '/' || s.back() == '\\');
}

std::string NormalizeFolderArg(const std::string& arg) {
    std::string s = arg;
    std::replace(s.begin(), s.end(), '\\', '/');
    if (!EndsWithSlash(s)) {
        s += '/';
    }
    return s;
}

}  // namespace

namespace odt {

// -----------------------------------------------------------------------------
// register_scene：将 LINEMOD 序列的多帧 RGB-D 配准到统一世界系并融合为 registeredScene.ply
//
// 依赖（每个序列目录下）：
//   - intrinsics.json     相机内参与 depth_scale（反投影）
//   - transforms.npy      每帧 4x4 世界变换（与 compute_gt_poses 一致）
//   - JPEGImages/*.jpg 与 depth/*.png  同索引成对
//
// 参数来源：registration_params.hpp（kReconstructionInterval、kVoxelR 等）
// -----------------------------------------------------------------------------

int RunRegisterScene(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ... register_scene <path|all>\n"
                  << "  path: e.g. LINEMOD/object/  (run from repo root)\n"
                  << "  all:  process all LINEMOD/*/\n";
        return 1;
    }

    // ---------- 解析待处理目录：单个 LINEMOD/<name>/ 或 all 枚举全部 ----------
    std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::string> folders;

    const std::string arg1 = argv[1];
    if (arg1 == "all") {
        folders = GlobLinemodFolders(cwd.string());
        if (folders.empty()) {
            std::cerr << "No LINEMOD subfolders in current directory.\n";
            return 1;
        }
    } else {
        std::string f = NormalizeFolderArg(arg1);
        bool ok = false;
        const auto globs = GlobLinemodFolders(cwd.string());
        for (const auto& g : globs) {
            if (g == f || NormalizeFolderArg(g) == f) {
                ok = true;
                break;
            }
        }
        if (!ok && std::filesystem::exists(std::filesystem::path(f))) {
            ok = true;
        }
        if (!ok) {
            std::cerr << "Invalid folder (expected LINEMOD/<name>/).\n";
            return 1;
        }
        folders.push_back(f);
    }

    // 融合与投票半径：voxel_r 控制「与已有合并点距离超过此值的点视为新表面并追加」；
    // inlier_r 用于跨帧最近邻投票统计（仅影响可选导出的 vote 筛选结果）。
    const int recon_iv = odt::kReconstructionInterval;
    const int label_iv = odt::kLabelInterval;
    const double voxel_r = odt::kVoxelR;
    const double inlier_r = voxel_r * 10;

    odt::AsyncProgress progress;

    for (const std::string& path : folders) {
        std::cout << "[register_scene] 序列目录: " << path << "\n";

        // ----- 输入：内参（反投影深度） -----
        odt::CameraIntrinsics intr;
        if (!odt::LoadIntrinsicsJson(path + "intrinsics.json", intr)) {
            std::cerr << "intrinsics.json missing\n";
            continue;
        }

        // ----- 输入：位姿表 transforms.npy，形状 (N, 4, 4) 或等价展平 -----
        nc::NdArray<double> transforms;
        std::size_t         n0 = 0, n1 = 0, n2 = 0;
        if (!odt::load_ndarray_npy(path + "transforms.npy", transforms, n0, n1, n2) ||
            n1 != 4 || n2 != 4) {
            std::cerr << "transforms.npy invalid\n";
            continue;
        }

        const int total_jpg = CountJpg(path + "JPEGImages");
        if (total_jpg < 1) {
            continue;
        }

        // 参与融合的帧数：按 recon_iv 抽帧，至少 1 帧，避免 n_iter==0
        const int n_iter = std::max(1, total_jpg / recon_iv);
        std::vector<std::shared_ptr<open3d::geometry::PointCloud>> originals;
        originals.reserve(static_cast<size_t>(n_iter));

        const std::string vis_dir = path + "register_scene_steps/";
        if (odt::kRegisterSceneExportStepPly) {
            std::filesystem::create_directories(vis_dir);
        }

        // =============================================================================
        // 步骤 1/3：逐帧加载 RGB-D → 相机系点云，再乘 transforms.npy 变到世界系
        //
        // - file_idx = Filename * recon_iv：与 JPEGImages/{file_idx}.jpg、depth/{file_idx}.png 对齐
        // - LoadSequenceRgbdPointCloud：读图、对齐深度尺寸、按 intr 反投影、上色（见 register_scene_fusion）
        // - ti：transforms 行索引，与 Python registrationParameters 中 label_interval 对齐方式一致
        // - 可选写出 step01_*：相机系首/末帧，以及变换后世界系首/末帧，便于检查配准
        // =============================================================================
        std::cout << "[register_scene] 步骤 1/3: 加载深度帧并应用 transforms.npy"
                  << " (抽帧=" << recon_iv << ", 共 " << n_iter << " 帧)\n";
        progress.begin_step("register_scene: 加载深度帧", static_cast<int64_t>(n_iter));
        for (int Filename = 0; Filename < n_iter; ++Filename) {
            const int file_idx = Filename * recon_iv;
            auto pcd = odt::LoadSequenceRgbdPointCloud(path, file_idx, intr);
            if (!pcd) {
                progress.end_step();
                std::cerr << "Failed frame " << file_idx << "\n";
                return 1;
            }
            if (odt::kRegisterSceneExportStepPly) {
                if (Filename == 0) {
                    odt::WritePointCloudPlyBinary(
                        vis_dir + "step01_frame_first_pre_transform.ply", *pcd);
                }
                if (Filename == n_iter - 1) {
                    odt::WritePointCloudPlyBinary(
                        vis_dir + "step01_frame_last_pre_transform.ply", *pcd);
                }
            }
            const int ti =
                static_cast<int>(recon_iv / std::max(1, label_iv)) * Filename;
            if (ti >= static_cast<int>(n0)) {
                progress.end_step();
                std::cerr << "transforms.npy rows < required index " << ti
                          << "\n";
                return 1;
            }
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    T(r, c) = transforms(ti, r * 4 + c);
                }
            }
            pcd->Transform(T);
            originals.push_back(pcd);
            progress.advance(1);
        }
        progress.end_step();

        if (odt::kRegisterSceneExportStepPly) {
            WriteRegisterSceneStepManifest(
                vis_dir + "README.txt", path, recon_iv, label_iv, voxel_r, inlier_r,
                n_iter, total_jpg);
            if (!originals.empty()) {
                odt::WritePointCloudPlyBinary(vis_dir + "step01_frame_first.ply",
                                              *originals.front());
                odt::WritePointCloudPlyBinary(vis_dir + "step01_frame_last.ply",
                                              *originals.back());
            }
            std::cout << "[register_scene] 已写出步骤点云目录: " << vis_dir << "\n";
        }

        // =============================================================================
        // 步骤 2/3：FusePointCloudsKdTreeMerge — 在世界系下顺序合并多帧点云
        //
        // - 第 0 帧：直接作为累加点集 pts / 颜色 cols，vote 全 0
        // - 第 k 帧：对当前帧每点找累加集上最近邻；若距离 < inlier_r 则给该累加点 vote++（统计跨帧重叠）
        //           若距离 > voxel_r 则将该点作为「新表面」追加到 pts（几何并集意义下的补充）
        // - 输出 pts, cols 即为「KD 合并后的并集」（与 voxel_r 相关的去重/追加规则）
        // - 可选 step02_merged_pre_vote.ply：与最终 registeredScene 点集一致
        // =============================================================================
        std::cout << "[register_scene] 步骤 2/3: 后处理融合 (KDTree 合并, voxel_r="
                  << voxel_r << ", inlier_r=" << inlier_r << ")\n";
        progress.begin_step("register_scene: 后处理融合",
                            static_cast<int64_t>(originals.size()));
        Eigen::MatrixXd pts, cols;
        std::vector<int> vote;
        odt::FusePointCloudsKdTreeMerge(
            originals, voxel_r, inlier_r, pts, cols, vote,
            [&progress]() { progress.advance(1); });
        progress.end_step();

        if (odt::kRegisterSceneExportStepPly) {
            odt::WritePointCloudPlyBinary(vis_dir + "step02_merged_pre_vote.ply",
                                          pts, cols);
        }

        // =============================================================================
        // 步骤 3/3：写出 registeredScene.ply + 可选 vote 筛选对比
        //
        // - registeredScene.ply：直接写入步骤 2 的全量 pts/cols（几何并集，不做 vote 剔除）
        // - 若开启 kRegisterSceneExportStepPly：另存 step03_vote_filtered.ply，仅保留 vote>0
        //   （多帧中曾被其他帧在 inlier_r 内「确认」过的累加点；点数通常更少，仅供对比）
        // - vote 全为 0 且多帧时，旧逻辑会丢光点；此处主结果已改为并集，vote 仅用于可选文件
        // =============================================================================
        std::cout << "[register_scene] 步骤 3/3: 写出 registeredScene.ply（KD 融合几何并集）\n";
        progress.begin_step("register_scene: 写出 PLY", 3);

        const std::string meshfile = path + "registeredScene.ply";
        odt::WritePointCloudPlyBinary(meshfile, pts, cols);
        progress.advance(1);

        Eigen::MatrixXd pts_f;
        Eigen::MatrixXd cols_f;
        if (odt::kRegisterSceneExportStepPly) {
            const bool single_cloud = originals.size() <= 1;
            std::vector<int> keep;
            for (size_t i = 0; i < vote.size(); ++i) {
                if (single_cloud || vote[i] > 0) {
                    keep.push_back(static_cast<int>(i));
                }
            }
            if (keep.empty() && pts.rows() > 0) {
                std::cerr
                    << "warning: no cross-frame inlier votes for optional export "
                       "(step03_vote_filtered); using full merge for that file too.\n";
                for (int i = 0; i < pts.rows(); ++i) {
                    keep.push_back(i);
                }
            }
            pts_f.resize(static_cast<int>(keep.size()), 3);
            cols_f.resize(static_cast<int>(keep.size()), 3);
            for (size_t i = 0; i < keep.size(); ++i) {
                const int r = keep[i];
                pts_f.row(static_cast<int>(i)) = pts.row(r);
                cols_f.row(static_cast<int>(i)) = cols.row(r);
            }
            odt::WritePointCloudPlyBinary(vis_dir + "step03_vote_filtered.ply", pts_f,
                                          cols_f);
        }
        progress.advance(1);
        progress.advance(1);
        progress.end_step();

        std::cout << "[register_scene] 已保存: " << meshfile
                  << " (点数=" << pts.rows() << ")";
        if (odt::kRegisterSceneExportStepPly && pts_f.rows() > 0) {
            std::cout << "；step03_vote_filtered.ply 点数=" << pts_f.rows();
        }
        std::cout << "\n";
    }
    return 0;
}

}  // namespace odt
