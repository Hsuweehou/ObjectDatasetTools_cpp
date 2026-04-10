#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <open3d/Open3D.h>

#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "odt/object_dataset_tools.hpp"
#include "odt/io/npy_io.hpp"
#include "odt/registration/register_scene_fusion.hpp"
#include "odt/registration/registration_params.hpp"
#include "odt/utils/async_progress.hpp"
#include "odt/utils/camera_utils.hpp"
#include "odt/utils/linemod_dataset.hpp"
#include "odt/utils/register_scene_manifest.hpp"

namespace odt {

int ObjectDatasetTools::registerScene(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ... register_scene <path|all> [workspace]\n"
                  << "  path: e.g. LINEMOD/object/  (run from repo root)\n"
                  << "  all:  process all LINEMOD/*/\n"
                  << "  all <workspace>: optional root containing LINEMOD/ (else use cwd)\n";
        return 1;
    }

    std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::string> folders;

    const std::string arg1 = argv[1];
    if (arg1 == "all") {
        std::string repo_root;
        if (argc >= 3 && argv[2] != nullptr && argv[2][0] != '\0') {
            repo_root = NormalizeRepoRootForLinemodWorkspace(argv[2]);
        } else {
            repo_root = cwd.string();
            std::replace(repo_root.begin(), repo_root.end(), '\\', '/');
        }
        folders = GlobLinemodFolders(repo_root);
        if (folders.empty()) {
            std::cerr << "No LINEMOD subfolders under: " << repo_root << "\n";
            return 1;
        }
    } else {
        const std::string f = NormalizeLinemodFolderPath(arg1);
        bool ok = false;
        const auto globs = GlobLinemodFolders(cwd.string());
        for (const auto& g : globs) {
            if (g == f) {
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

    const int recon_iv = odt::GetRegistrationParams().reconstruction_interval;
    const int label_iv = odt::GetRegistrationParams().label_interval;
    const double voxel_r = odt::GetRegistrationParams().voxel_r;
    const double inlier_r = voxel_r * 10;

    odt::AsyncProgress progress;

    const size_t nfold = std::max<size_t>(folders.size(), 1);
    size_t       folder_i = 0;
    for (const std::string& path : folders) {
        ++folder_i;
        const int nfold_i = static_cast<int>(nfold);
        auto      emit_rs_pct = [&](int local_0_100) {
            const int base = static_cast<int>((folder_i - 1) * 100 / nfold_i);
            const int span = 100 / nfold_i;
            int       g    = base + local_0_100 * span / 100;
            if (g > 100) {
                g = 100;
            }
            std::cout << "[odt_rs] pct " << g << "\n" << std::flush;
        };

        std::cout << "[register_scene] 序列目录: " << path << "\n";
        emit_rs_pct(2);

        odt::CameraIntrinsics intr;
        if (!odt::LoadIntrinsicsJson(path + "intrinsics.json", intr)) {
            std::cerr << "intrinsics.json missing\n";
            continue;
        }

        nc::NdArray<double> transforms;
        std::size_t         n0 = 0, n1 = 0, n2 = 0;
        if (!odt::load_ndarray_npy(path + "transforms.npy", transforms, n0, n1, n2) ||
            n1 != 4 || n2 != 4) {
            std::cerr << "transforms.npy invalid\n";
            continue;
        }

        const int total_jpg = CountJpegImagesInDir(path + "JPEGImages");
        if (total_jpg < 1) {
            continue;
        }

        const int n_iter = std::max(1, total_jpg / recon_iv);
        std::vector<std::shared_ptr<open3d::geometry::PointCloud>> originals;
        originals.reserve(static_cast<size_t>(n_iter));

        const std::string vis_dir = path + "register_scene_steps/";
        if (odt::GetRegistrationParams().register_scene_export_step_ply) {
            std::filesystem::create_directories(vis_dir);
        }

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
            if (odt::GetRegistrationParams().register_scene_export_step_ply) {
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
            {
                const int lp =
                    5 + static_cast<int>((Filename + 1) * 55 / std::max(1, n_iter));
                emit_rs_pct(std::min(60, lp));
            }
        }
        progress.end_step();
        emit_rs_pct(62);

        if (odt::GetRegistrationParams().register_scene_export_step_ply) {
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

        std::cout << "[register_scene] 步骤 2/3: 后处理融合 (KDTree 合并, voxel_r="
                  << voxel_r << ", inlier_r=" << inlier_r << ")\n";
        progress.begin_step("register_scene: 后处理融合",
                            static_cast<int64_t>(originals.size()));
        Eigen::MatrixXd pts, cols;
        std::vector<int> vote;
        size_t          fuse_done = 0;
        const size_t    n_orig    = originals.size();
        odt::FusePointCloudsKdTreeMerge(
            originals, voxel_r, inlier_r, pts, cols, vote,
            [&progress, &emit_rs_pct, &fuse_done, n_orig]() {
                progress.advance(1);
                ++fuse_done;
                if (n_orig > 0) {
                    const int lp = 62 + static_cast<int>(fuse_done * 28 /
                                                         std::max<size_t>(n_orig, 1));
                    emit_rs_pct(std::min(92, lp));
                }
            });
        progress.end_step();
        emit_rs_pct(93);

        if (odt::GetRegistrationParams().register_scene_export_step_ply) {
            odt::WritePointCloudPlyBinary(vis_dir + "step02_merged_pre_vote.ply",
                                          pts, cols);
        }

        std::cout << "[register_scene] 步骤 3/3: 写出 registeredScene.ply（KD 融合几何并集）\n";
        progress.begin_step("register_scene: 写出 PLY", 3);

        const std::string meshfile = path + "registeredScene.ply";
        odt::WritePointCloudPlyBinary(meshfile, pts, cols);
        progress.advance(1);
        emit_rs_pct(97);
        std::cout << "[odt_rs] ply " << meshfile << "\n" << std::flush;

        Eigen::MatrixXd pts_f;
        Eigen::MatrixXd cols_f;
        if (odt::GetRegistrationParams().register_scene_export_step_ply) {
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
        if (odt::GetRegistrationParams().register_scene_export_step_ply && pts_f.rows() > 0) {
            std::cout << "；step03_vote_filtered.ply 点数=" << pts_f.rows();
        }
        std::cout << "\n";
    }
    std::cout << "[odt_rs] pct 100\n" << std::flush;
    return 0;
}

}  // namespace odt
