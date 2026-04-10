#pragma once

#include <fstream>
#include <string>

namespace odt {

/// 写出 `register_scene_steps/README.txt`，说明中间 PLY 含义
inline void WriteRegisterSceneStepManifest(const std::string& path,
                                           const std::string& seq_path,
                                           int                  recon_iv,
                                           int                  label_iv,
                                           double               voxel_r,
                                           double               inlier_r,
                                           int                  n_iter,
                                           int                  total_jpg) {
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

}  // namespace odt
