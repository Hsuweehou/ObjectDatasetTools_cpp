#pragma once

namespace odt {

/// AIR 相机采集（LINEMOD 目录结构）：定时长模式或 interactive（s/f/q）
int RunRecordAir(int argc, char** argv);

/// 位姿图 + ICP，原 compute_gt_poses
int RunComputeGtPoses(int argc, char** argv);

/// 场景点云合并，原 register_scene
int RunRegisterScene(int argc, char** argv);

/// 标签 / mask / transforms，原 create_label_files
int RunCreateLabelFiles(int argc, char** argv);

}  // namespace odt
