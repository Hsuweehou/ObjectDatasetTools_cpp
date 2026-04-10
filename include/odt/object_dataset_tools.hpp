#pragma once

namespace odt {

/// 数据集工具 CLI 子命令的统一入口（原各 `Run*` 自由函数）
class ObjectDatasetTools {
public:
    /// AIR 相机采集（LINEMOD 目录结构）：定时长模式或 interactive（s/f/q）
    int recordAir(int argc, char** argv);

    /// 位姿图 + ICP，原 compute_gt_poses
    int computeGtPoses(int argc, char** argv);

    /// 场景点云合并，原 register_scene
    int registerScene(int argc, char** argv);

    /// 标签 / mask / transforms，原 create_label_files
    int createLabelFiles(int argc, char** argv);

    /// 与 Python `getmeshscale.py` 一致：各 `LINEMOD/<name>/<name>.ply` 相邻顶点最大间距（米）
    int getMeshScale(int argc, char** argv);
};

}  // namespace odt
