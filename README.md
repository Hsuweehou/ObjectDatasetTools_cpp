# ObjectDatasetTools_cpp

### C++ version of ObjectDatasetTools(https://github.com/F2Wang/ObjectDatasetTools)

**LINEMOD 风格物体数据集** 的 C++ 工具：通过 DS-A10400相机 采集 RGB-D、估计帧间位姿与全局优化、融合场景点云，并生成训练用标签与网格尺度信息。对应原 Python/ObjectDatasetTools 流水线中的若干步骤，以 `object_dataset_tools` 单一可执行程序提供子命令入口。

---

## 目录结构

```
ObjectDatasetTools_cpp/
├── CMakeLists.txt          # 工程配置：库 odt_core / odt_tools、可执行文件 object_dataset_tools
├── config/
│   └── algorithm_params.json   # 配准、ICP、AprilTag/ArUco 相关默认参数（registration 段）
├── include/odt/            # 公共头文件（命名空间 odt）
│   ├── object_dataset_tools.hpp    # CLI 入口类 ObjectDatasetTools
│   ├── io/npy_io.hpp               # .npy 读写
│   ├── registration/             # 点云配准、融合、位姿可视化、参数
│   └── utils/                    # 相机、LINEMOD 数据集、AIR 采集、标签投影等
├── src/                    # 实现：按功能分文件（registration/、utils/ 与各子命令 .cpp）
├── test/main.cpp           # 可执行程序入口：解析子命令并分发
└── data/                   # 示例或测试数据（如 cylinder_bottle）
```

- **`odt_core`**（静态库）：配准核心（ICP、特征/标记辅助、场景融合、参数加载）、相机与异步进度等，链接 OpenCV、Open3D、Eigen、nlohmann_json。
- **`odt_tools`**（静态库）：各业务子命令实现（采集、GT 位姿、注册场景、标签、网格尺度），依赖 `odt_core`、AIRScanner、glog、NumCpp 等。

---

## 依赖

### 第三方库（CMake 默认路径）

与仓库根工程一致，默认期望 **`3rdParty` 与仓库根目录并列**（即 `project_template_scanner/../3rdParty`）。可通过缓存/环境变量 **`THIRD_PARTY_LIBRARY_DIR`** 覆盖。

| 依赖 | 说明 |
|------|------|
| **OpenCV** | 默认 `OpenCV_DIR` → `{THIRD_PARTY}/OpenCV4.X_GPU/x64/vc16/lib` |
| **Open3D** | `Open3D_DIR` → `{THIRD_PARTY}/open3d/cmake` |
| **Eigen3** | `Eigen3_DIR` → `{THIRD_PARTY}/Eigen/share/eigen3/cmake` |
| **gflags / glog** | glog 依赖 gflags；路径见 `CMakeLists.txt` |
| **nlohmann/json** | 头文件：`{THIRD_PARTY}/nlohmann/include` |
| **NumCpp** | 头文件：`NUMCPP_INCLUDE_DIR` 默认 `{THIRD_PARTY}/NumCpp_2.16.1/include` |

### AIRScanner SDK

- CMake 变量 **`ODT_AIR_SDK_ROOT`**：SDK 根目录（含 `include/`、`lib/cmake/AIRScanner`），默认 **`仓库根/SDK_C++`**。
- 也可设置 **`AIRScanner_DIR`** 指向 `.../lib/cmake/AIRScanner`。

### 运行时（Windows）

构建脚本会在生成 `object_dataset_tools` 后尝试将 **Orbbec / Onplus 相关 DLL、AIRScanner、OpenCV、Open3D、glog** 等复制到可执行文件目录；路径依赖仓库内 `scanner_SDKs`、`ORBBEC_SDK_ROOT` 等，详见 `CMakeLists.txt` 中 `POST_BUILD` 段。

---

## 编译

**要求**：CMake ≥ 3.16，C++17，MSVC 建议在 Visual Studio 环境中配置（工程使用 `/utf-8`）。

### 方式一：作为仓库根工程的一部分

在仓库根目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target object_dataset_tools
```

### 方式二：仅配置本目录

在 `algorithm/ObjectDatasetTools_cpp` 下（需已准备好 `3rdParty` 与 SDK）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

生成物通常为 `build/Release/object_dataset_tools.exe`（多配置生成器）或 `build/object_dataset_tools`（单配置）。

**常用 CMake 变量**：`THIRD_PARTY_LIBRARY_DIR`、`ODT_AIR_SDK_ROOT`、`OpenCV_DIR`、`Open3D_DIR`、`NUMCPP_INCLUDE_DIR`。

---

## 运行

可执行文件：**`object_dataset_tools`**。

```text
Usage: object_dataset_tools <command> [args...]

Commands:
  record_air           <output_dir> [timed ... | interactive ...]
  compute_gt_poses     <sequence_dir>
  register_scene       <path|all> [workspace]
  create_label_files   <path|all>
  get_mesh_scale       [<LINEMOD/obj/>|all]   (省略 = 当前目录下 ./LINEMOD 全部对象)
```

在 Windows 下程序会尽量将控制台设为 **UTF-8**，避免中文帮助信息乱码。

### `record_air`

使用 AIRScanner 按 **LINEMOD 目录结构** 写入采集数据。

- **定时长**：`record_air <output_dir> [timed] <record_len_sec> <countdown_sec> [ip] [port] [scanner_params.json]`；省略 `timed` 时参数位置与旧版兼容：`record_air <out> [record_len] [countdown] [ip] [port] [params]`。
- **交互**：`record_air <output_dir> interactive [ip] [port] [scanner_params.json]`（`s`/`f`/`q` 等按键逻辑见源码注释）。

网络与参数也可通过环境变量 **`SCANNER_IP`、`SCANNER_PORT`、`SCANNER_PARAMS_PATH`** 提供。

### `compute_gt_poses`

对 **单条序列目录**（例如 `LINEMOD/某物体名`，需含 `JPEGImages`、`depth`、`intrinsics.json`）做帧间关系：特征/标记辅助、ICP、Open3D **全局位姿图优化**，输出位姿等到约定目录（如 `tag_pose`）。

参数间隔、体素、ICP 与标记诊断等来自 **`config/algorithm_params.json`** 中的 `registration` 配置。

### `register_scene`

将多帧深度/RGB 融合为 **注册场景点云**（`path` 为某条序列或 `all` 批量；可选第二个参数指定工作区根目录）。可选在 `register_scene_steps` 下导出中间 PLY 便于调试（见 `algorithm_params.json` 中 `register_scene_export_step_ply`）。

### `create_label_files`

根据配准结果与相机内参生成 **标签、mask、变换** 等 LINEMOD 训练所需文件（`path|all` 同上）。

### `get_mesh_scale`

与各 `LINEMOD/<name>/<name>.ply` 一致：计算 **相邻顶点最大间距**（与 Python `getmeshscale.py` 语义一致），用于网格尺度；可指定单个 `obj` 路径或 `all`。

---

## 实现功能概要

| 模块 | 作用 |
|------|------|
| **采集 `record_air`** | AIRScanner 封装（`air_record_scanner`）、LINEMOD 目录准备、深度/RGB/点云写入 |
| **GT 位姿 `compute_gt_poses`** | RGB-D 点云构建、帧对匹配、RANSAC 风格匹配、ICP（点面等）、AprilTag/ArUco 类标记配准、全局优化、进度与可视化辅助 |
| **场景注册 `register_scene`** | 多帧融合、体素合并、可选投票过滤；与 `register_scene_fusion` 等实现联动 |
| **标签 `create_label_files`** | 2D 标签与投影（`label_projection` 等） |
| **网格尺度 `get_mesh_scale`** | 读取 PLY，统计相邻顶点最大距离 |

算法参数集中在 **`config/algorithm_params.json`**（如体素 `voxel_size` / `voxel_r`、`label_interval`、`reconstruction_interval`、ICP 迭代次数、ArUco 标记边长、角点深度窗口与 RMSE 阈值等）。修改后需与程序加载路径一致（由 `registration_params` 等逻辑解析）。

