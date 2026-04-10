#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <open3d/Open3D.h>

#include <Eigen/Core>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "odt/object_dataset_tools.hpp"
#include "odt/utils/linemod_dataset.hpp"

namespace {

/// 与 Python `getmeshscale.py` 一致：顶点列表中相邻两点距离的最大值（非全对距离、非包围盒对角线）
double MaxConsecutiveVertexSpan(const std::vector<Eigen::Vector3d>& pts) {
    if (pts.size() < 2) {
        return 0.0;
    }
    double m = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
        const double d = (pts[i] - pts[i - 1]).norm();
        if (d > m) {
            m = d;
        }
    }
    return m;
}

bool LoadVerticesFromObjectPly(const std::string& path,
                               std::vector<Eigen::Vector3d>& out_vertices) {
    out_vertices.clear();
    open3d::geometry::TriangleMesh mesh;
    if (open3d::io::ReadTriangleMesh(path, mesh) && mesh.HasVertices()) {
        out_vertices = mesh.vertices_;
        return true;
    }
    open3d::geometry::PointCloud pcd;
    if (open3d::io::ReadPointCloud(path, pcd) && pcd.HasPoints()) {
        out_vertices = pcd.points_;
        return true;
    }
    return false;
}

}  // namespace

namespace odt {

int ObjectDatasetTools::getMeshScale(int argc, char** argv) {
    std::vector<std::string> folders;
    if (argc < 2) {
        folders = GlobLinemodFolders(std::filesystem::current_path().string());
    } else {
        const std::string arg1 = argv[1];
        if (arg1 == "all") {
            folders = GlobLinemodFolders(std::filesystem::current_path().string());
        } else {
            folders.push_back(NormalizeLinemodFolderPath(arg1));
        }
    }

    if (folders.empty()) {
        std::cerr << "No LINEMOD folders (use from repo root with LINEMOD/*/ or pass "
                     "<path|all>).\n";
        return 1;
    }

    for (const std::string& folder : folders) {
        std::cout << folder << "\n";
        const std::string name  = ObjectNameFromFolder(folder);
        const std::string ply   = folder + name + ".ply";
        std::vector<Eigen::Vector3d> vertices;
        if (!LoadVerticesFromObjectPly(ply, vertices)) {
            std::cout << "Mesh does not exist\n";
            continue;
        }
        const double max_d = MaxConsecutiveVertexSpan(vertices);
        std::cout << "Max vertice distance is: " << max_d << " m.\n";
    }
    return 0;
}

}  // namespace odt
