#include "odt/registration/register_scene_fusion.hpp"

#include <open3d/Open3D.h>
#include <open3d/geometry/KDTreeFlann.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>
#include <utility>

namespace odt {

std::shared_ptr<open3d::geometry::PointCloud> LoadSequenceRgbdPointCloud(
    const std::string& sequence_path,
    int file_index,
    const CameraIntrinsics& intr) {
    std::string base = sequence_path;
    while(!base.empty() && (base.back() == '/' || base.back() == '\\')) {
        base.pop_back();
    }
    std::replace(base.begin(), base.end(), '\\', '/');
    base += '/';

    const std::string img_path   = base + "JPEGImages/" + std::to_string(file_index) + ".jpg";
    const std::string depth_path = base + "depth/" + std::to_string(file_index) + ".png";

    cv::Mat bgr       = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat depth_u16 = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if(bgr.empty() || depth_u16.empty() || depth_u16.type() != CV_16U) {
        return nullptr;
    }
    ResizeDepthUint16ToMatchColor(bgr, depth_u16);
    cv::Mat xyz;
    DepthUint16ToPointCloud(depth_u16, intr, xyz);

    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.reserve(static_cast<size_t>(bgr.rows * bgr.cols));
    pcd->colors_.reserve(static_cast<size_t>(bgr.rows * bgr.cols));

    for(int v = 0; v < bgr.rows; ++v) {
        const cv::Vec3f* row_xyz = xyz.ptr<cv::Vec3f>(v);
        const cv::Vec3b* row_bgr = bgr.ptr<cv::Vec3b>(v);
        for(int u = 0; u < bgr.cols; ++u) {
            const cv::Vec3f& p = row_xyz[u];
            if(p[2] <= 0.f) {
                continue;
            }
            pcd->points_.emplace_back(static_cast<double>(p[0]), static_cast<double>(p[1]),
                                        static_cast<double>(p[2]));
            const cv::Vec3b& c = row_bgr[u];
            pcd->colors_.emplace_back(static_cast<double>(c[2]) / 255.0,
                                        static_cast<double>(c[1]) / 255.0,
                                        static_cast<double>(c[0]) / 255.0);
        }
    }
    return pcd;
}

bool WritePointCloudPlyBinary(const std::string& path,
                              const open3d::geometry::PointCloud& pcd) {
    return open3d::io::WritePointCloud(path, pcd, /*write_ascii=*/false);
}

bool WritePointCloudPlyBinary(const std::string& path,
                              const Eigen::MatrixXd& pts,
                              const Eigen::MatrixXd& cols) {
    open3d::geometry::PointCloud pcd;
    const int n = static_cast<int>(pts.rows());
    if(cols.rows() != n || pts.cols() != 3 || cols.cols() != 3) {
        return false;
    }
    pcd.points_.reserve(static_cast<size_t>(n));
    pcd.colors_.reserve(static_cast<size_t>(n));
    for(int i = 0; i < n; ++i) {
        pcd.points_.emplace_back(pts(i, 0), pts(i, 1), pts(i, 2));
        pcd.colors_.emplace_back(cols(i, 0), cols(i, 1), cols(i, 2));
    }
    return WritePointCloudPlyBinary(path, pcd);
}

void FusePointCloudsKdTreeMerge(
    const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& originals,
    double voxel_r,
    double inlier_r,
    Eigen::MatrixXd& pts,
    Eigen::MatrixXd& cols,
    std::vector<int>& vote,
    const std::function<void()>& on_frame) {
    pts.resize(0, 3);
    cols.resize(0, 3);
    vote.clear();
    if(originals.empty()) {
        return;
    }

    for(size_t pid = 0; pid < originals.size(); ++pid) {
        if(on_frame) {
            on_frame();
        }
        const auto& pcd = originals[pid];
        if(!pcd || pcd->points_.empty()) {
            continue;
        }

        const size_t nq = pcd->points_.size();
        if(pid == 0) {
            pts.resize(static_cast<int>(nq), 3);
            cols.resize(static_cast<int>(nq), 3);
            vote.assign(nq, 0);
            for(size_t i = 0; i < nq; ++i) {
                pts.row(static_cast<int>(i)) = pcd->points_[i].transpose();
                cols.row(static_cast<int>(i)) = pcd->colors_[i].transpose();
            }
            continue;
        }

        open3d::geometry::PointCloud acc;
        acc.points_.reserve(static_cast<size_t>(pts.rows()));
        for(int i = 0; i < pts.rows(); ++i) {
            acc.points_.emplace_back(pts(i, 0), pts(i, 1), pts(i, 2));
        }
        open3d::geometry::KDTreeFlann kdtree(acc);

        std::vector<double> dist(nq);
        std::vector<int>    nn_idx(nq);
        for(size_t i = 0; i < nq; ++i) {
            std::vector<int>    idx;
            std::vector<double> dists_sq;
            if(kdtree.SearchKNN(pcd->points_[i], 1, idx, dists_sq) && !idx.empty()) {
                dist[i] = std::sqrt(dists_sq[0]);
                nn_idx[i] = idx[0];
            } else {
                dist[i] = std::numeric_limits<double>::infinity();
                nn_idx[i] = -1;
            }
        }

        for(size_t i = 0; i < nq; ++i) {
            if(dist[i] < inlier_r && nn_idx[i] >= 0) {
                vote[static_cast<size_t>(nn_idx[i])] += 1;
            }
        }

        std::vector<int> append_rows;
        for(size_t i = 0; i < nq; ++i) {
            if(dist[i] > voxel_r) {
                append_rows.push_back(static_cast<int>(i));
            }
        }

        const int old_n = pts.rows();
        const int add_n = static_cast<int>(append_rows.size());
        if(add_n > 0) {
            Eigen::MatrixXd new_pts(old_n + add_n, 3);
            Eigen::MatrixXd new_cols(old_n + add_n, 3);
            if(old_n > 0) {
                new_pts.topRows(old_n) = pts;
                new_cols.topRows(old_n) = cols;
            }
            for(int j = 0; j < add_n; ++j) {
                const int r = append_rows[static_cast<size_t>(j)];
                new_pts.row(old_n + j) = pcd->points_[static_cast<size_t>(r)].transpose();
                new_cols.row(old_n + j) = pcd->colors_[static_cast<size_t>(r)].transpose();
            }
            pts  = std::move(new_pts);
            cols = std::move(new_cols);
            vote.resize(static_cast<size_t>(old_n + add_n), 0);
        }
    }
}

}  // namespace odt
