#include "odt/odt_registration.hpp"

#include <open3d/Open3D.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <open3d/pipelines/registration/Registration.h>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#if defined(__has_include)
#if __has_include(<opencv2/objdetect/aruco_detector.hpp>)
#include <opencv2/objdetect/aruco_detector.hpp>
#define ODT_HAS_ARUCO_DETECTOR 1
#else
#include <opencv2/aruco.hpp>
#define ODT_HAS_ARUCO_DETECTOR 0
#endif
#else
#include <opencv2/aruco.hpp>
#define ODT_HAS_ARUCO_DETECTOR 0
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace odt {

namespace {

Eigen::Vector3d Median_xyzComponents(const std::vector<Eigen::Vector3d>& pts) {
    Eigen::Vector3d out;
    for (int d = 0; d < 3; ++d) {
        std::vector<double> comp;
        comp.reserve(pts.size());
        for (const auto& p : pts) {
            comp.push_back(p(d));
        }
        const size_t mid = comp.size() / 2;
        std::nth_element(comp.begin(), comp.begin() + static_cast<std::ptrdiff_t>(mid),
                         comp.end());
        out(d) = comp[mid];
    }
    return out;
}

/// 角点 (u,v) 邻域内有效 depth_xyz 点，各坐标分量取中值；抑制单像素飞点
std::optional<Eigen::Vector3d> RobustCornerXyz(const cv::Mat& depth_xyz,
                                                float u,
                                                float v,
                                                int half_win,
                                                int min_valid) {
    const int cx = static_cast<int>(std::lround(u));
    const int cy = static_cast<int>(std::lround(v));
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(static_cast<size_t>((2 * half_win + 1) * (2 * half_win + 1)));
    for (int dy = -half_win; dy <= half_win; ++dy) {
        for (int dx = -half_win; dx <= half_win; ++dx) {
            const int col = cx + dx;
            const int row = cy + dy;
            if (row < 0 || row >= depth_xyz.rows || col < 0 ||
                col >= depth_xyz.cols) {
                continue;
            }
            const cv::Vec3f& p = depth_xyz.at<cv::Vec3f>(row, col);
            if (p[2] > 0.f && std::isfinite(p[0]) && std::isfinite(p[1]) &&
                std::isfinite(p[2])) {
                pts.emplace_back(static_cast<double>(p[0]),
                                 static_cast<double>(p[1]),
                                 static_cast<double>(p[2]));
            }
        }
    }
    if (static_cast<int>(pts.size()) < min_valid) {
        return std::nullopt;
    }
    return Median_xyzComponents(pts);
}

double RmseRigid(const Eigen::Matrix4d& T,
                 const std::vector<Eigen::Vector3d>& src,
                 const std::vector<Eigen::Vector3d>& dst) {
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);
    double s = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const Eigen::Vector3d e = (R * src[i] + t) - dst[i];
        s += e.squaredNorm();
    }
    return std::sqrt(s / static_cast<double>(std::max<size_t>(1, src.size())));
}

Eigen::Matrix4d RigidTransform3D(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    const Eigen::Vector3d centroid_A = A.colwise().mean();
    const Eigen::Vector3d centroid_B = B.colwise().mean();
    Eigen::MatrixXd AA = A.rowwise() - centroid_A.transpose();
    Eigen::MatrixXd BB = B.rowwise() - centroid_B.transpose();
    Eigen::Matrix3d H = AA.transpose() * BB;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = V * svd.matrixU().transpose();
    }
    Eigen::Vector3d t = -R * centroid_A + centroid_B;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

#if ODT_HAS_ARUCO_DETECTOR
void DetectAruco(const cv::Mat& gray,
                 std::vector<int>& ids,
                 std::vector<std::vector<cv::Point2f>>& corners) {
    cv::aruco::Dictionary dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dict, params);
    detector.detectMarkers(gray, corners, ids);
}
#else
void DetectAruco(const cv::Mat& gray,
                 std::vector<int>& ids,
                 std::vector<std::vector<cv::Point2f>>& corners) {
    cv::Ptr<cv::aruco::Dictionary> dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    std::vector<std::vector<cv::Point2f>> rejected;
    cv::aruco::detectMarkers(gray, dict, corners, ids, params, rejected);
}
#endif

}  // namespace

std::optional<Eigen::Matrix4d> MatchRansacStyle(const Eigen::MatrixXd& p,
                                                const Eigen::MatrixXd& p_prime,
                                                double tol) {
    if (p.rows() != p_prime.rows() || p.rows() < 3) {
        return std::nullopt;
    }
    const int n = static_cast<int>(p.rows());
    const int k = std::max(1, static_cast<int>(n * 0.7));
    Eigen::Matrix4d T = RigidTransform3D(p, p_prime);
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::MatrixXd transformed = (R * p.transpose()).transpose();
    transformed.rowwise() += t.transpose();
    Eigen::VectorXd dist = (transformed - p_prime).rowwise().norm();
    std::vector<double> ev(n);
    for (int i = 0; i < n; ++i) {
        ev[static_cast<size_t>(i)] = dist(i);
    }
    std::partial_sort(ev.begin(), ev.begin() + k, ev.end());
    double rmse = 0;
    for (int i = 0; i < k; ++i) {
        rmse += ev[static_cast<size_t>(i)];
    }
    rmse /= static_cast<double>(k);
    if (rmse < tol) {
        return T;
    }
    return std::nullopt;
}

std::shared_ptr<open3d::geometry::PointCloud> BuildPcdFromRgbd(
    const cv::Mat& cad_bgr,
    const cv::Mat& depth_xyz,
    double voxel_size,
    bool estimate_normals) {
    using namespace open3d;
    CV_Assert(cad_bgr.type() == CV_8UC3);
    CV_Assert(depth_xyz.type() == CV_32FC3 &&
              depth_xyz.size() == cad_bgr.size());
    auto pcd = std::make_shared<geometry::PointCloud>();
    for (int v = 0; v < cad_bgr.rows; ++v) {
        const cv::Vec3b* pc = cad_bgr.ptr<cv::Vec3b>(v);
        const cv::Vec3f* pd = depth_xyz.ptr<cv::Vec3f>(v);
        for (int u = 0; u < cad_bgr.cols; ++u) {
            const cv::Vec3f& pt = pd[u];
            if (pt[2] <= 0 || !std::isfinite(pt[2])) {
                continue;
            }
            pcd->points_.emplace_back(static_cast<double>(pt[0]),
                                      static_cast<double>(pt[1]),
                                      static_cast<double>(pt[2]));
            const cv::Vec3b& c = pc[u];
            pcd->colors_.emplace_back(static_cast<double>(c[2]) / 255.0,
                                      static_cast<double>(c[1]) / 255.0,
                                      static_cast<double>(c[0]) / 255.0);
        }
    }
    auto down = pcd->VoxelDownSample(voxel_size);
    if (estimate_normals && !down->points_.empty()) {
        down->EstimateNormals(geometry::KDTreeSearchParamHybrid(0.002 * 2, 30));
    }
    return down;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix6d> RunIcp(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    double voxel_size,
    double max_corr_coarse,
    double max_corr_fine,
    IcpMethod method) {
    using namespace open3d::pipelines::registration;
    (void)voxel_size;
    const open3d::pipelines::registration::ICPConvergenceCriteria icp_crit(
        1e-6, 1e-6, kIcpMaxIterations);
    Eigen::Matrix4d transformation_icp = Eigen::Matrix4d::Identity();
    if (method == IcpMethod::kPointToPlane) {
        auto coarse = RegistrationICP(
            source, target, max_corr_coarse, Eigen::Matrix4d::Identity(),
            TransformationEstimationPointToPlane(), icp_crit);
        auto fine =
            RegistrationICP(source, target, max_corr_fine, coarse.transformation_,
                            TransformationEstimationPointToPlane(), icp_crit);
        transformation_icp = fine.transformation_;
    } else {
        auto result = RegistrationColoredICP(
            source, target, max_corr_fine, Eigen::Matrix4d::Identity(),
            TransformationEstimationForColoredICP(), icp_crit);
        transformation_icp = result.transformation_;
    }
    Eigen::Matrix6d information_icp = GetInformationMatrixFromPointClouds(
        source, target, max_corr_fine, transformation_icp);
    return {transformation_icp, information_icp};
}

std::optional<Eigen::Matrix4d> FeatureRegistrationRgb(
    const cv::Mat& cad_src_bgr,
    const cv::Mat& depth_xyz_src,
    const cv::Mat& cad_dst_bgr,
    const cv::Mat& depth_xyz_dst,
    int min_match_count) {
    cv::Mat gray_src, gray_dst;
    cv::cvtColor(cad_src_bgr, gray_src, cv::COLOR_BGR2GRAY);
    cv::cvtColor(cad_dst_bgr, gray_dst, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    sift->detectAndCompute(gray_src, cv::noArray(), kp1, des1);
    sift->detectAndCompute(gray_dst, cv::noArray(), kp2, des2);
    if (des1.empty() || des2.empty()) {
        return std::nullopt;
    }
    cv::BFMatcher bf(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> matches;
    bf.knnMatch(des1, des2, matches, 2);
    std::vector<cv::DMatch> good;
    for (const auto& m : matches) {
        if (m.size() < 2) {
            continue;
        }
        if (m[0].distance < 0.7f * m[1].distance) {
            good.push_back(m[0]);
        }
    }
    if (static_cast<int>(good.size()) <= min_match_count) {
        return std::nullopt;
    }
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto& m : good) {
        src_pts.push_back(kp1[m.queryIdx].pt);
        dst_pts.push_back(kp2[m.trainIdx].pt);
    }
    cv::Mat mask;
    cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);
    if (mask.empty()) {
        return std::nullopt;
    }
    std::vector<cv::Point2f> src_inlier, dst_inlier;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i, 0)) {
            src_inlier.push_back(src_pts[static_cast<size_t>(i)]);
            dst_inlier.push_back(dst_pts[static_cast<size_t>(i)]);
        }
    }
    std::vector<Eigen::Vector3d> src_good, dst_good;
    for (size_t i = 0; i < src_inlier.size(); ++i) {
        int row_s = static_cast<int>(std::lround(src_inlier[i].y));
        int col_s = static_cast<int>(std::lround(src_inlier[i].x));
        int row_d = static_cast<int>(std::lround(dst_inlier[i].y));
        int col_d = static_cast<int>(std::lround(dst_inlier[i].x));
        if (row_s < 0 || row_s >= depth_xyz_src.rows || col_s < 0 ||
            col_s >= depth_xyz_src.cols) {
            continue;
        }
        if (row_d < 0 || row_d >= depth_xyz_dst.rows || col_d < 0 ||
            col_d >= depth_xyz_dst.cols) {
            continue;
        }
        cv::Vec3f ps = depth_xyz_src.at<cv::Vec3f>(row_s, col_s);
        cv::Vec3f pd = depth_xyz_dst.at<cv::Vec3f>(row_d, col_d);
        if (ps[2] <= 0 || pd[2] <= 0) {
            continue;
        }
        src_good.emplace_back(ps[0], ps[1], ps[2]);
        dst_good.emplace_back(pd[0], pd[1], pd[2]);
    }
    if (src_good.size() < 4) {
        return std::nullopt;
    }
    Eigen::MatrixXd A(static_cast<int>(src_good.size()), 3);
    Eigen::MatrixXd B(static_cast<int>(dst_good.size()), 3);
    for (size_t i = 0; i < src_good.size(); ++i) {
        A.row(static_cast<int>(i)) = src_good[i].transpose();
        B.row(static_cast<int>(i)) = dst_good[i].transpose();
    }
    return MatchRansacStyle(A, B, kMarkerMatchRmseTolMeters);
}

std::optional<Eigen::Matrix4d> MarkerRegistration(
    const cv::Mat& cad_src_bgr,
    const cv::Mat& depth_xyz_src,
    const cv::Mat& cad_dst_bgr,
    const cv::Mat& depth_xyz_dst,
    int diag_src_frame,
    int diag_dst_frame) {
    const int hw = kMarkerCornerDepthHalfWindow;
    const int min_v = kMarkerCornerMinValidSamples;
    auto diag_prefix = [&]() -> std::string {
        if (!kMarkerRegistrationDiagnostics) {
            return {};
        }
        if (diag_src_frame >= 0 && diag_dst_frame >= 0) {
            return "[MarkerRegistration] frames " + std::to_string(diag_src_frame) +
                   "->" + std::to_string(diag_dst_frame) + " ";
        }
        return "[MarkerRegistration] ";
    };
    // 非相邻帧 Marker 失败是常态（见 compute_gt_poses 中 step 抽样）；默认只打相邻帧诊断以免刷屏
    const bool diag_this_pair =
        kMarkerRegistrationDiagnostics &&
        (kMarkerRegistrationDiagVerbose || diag_src_frame < 0 || diag_dst_frame < 0 ||
         diag_dst_frame == diag_src_frame + 1);

    cv::Mat g1, g2;
    cv::cvtColor(cad_src_bgr, g1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(cad_dst_bgr, g2, cv::COLOR_BGR2GRAY);
    std::vector<int> ids1, ids2;
    std::vector<std::vector<cv::Point2f>> c1, c2;
    DetectAruco(g1, ids1, c1);
    DetectAruco(g2, ids2, c2);
    if (ids1.empty() || ids2.empty()) {
        if (diag_this_pair) {
            std::cerr << diag_prefix() << "fail: empty_ids (src=" << ids1.size()
                      << " dst=" << ids2.size() << ")\n";
        }
        return std::nullopt;
    }
    std::vector<int> common;
    for (int a : ids1) {
        if (std::find(ids2.begin(), ids2.end(), a) != ids2.end()) {
            if (std::find(common.begin(), common.end(), a) == common.end()) {
                common.push_back(a);
            }
        }
    }
    if (common.size() < 2) {
        if (diag_this_pair) {
            std::cerr << diag_prefix() << "fail: common_ids=" << common.size()
                      << " (need>=2)\n";
        }
        return std::nullopt;
    }
    std::vector<Eigen::Vector3d> src_pts, dst_pts;
    int corner_robust_fail = 0;
    for (size_t i = 0; i < ids2.size(); ++i) {
        const int id = ids2[i];
        auto it = std::find(ids1.begin(), ids1.end(), id);
        if (it == ids1.end()) {
            continue;
        }
        const size_t j = static_cast<size_t>(std::distance(ids1.begin(), it));
        for (int k = 0; k < 4; ++k) {
            const float u1 = c1[j][static_cast<size_t>(k)].x;
            const float v1 = c1[j][static_cast<size_t>(k)].y;
            const float u2 = c2[i][static_cast<size_t>(k)].x;
            const float v2 = c2[i][static_cast<size_t>(k)].y;
            const int col1 = static_cast<int>(std::lround(u1));
            const int row1 = static_cast<int>(std::lround(v1));
            const int col2 = static_cast<int>(std::lround(u2));
            const int row2 = static_cast<int>(std::lround(v2));
            if (row1 < hw || row1 >= depth_xyz_src.rows - hw || col1 < hw ||
                col1 >= depth_xyz_src.cols - hw) {
                ++corner_robust_fail;
                continue;
            }
            if (row2 < hw || row2 >= depth_xyz_dst.rows - hw || col2 < hw ||
                col2 >= depth_xyz_dst.cols - hw) {
                ++corner_robust_fail;
                continue;
            }
            const std::optional<Eigen::Vector3d> p1 =
                RobustCornerXyz(depth_xyz_src, u1, v1, hw, min_v);
            const std::optional<Eigen::Vector3d> p2 =
                RobustCornerXyz(depth_xyz_dst, u2, v2, hw, min_v);
            if (!p1.has_value() || !p2.has_value()) {
                ++corner_robust_fail;
                continue;
            }
            src_pts.push_back(*p1);
            dst_pts.push_back(*p2);
        }
    }
    if (src_pts.size() < 4) {
        if (diag_this_pair) {
            std::cerr << diag_prefix() << "fail: n_corr_3d=" << src_pts.size()
                      << " corner_robust_fail=" << corner_robust_fail
                      << " (need>=4)\n";
        }
        return std::nullopt;
    }
    Eigen::MatrixXd A(static_cast<int>(src_pts.size()), 3);
    Eigen::MatrixXd B(static_cast<int>(dst_pts.size()), 3);
    for (size_t i = 0; i < src_pts.size(); ++i) {
        A.row(static_cast<int>(i)) = src_pts[i].transpose();
        B.row(static_cast<int>(i)) = dst_pts[i].transpose();
    }
    std::optional<Eigen::Matrix4d> opt =
        MatchRansacStyle(A, B, kMarkerMatchRmseTolMeters);
    if (diag_this_pair) {
        const std::string pfx = diag_prefix();
        if (opt.has_value()) {
            const double rmse = RmseRigid(*opt, src_pts, dst_pts);
            std::cerr << pfx << "ok n_corr=" << src_pts.size()
                      << " rmse_m=" << rmse << " robust_skip_corners="
                      << corner_robust_fail << " win_half=" << hw << "\n";
        } else {
            const Eigen::Matrix4d T_try = RigidTransform3D(A, B);
            const double rmse_raw = RmseRigid(T_try, src_pts, dst_pts);
            std::cerr << pfx << "fail: MatchRansacStyle tol=" << kMarkerMatchRmseTolMeters
                      << " n_corr=" << src_pts.size()
                      << " rmse_if_unfiltered_m=" << rmse_raw
                      << " robust_skip_corners=" << corner_robust_fail << "\n";
        }
    }
    return opt;
}

}  // namespace odt
