#include "odt/registration/pose_visualization.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

#if defined(__has_include)
#if __has_include(<opencv2/objdetect/aruco_detector.hpp>)
#include <opencv2/objdetect/aruco_detector.hpp>
#define ODT_POSE_VIZ_ARUCO_DETECTOR 1
#else
#define ODT_POSE_VIZ_ARUCO_DETECTOR 0
#endif
#else
#define ODT_POSE_VIZ_ARUCO_DETECTOR 0
#endif

#include <iostream>
#include <vector>

namespace odt {

namespace {

void DetectArucoMarkers(const cv::Mat& gray,
                        std::vector<int>& ids,
                        std::vector<std::vector<cv::Point2f>>& corners) {
#if ODT_POSE_VIZ_ARUCO_DETECTOR
    cv::aruco::Dictionary dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dict, params);
    detector.detectMarkers(gray, corners, ids);
#else
    cv::Ptr<cv::aruco::Dictionary> dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    std::vector<std::vector<cv::Point2f>> rejected;
    cv::aruco::detectMarkers(gray, dict, corners, ids, params, rejected);
#endif
}

void IntrinsicsToCvK(const CameraIntrinsics& intr,
                     cv::Mat& K,
                     cv::Mat& dist_coeffs) {
    K = (cv::Mat_<double>(3, 3) << intr.fx, 0.0, intr.ppx, 0.0, intr.fy,
         intr.ppy, 0.0, 0.0, 1.0);
    dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
}

}  // namespace

bool DrawArucoTagAxesOnImage(cv::Mat& bgr,
                             const CameraIntrinsics& intr,
                             double marker_length_m) {
    if (bgr.empty()) {
        return false;
    }
    try {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        DetectArucoMarkers(gray, ids, corners);
        if (ids.empty()) {
            return false;
        }
        cv::Mat K, dist;
        IntrinsicsToCvK(intr, K, dist);
        const float ml = static_cast<float>(marker_length_m);
        if (!(ml > 0.f)) {
            return false;
        }
        // 与 OpenCV estimatePoseSingleMarkers 相同的中心原点、角点顺序（顺时针自左上角）
        const float h = ml * 0.5f;
        const std::vector<cv::Point3f> obj = {
            {-h, h, 0.f},
            {h, h, 0.f},
            {h, -h, 0.f},
            {-h, -h, 0.f},
        };
        for (size_t i = 0; i < corners.size(); ++i) {
            if (corners[i].size() != 4) {
                continue;
            }
            cv::Mat rvec, tvec;
            try {
                cv::solvePnP(obj, corners[i], K, dist, rvec, tvec, false,
                             cv::SOLVEPNP_IPPE_SQUARE);
            } catch (const cv::Exception&) {
                try {
                    cv::solvePnP(obj, corners[i], K, dist, rvec, tvec, false,
                                 cv::SOLVEPNP_ITERATIVE);
                } catch (const cv::Exception&) {
                    continue;
                }
            }
            cv::drawFrameAxes(bgr, K, dist, rvec, tvec, ml * 0.5f, 3);
            cv::Point2f ctr(0.f, 0.f);
            for (int k = 0; k < 4; ++k) {
                ctr.x += corners[i][static_cast<size_t>(k)].x;
                ctr.y += corners[i][static_cast<size_t>(k)].y;
            }
            ctr.x *= 0.25f;
            ctr.y *= 0.25f;
            const cv::Point ic(static_cast<int>(std::lround(ctr.x)),
                               static_cast<int>(std::lround(ctr.y)));
            cv::circle(bgr, ic, 6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::circle(bgr, ic, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        }
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[DrawArucoTagAxesOnImage] OpenCV: " << e.what() << "\n";
        return false;
    }
}

}  // namespace odt
