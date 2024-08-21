#pragma once
#ifndef BRICK_POSE_DETECTION_3D_H
#define BRICK_POSE_DETECTION_3D_H

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
  
#include "PoseEstimator.h"
#include "SegmentDetector.h"


enum class BrickPoseDetection3DStatus {
	SUCCESS,
    NO_BRICK_DETECTED,
    FAILED_TO_DETECT_BRICK,
    FAIL 
};

class BrickPoseDetection3D {
public:
    BrickPoseDetection3D(const std::string& modelBasePath,const PoseHyperParams &hyperParams = PoseHyperParams());
    BrickPoseDetection3DStatus process(const cv::Mat& rgbImage, const DepthImage& depthImage, const CameraParams& cameraParams,Pose3D &brick3DPose,std::optional<std::string> dumpPath = std::nullopt);
    BrickPoseDetection3DStatus getBrick2DMask(const cv::Mat& rgbImage, const DepthImage& depthImage, cv::Mat& maskImage);
private:
    std::unique_ptr<SegAnythingONNX> mSegmentationModel;
    std::unique_ptr<BoxPoseEstimator> mRotateEstimator;
    PoseHyperParams mHyperParams;
    cv::Size getModelInputSize() const;
    void calculateCropForNetwork(const cv::Size& imageSize, const cv::Size& networkInputSize, const cv::Point& detectedBrickPoint, cv::Rect& cropRect);
    void filterFullMask(const cv::Mat& fullMask, const cv::Point& inputPoint, cv::Mat& filteredMask); // Updated declaration
    void findPointInMiddleBrick(const DepthImage& depthMap, cv::Point& pointOfInterest);
};


#endif // BRICK_POSE_DETECTION_3D_H
