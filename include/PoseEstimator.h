/**
 * @file PoseEstimator.h
 * @brief Provides functionality to estimate the pose (rotation and translation) of a brick based on 2D segmentation and depth images.
 */

#pragma once

#include <Eigen/Dense>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "CommonTypes.h"

/**
 * @enum PoseEstimatorStatus
 * @brief Represents the status of the pose estimation process.
 */
enum PoseEstimatorStatus {
    SUCCESS,  ///< Pose estimation was successful
    FAIL      ///< Pose estimation failed
};

/**
 * @class PoseEstimator
 * @brief Abstract base class for estimating the pose of an object based on 2D segmentation and depth images.
 */
class PoseEstimator {
public:
    virtual PoseEstimatorStatus detectPose(const cv::Mat& InputImage, const cv::Mat& objectMask,
        const DepthImage& depthImage, const CameraParams& cameraParams, Pose3D& detectedPose) = 0;
    virtual ~PoseEstimator() = default;
};

/**
 * @class BoxPoseEstimator
 * @brief Class for estimating the pose of a box based on 2D segmentation and depth images.
 * @details The class uses the symetric shape of the box and known dimensions of the box 
   to estimate the pose (rotation and translation)
 */
class BoxPoseEstimator : public PoseEstimator {
public:
    /**
     * @brief Constructor for BoxPoseEstimator.
     * @param boxTemplateSize The template dimensions of the box.
     * @param objectCoordinateSystem The object coordinate system mapping.
     * @param cameraAxisOrder The camera axis order mapping.
     * @param debugMode Whether to enable debug mode.
     */
    BoxPoseEstimator(const TemplateDimBOX& boxTemplateSize, const ObjectCoordinateSystem& objectCoordinateSystem, const CameraAxisOrder& cameraAxisOrder);
    

    /**
     * @brief Detects the pose of the box.
     * @param InputImage The input RGB image.
     * @param objectMask The mask of the object (box) in the image.
     * @param depthImage The depth image corresponding to the RGB image.
     * @param cameraParams The camera parameters.
     * @param detectedPose The detected pose (rotation and translation) of the box.
     * @return The status of the pose estimation process.
     */
    PoseEstimatorStatus detectPose(const cv::Mat& InputImage, const cv::Mat& objectMask,
        const DepthImage& depthImage, const CameraParams& cameraParams, Pose3D& detectedPose) override;

    void setBoxTemplateSize(const TemplateDimBOX &boxTemplateSize);

    void setObjectCoordinateSystem(const std::map<Dimension, Axis>& objectCoordinateSystem);

    void setCameraAxisOrder(const std::map<Axis, std::vector<float>>& cameraAxisOrder);

private:
    /**
     * @brief Calculates the 3D points of the box from the 2D segmentation and depth image.
     * @param segBox2D The 2D segmentation mask of the box.
     * @param depthImage The depth image.
     * @param cameraParams The camera parameters.
     * @param pointCloud3D The resulting 3D point cloud of the box.
     * @param cleanZNoise Whether to clean noise in the Z-axis.
     */
    void calculateBox3DPointColoud(const cv::Mat& segBox2D, const DepthImage& depthImag, const CameraParams& cameraParams, Eigen::MatrixXd& pointCloud3D, bool cleanZNoise = true);

    /**
     * @brief Calculates the rotation matrix iteratively using PCA components.
     * @param pcaComponents The PCA components.
     * @param rotationMatrix The resulting rotation matrix.
     */
    void calculateRotationMatrixIterative(const Eigen::Matrix3d& pcaComponents, Eigen::Matrix3d& rotationMatrix);

    /**
     * @brief Checks if a length is within tolerance of a template dimension.
     * @param length The length to check.
     * @param templateDim The template dimension.
     * @return True if the length is within tolerance, false otherwise.
     */
    bool isWithinTolerance(double length, double templateDim);

    /**
     * @brief Determines the matching dimensions to the X and Y axes.
     * @param TemplateDimBOX The template dimensions of the box.
     * @param rotatedBrick3D The rotated 3D points of the box.
     * @param matchedDims The resulting matched dimensions.
     */
    void determineSideToXYAxis(const TemplateDimBOX &boxTemplateSize,const DepthImage& depthImage, const Eigen::MatrixXd& rotatedBrick3D, std::vector<AxisDimension>& matchedDims);

    /**
     * @brief Adjusts the rotation matrix to align axes.
     * @param matchedDims The matched dimensions.
     * @param rotationMatrix The rotation matrix to adjust.
     * @param changed Whether the rotation matrix was changed.
     */
    void adjustRotationToAlignAxes(const std::vector<AxisDimension>& matchedDims, Eigen::Matrix3d& rotationMatrix, bool& changed);

    /**
     * @brief Converts a rotation matrix to Euler angles.
     * @param R The rotation matrix.
     * @return The Euler angles corresponding to the rotation matrix.
     */
    Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d& R);

    TemplateDimBOX mBoxTemplateSize; ///< The template dimensions of the box.
    ObjectCoordinateSystem mObjectCoordinateSystem; ///< The object coordinate system mapping.
    CameraAxisOrder mCameraAxisOrder; ///< The camera axis order mapping.
};
