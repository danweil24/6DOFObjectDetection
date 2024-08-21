#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include "PoseEstimator.h"
#include "Utils.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


BoxPoseEstimator::BoxPoseEstimator(const TemplateDimBOX& boxTemplateSize, const ObjectCoordinateSystem& objectCoordinateSystem,
    const CameraAxisOrder& cameraAxisOrder) : mBoxTemplateSize(boxTemplateSize),
    mObjectCoordinateSystem(objectCoordinateSystem), mCameraAxisOrder(cameraAxisOrder) {};
 
PoseEstimatorStatus BoxPoseEstimator::detectPose(const cv::Mat& inputImage, const cv::Mat& objectMask,
    const DepthImage& depthImage, const CameraParams& cameraParams, Pose3D& detectedPose) {
    // Start timing
    auto start = std::chrono::steady_clock::now();

    // Calculate 3D brick points
    Eigen::MatrixXd segmentedBrick3DFiltered;
    calculateBox3DPointColoud(objectMask, depthImage, cameraParams, segmentedBrick3DFiltered, true);

    // Time for 3D brick calculation
    auto timeBrick3D = std::chrono::steady_clock::now();
    LOG(VerbosityLevel::MEDIUM, "Time brick 3D = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(timeBrick3D - start).count()) + "[ms]");

    // Center the 3D point cloud data
    Eigen::Vector3d centroid = segmentedBrick3DFiltered.colwise().mean();
    Eigen::MatrixXd centeredBrick3D = segmentedBrick3DFiltered.rowwise() - centroid.transpose();

    // Apply PCA to find the principal axes 
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centeredBrick3D, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d pcaComponents = svd.matrixV();
    Eigen::Matrix3d rotationMatrix;
    calculateRotationMatrixIterative(pcaComponents, rotationMatrix);
    Eigen::Vector3d eulerAngles = rotationMatrixToEulerAngles(rotationMatrix);
    eulerAngles = eulerAngles * 180.0 / M_PI; // Convert to degrees

    // Time for finding rotation matrix
    auto timeFindRotationMatrix = std::chrono::steady_clock::now();
    LOG(VerbosityLevel::MEDIUM, "Time find rotation matrix = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(timeFindRotationMatrix - timeBrick3D).count()) + "[ms]");

    // Rotate the point cloud using the rotation matrix
    Eigen::MatrixXd rotatedPointCloud = (centeredBrick3D * rotationMatrix.transpose()).rowwise() + centroid.transpose();

    // Time for rotating point cloud
    auto timeRotatePointCloud = std::chrono::steady_clock::now();
    LOG(VerbosityLevel::MEDIUM, "Time rotate point cloud = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(timeRotatePointCloud - timeFindRotationMatrix).count()) + "[ms]");

    // Determine the correct side axis based on brick size
    auto timeDetermineSideAxis = std::chrono::steady_clock::now();
    std::vector<AxisDimension> matchedDims;
    determineSideToXYAxis(mBoxTemplateSize,depthImage, rotatedPointCloud, matchedDims);

    bool isChanges = false;
    Eigen::Vector3d adjustedEulerAngles = eulerAngles;
    if (!matchedDims.empty()) {
        adjustRotationToAlignAxes(matchedDims, rotationMatrix, isChanges);
        adjustedEulerAngles = rotationMatrixToEulerAngles(rotationMatrix) * 180.0 / M_PI;
        LOG(VerbosityLevel::HIGH, "Adjusted Euler Angles (degrees): " + std::to_string(adjustedEulerAngles[0]) + ", " + std::to_string(adjustedEulerAngles[1]) + ", " + std::to_string(adjustedEulerAngles[2]));
    }
    else {
       LOG(VerbosityLevel::HIGH, "No side axis matched, need more data");
    }

    Eigen::Vector3d finalTranslation = rotatedPointCloud.colwise().mean();

    // Translate to the final position vector
    detectedPose.translation = std::vector<double>(finalTranslation.data(), finalTranslation.data() + finalTranslation.size());
    detectedPose.rotation = std::vector<double>(adjustedEulerAngles.data(), adjustedEulerAngles.data() + adjustedEulerAngles.size());

    // Final timing
    auto end = std::chrono::steady_clock::now();
    LOG(VerbosityLevel::MEDIUM, "Time determine side axis = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - timeDetermineSideAxis).count()) + "[ms]");
    LOG(VerbosityLevel::MEDIUM, "Time detectPose = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + "[ms]");

    return PoseEstimatorStatus::SUCCESS;
}

void BoxPoseEstimator::setBoxTemplateSize(const TemplateDimBOX& boxTemplateSize)
{
	mBoxTemplateSize = boxTemplateSize;
}

void BoxPoseEstimator::setObjectCoordinateSystem(const std::map<Dimension, Axis>& objectCoordinateSystem)
{
    for (const auto& item : objectCoordinateSystem) {
		mObjectCoordinateSystem[item.first] = item.second;
	}
}

void BoxPoseEstimator::setCameraAxisOrder(const std::map<Axis,std::vector<float>>& cameraAxisOrder)
{
    for (const auto& item : cameraAxisOrder) {
		mCameraAxisOrder[item.first] = item.second;
	}
}

void BoxPoseEstimator::calculateBox3DPointColoud(const cv::Mat& segBrick2D, const DepthImage& depthImage, const CameraParams& cameraParams, Eigen::MatrixXd& pointCloud3D, bool cleanZNoise) {
    // Locate the non-zero pixel values in the segmentation mask
    std::vector<cv::Point> pixelLocations;
    cv::findNonZero(segBrick2D, pixelLocations);

    // Containers for 3D points and their Z values
    std::vector<cv::Point3d> points3D;
    std::vector<double> zValues;

    // Calculate 3D points from 2D segmentation mask and depth image
    for (const auto& point : pixelLocations) {
        double depth = static_cast<double>(depthImage.depthImageMat.at<ushort>(point.y, point.x)); // Assuming depthImage is CV_16U
        cv::Point3d xyz;
        uvToXyz(point, depth, cameraParams, xyz);
        points3D.push_back(xyz);
        zValues.push_back(xyz.z);
    }

    if (cleanZNoise && !zValues.empty()) {
        // Convert points3D to cv::Mat for easy processing
        cv::Mat pointsMat(points3D.size(), 3, CV_64F);
        for (size_t i = 0; i < points3D.size(); ++i) {
            pointsMat.at<double>(i, 0) = points3D[i].x;
            pointsMat.at<double>(i, 1) = points3D[i].y;
            pointsMat.at<double>(i, 2) = points3D[i].z;
        }

        // Calculate the mean and standard deviation of the Z values
        cv::Scalar mean, stddev;
        cv::meanStdDev(pointsMat.col(2), mean, stddev);

        // Define a threshold based on one standard deviation 1 standard deviation
        double threshold = stddev[0];

        // Filter out points that are too far from the mean depth based on the threshold
        cv::Mat filteredPointsMat;
        for (int i = 0; i < pointsMat.rows; ++i) {
            if (std::abs(pointsMat.at<double>(i, 2) - mean[0]) < threshold) {
                filteredPointsMat.push_back(pointsMat.row(i));
            }
        }

        // Convert filtered points to Eigen::MatrixXd
        pointCloud3D.resize(filteredPointsMat.rows, 3);
        for (int i = 0; i < filteredPointsMat.rows; ++i) {
            pointCloud3D(i, 0) = filteredPointsMat.at<double>(i, 0);
            pointCloud3D(i, 1) = filteredPointsMat.at<double>(i, 1);
            pointCloud3D(i, 2) = filteredPointsMat.at<double>(i, 2);
        }
    }
    else {
        // Convert all points to Eigen::MatrixXd
        pointCloud3D.resize(points3D.size(), 3);
        for (size_t i = 0; i < points3D.size(); ++i) {
            pointCloud3D(i, 0) = points3D[i].x;
            pointCloud3D(i, 1) = points3D[i].y;
            pointCloud3D(i, 2) = points3D[i].z;
        }
    }
}

void BoxPoseEstimator::calculateRotationMatrixIterative(const Eigen::Matrix3d& pcaComponents, Eigen::Matrix3d& rotationMatrix) {
    auto alignToAxis = [](const Eigen::Vector3d& mainAxis, const Eigen::Vector3d& targetAxis) {
        Eigen::Vector3d rotationAxis = mainAxis.cross(targetAxis);
        double angle = acos(mainAxis.dot(targetAxis));
        if (rotationAxis.norm() != 0) {
            rotationAxis.normalize();
        }
        Eigen::Matrix3d K;
        K << 0, -rotationAxis.z(), rotationAxis.y(),
            rotationAxis.z(), 0, -rotationAxis.x(),
            -rotationAxis.y(), rotationAxis.x(), 0;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R = I + sin(angle) * K + (1 - cos(angle)) * K * K;
        return R;
        };

    Eigen::Matrix3d pcaComponentsT = pcaComponents;

    Eigen::Matrix3d rotationMatrixX = alignToAxis(pcaComponentsT.col(0), Eigen::Vector3d(1, 0, 0));
    Eigen::Matrix3d rotatedPcaComponentsX = rotationMatrixX * pcaComponentsT;

    Eigen::Matrix3d rotationMatrixY = alignToAxis(rotatedPcaComponentsX.col(1), Eigen::Vector3d(0, 1, 0));
    Eigen::Matrix3d rotatedPcaComponentsY = rotationMatrixY * rotatedPcaComponentsX;

    rotationMatrix = rotationMatrixY * rotationMatrixX;
}

bool BoxPoseEstimator::isWithinTolerance(double length, double templateDim) {
    if (templateDim == 0) return false;
    double ar = length / templateDim;
    if (ar > 1) ar = 1 / ar;
    return ar >= 0.8;
}

void BoxPoseEstimator::determineSideToXYAxis(const TemplateDimBOX &templateDims3D, const DepthImage& depthImage, const Eigen::MatrixXd& rotatedBrick3D, std::vector<AxisDimension>& matchedDims) {
    Eigen::Vector3d minCoords = rotatedBrick3D.colwise().minCoeff();
    Eigen::Vector3d maxCoords = rotatedBrick3D.colwise().maxCoeff();
    Eigen::Vector3d lengths = maxCoords - minCoords;
    Eigen::Vector3d templateLegth = Eigen::Vector3d(templateDims3D.dims[0], templateDims3D.dims[1], templateDims3D.dims[2]);
    std::vector<Dimension> templateDimsOrder = templateDims3D.dimOrder; 
    LOG(VerbosityLevel::MEDIUM, "lengths: " + std::to_string(lengths[0]) + ", " + std::to_string(lengths[1]) + ", " + std::to_string(lengths[2]) + " mm");
    LOG(VerbosityLevel::MEDIUM, "templateDims3D: " + std::to_string(templateLegth[0]) + ", " + std::to_string(templateLegth[1]) + ", " + std::to_string(templateLegth[2]) + " mm");
    float scaleTemplateToDepth = (templateDims3D.unit == depthImage.unit) ? 1.0f :  getScaleFactorBetweenUnits(templateDims3D.unit, depthImage.unit);
    scaleTemplateToDepth = scaleTemplateToDepth * depthImage.depthScale;
    auto checkMatch = [&](double axisSize, Dimension& match) {
        for (int j = 0; j < templateDimsOrder.size(); j++) {
            if (isWithinTolerance(axisSize * scaleTemplateToDepth, templateLegth[j])) {
                match = templateDimsOrder[j];
                return true;
            }
        }
        return false;
        };
    for (int i = 0; i < 2; i++) { // only XY dimensions
        Dimension match;
        if (checkMatch(lengths[i], match)) {
            matchedDims.push_back({ static_cast<Axis>(i), match });
        }
    }
}

void BoxPoseEstimator::adjustRotationToAlignAxes(const std::vector<AxisDimension>& matchedDims, Eigen::Matrix3d& rotationMatrix, bool& changed) {
    Eigen::Matrix3d rotationAlign = Eigen::Matrix3d::Identity();
    for (const auto& axisDim : matchedDims) {
        Eigen::Vector3d currentAxis = Eigen::Vector3d{mCameraAxisOrder[axisDim.axis][0], mCameraAxisOrder[axisDim.axis][1], mCameraAxisOrder[axisDim.axis][2]};
        Eigen::Vector3d targetAxis = Eigen::Vector3d{ mCameraAxisOrder[mObjectCoordinateSystem[axisDim.dimension]][0], mCameraAxisOrder[mObjectCoordinateSystem[axisDim.dimension]][1], mCameraAxisOrder[mObjectCoordinateSystem[axisDim.dimension]][2] };

        if (!currentAxis.isApprox(targetAxis)) {
            changed = true;
            LOG(VerbosityLevel::HIGH, "Adjusting rotation to align " + std::to_string(static_cast<int>(axisDim.axis)) + " axis with " + std::to_string(static_cast<int>(axisDim.dimension)) + " orientation");
            Eigen::Vector3d v = currentAxis.cross(targetAxis);
            double c = currentAxis.dot(targetAxis);
            double s = v.norm();

            if (s != 0) { // rodriquez formula
                Eigen::Matrix3d K;
                K << 0, -v.z(), v.y(),
                    v.z(), 0, -v.x(),
                    -v.y(), v.x(), 0;
                Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d R = I + K + K * K * ((1 - c) / (s * s));
                rotationAlign = R * rotationAlign;
            }
        }
    }

    rotationMatrix = rotationAlign * rotationMatrix;
}

Eigen::Vector3d BoxPoseEstimator::rotationMatrixToEulerAngles(const Eigen::Matrix3d& rotationMatrix) {
    double sinY = -rotationMatrix(2, 0);
    double cosY = std::sqrt(1 - sinY * sinY);

    double thetaX, thetaY, thetaZ;

    if (cosY > 1e-6) { // Not at gimbal lock
        thetaX = std::atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));
        thetaY = std::asin(sinY);
        thetaZ = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));
    }
    else { // Gimbal lock
        thetaX = std::atan2(-rotationMatrix(0, 1), rotationMatrix(1, 1));
        thetaY = std::asin(sinY);
        thetaZ = 0;
    }

    return Eigen::Vector3d(thetaX, thetaY, thetaZ);
}
