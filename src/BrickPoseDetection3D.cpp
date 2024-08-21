
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include "BrickPoseDetection3D.h"



BrickPoseDetection3D::BrickPoseDetection3D(const std::string& modelBasePath, const PoseHyperParams& hyperParams) {
    mSegmentationModel = std::make_unique<SegAnythingONNX>(modelBasePath + hyperParams.encoderFileName , modelBasePath + hyperParams.decoderFileName);
    mRotateEstimator = std::make_unique<BoxPoseEstimator>(hyperParams.templateDimBox, hyperParams.objectCoordinateSystem, hyperParams.cameraAxisOrder);
    mHyperParams = hyperParams;
    LOG(VerbosityLevel::LOW, "Inized Brick BrickPoseDetection3D hyperparams " + mHyperParams.toString());
}


cv::Size BrickPoseDetection3D::getModelInputSize() const {
    return mSegmentationModel->getModelInputSize();
}


void BrickPoseDetection3D::calculateCropForNetwork(const cv::Size& imageSize, const cv::Size& networkInputSize, const cv::Point& detectedBrickPoint, cv::Rect& cropRect) {
    cv::Point center(imageSize.width / 2, imageSize.height / 2);
    int width = std::min(networkInputSize.width, imageSize.width);
    int height = std::min(networkInputSize.height, imageSize.height);
    int startX = std::max(0, center.x - width / 2);
    int startY = std::max(0, center.y - height / 2);

    if (detectedBrickPoint.x < startX || detectedBrickPoint.x > startX + width || detectedBrickPoint.y < startY || detectedBrickPoint.y > startY + height) {
        startX = std::max(0, detectedBrickPoint.x - width / 2);
        startY = std::max(0, detectedBrickPoint.y - height / 2);
    }

    cropRect = cv::Rect(startX, startY, width, height);
}


void BrickPoseDetection3D::findPointInMiddleBrick(const DepthImage& depthMap, cv::Point& pointOfInterest) {
    // Create wall mask based on depth thresholds 
    cv::Mat wallMask;
    // Scale the depth thresholds if the depth map is in a different unit
    float scaleThresholdToDepth = (mHyperParams.workingSizeUnit == depthMap.unit) ? 1.0f : getScaleFactorBetweenUnits(depthMap.unit, WORKING_SIZE_UNIT);
    if (depthMap.depthScale == 0) {
        throw std::runtime_error("Depth scale is zero");
	}
    scaleThresholdToDepth = scaleThresholdToDepth / depthMap.depthScale ;
    cv::inRange(depthMap.depthImageMat, mHyperParams.wallDepthThresholdLow * scaleThresholdToDepth, mHyperParams.wallDepthThresholdHigh * scaleThresholdToDepth, wallMask);

    // Calculate the center of the wall mask
    int midY = wallMask.rows / 2;
    int midX = wallMask.cols / 2;

    // Find non-zero points in the wall mask
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(wallMask, nonZeroPoints);

    if (!nonZeroPoints.empty()) {
        // Find the point closest to the center
        double minDist = std::numeric_limits<double>::max();
        cv::Point centerPoint;

        for (const auto& point : nonZeroPoints) {
            double dist = std::sqrt(std::pow(point.y - midY, 2) + std::pow(point.x - midX, 2));
            if (dist < minDist) {
                minDist = dist;
                centerPoint = point;
            }
        }

        // Offset the center point
        centerPoint.y = std::min(centerPoint.y + POINT_OFFSET_Y_DIR, depthMap.depthImageMat.rows - 1);
        pointOfInterest = centerPoint; 

    }
    else {
        LOG(VerbosityLevel::LOW, "No valid points found in the wall mask, taken min depth " 
            + std::to_string(WALL_DEPTH_THRESHOLD_LOW) + "max depth " + \
            std::to_string(WALL_DEPTH_THRESHOLD_HIGH)\
        + "units " +sizeUnitToString(WORKING_SIZE_UNIT));
        pointOfInterest = cv::Point(midX, midY);
    }
}

BrickPoseDetection3DStatus BrickPoseDetection3D::process(const cv::Mat& rgbImage, const DepthImage& depthImage, const CameraParams& cameraParams, Pose3D &brick3DPose, std::optional<std::string> dumpPath){
    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat maskImage;
    auto status = getBrick2DMask(rgbImage, depthImage, maskImage);

    if (status == BrickPoseDetection3DStatus::FAIL) {
        return BrickPoseDetection3DStatus::NO_BRICK_DETECTED;
    }

    if (dumpPath) {
        std::string dumpPathStr = dumpPath.value() + "uintMaskImageW_" + std::to_string(maskImage.cols) + "_H_" + std::to_string(maskImage.rows) + ".raw";
        char * maskData = maskImage.ptr<char>(0);
        std::ofstream maskFile(dumpPathStr, std::ios::binary);
        maskFile.write(reinterpret_cast<char*>(maskData), maskImage.total() * maskImage.channels() * sizeof(char));
    }

    // Perform rotation detection
    PoseEstimatorStatus poseStatus = mRotateEstimator->detectPose(rgbImage, maskImage, depthImage, cameraParams, brick3DPose);
    if (poseStatus != PoseEstimatorStatus::SUCCESS) {
		return BrickPoseDetection3DStatus::FAIL; // Return empty output on failure
	}
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LOG(VerbosityLevel::LOW, "BrickPoseDetection3D::process took " + std::to_string(duration) + " ms");
    return BrickPoseDetection3DStatus::SUCCESS;
}

BrickPoseDetection3DStatus BrickPoseDetection3D::getBrick2DMask(const cv::Mat& rgbImage, const DepthImage& depthImage, cv::Mat& maskImage) {
    // Assume center of the image as the brick point for the first part
    cv::Point centerPointBrick;
    findPointInMiddleBrick(depthImage, centerPointBrick);

    LOG(VerbosityLevel::MEDIUM, "centerPointBrick: x " + std::to_string(centerPointBrick.x) + " y " + std::to_string(centerPointBrick.y));

    // Calculate crop rectangle based on network input size and center point
    cv::Rect cropRect;
    calculateCropForNetwork(rgbImage.size(), getModelInputSize(), centerPointBrick, cropRect);

    // Crop the center of the image
    cv::Mat centerCroppedImage = rgbImage(cropRect);

    // Adjust brick point to the cropped image
    cv::Point brickPointCentered = centerPointBrick - cropRect.tl();

    // Run segmentation model
    SegOutput segOutput = mSegmentationModel->process(centerCroppedImage, brickPointCentered);

    // Check if segmentation output is valid
    if (segOutput.mSegmentations.empty() || segOutput.mSegmentations[0].empty()) {
        std::cerr << "Segmentation output is empty!" << std::endl;
        return BrickPoseDetection3DStatus::FAIL;
    }

    // Convert segmentation result to binary mask with 0 and 1 values
    cv::Mat cropMask;
    cv::threshold(segOutput.mSegmentations[0], cropMask, 0, 1, cv::THRESH_BINARY);

    // Ensure cropMask is CV_8U for connectedComponentsWithStats
    cropMask.convertTo(cropMask, CV_8U);

    // Create full mask and place the segmented mask back to its original location
    cv::Mat fullMask = cv::Mat::zeros(rgbImage.size(), CV_8U); // CV_8U is necessary for connectedComponentsWithStats
    cropMask.copyTo(fullMask(cropRect)); // Copy cropMask to the ROI in fullMask

    // Post-process the full mask
    filterFullMask(fullMask, centerPointBrick, maskImage);

    return BrickPoseDetection3DStatus::SUCCESS;
}

void BrickPoseDetection3D::filterFullMask(const cv::Mat& fullMask, const cv::Point& inputPoint, cv::Mat& filteredMask) {
    if (!(0 <= inputPoint.x && inputPoint.x < fullMask.cols && 0 <= inputPoint.y && inputPoint.y < fullMask.rows)) {
        throw std::runtime_error("Input point is outside the mask boundaries");
    }

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(fullMask, labels, stats, centroids);

    int inputLabel = labels.at<int>(inputPoint);
    filteredMask = (labels == inputLabel);
}


