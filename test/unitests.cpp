#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include "PoseEstimator.h"
#include "SegmentDetector.h"
#include <opencv2/core/eigen.hpp>
#include "BrickPoseDetection3D.h"
#include "CommonTypes.h"

void testSegmentationModel(int numOfImage) {
    std::string base_models_path = "C:\\development\\Repository\\BrickDetection\\BrickDetection\\models\\";
    std::string encoderPath = base_models_path+ "sam_vit_b_01ec64_640_encoder_simplified.onnx";
    std::string decoderPath = base_models_path + "sam_vit_b_01ec64_640_decoder_simplified.onnx";
    SegAnythingONNX segModel(encoderPath, decoderPath);

    for (int i = 0; i < numOfImage; i++) {
        std::cout << "Processing image " << i << std::endl;
        std::string input_base_path = "C:\\development\\Repository\\BrickDetection\\BrickDetection\\20240808_640\\" + std::to_string(i) + "\\";

        std::string imagePath = input_base_path + "color.png";

        // Load image
     // Load image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }

        // Convert the image from BGR to RGB
        cv::Mat imageRGB = image;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

        // Perform segmentation
        cv::Point pointOfInterest; // middle of the image
        pointOfInterest.x = imageRGB.cols / 2;
        pointOfInterest.y = imageRGB.rows / 2;
        SegOutput segOutput = segModel.process(imageRGB, pointOfInterest);

        // Save the segmentation result
        std::string segOutputPath = input_base_path + "segmentation.raw";
        float * segOutputData = segOutput.mSegmentations[0].ptr<float>(0);
        std::ofstream segOutputFile(segOutputPath, std::ios::binary);
        segOutputFile.write(reinterpret_cast<char*>(segOutputData), segOutput.mSegmentations[0].total() * segOutput.mSegmentations[0].channels() * sizeof(float));
    }
}
void testBrickPoseDetectionModel(int numOfImage) {
    std::string base_models_path = "C:\\development\\Repository\\BrickDetection\\BrickDetection\\models\\";
    BrickPoseDetection3D brickPoseModel(base_models_path);

    for (int i = 0; i < numOfImage; i++) {
        std::cout << "Processing image " << i << std::endl;
        std::string input_base_path = "C:\\development\\Repository\\BrickDetection\\BrickDetection\\20240808_640\\" + std::to_string(i) + "\\";

        std::string imagePath = input_base_path + "color.png";
        std::string depthPath = input_base_path + "depth.png";
        std::string camParamsPath = input_base_path + "cam.json";

        // Load RGB image
        cv::Mat rgbImage = cv::imread(imagePath);
        if (rgbImage.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2RGB);

        // Load depth image
        cv::Mat depthImage = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
        if (depthImage.empty()) {
            throw std::runtime_error("Failed to load depth image: " + depthPath);
        }

        // Load camera parameters
        CameraParams cameraParams = readCameraParams(camParamsPath);

        // Process the images
        DepthImage depthImageStruct(depthImage, SIZEUNIT::MM, 0.1f);
        Pose3D brickPose;
        brickPoseModel.process(rgbImage, depthImageStruct, cameraParams, brickPose,input_base_path);

        // Save the segmentation result
        std::string segOutputPath = input_base_path + "result.txt";
        brickPose.dump(segOutputPath);
    }
}
void testRotationEstimation(int numOfImage) {
    for (int i = 0; i < numOfImage; i++) {
        std::cout << "Processing image " << i << std::endl;
        std::string input_base_path = "C:\\development\\Repository\\BrickDetection\\BrickDetection\\20240808_640\\" + std::to_string(i) + "\\";

        std::string imagePath = input_base_path + "color.png";
        std::string jsonPath = input_base_path + "cam.json";
        std::string segFilePath = input_base_path + "seg.raw";
        std::string depthFilePath = input_base_path + "depth.png";

        cv::Mat segImage = cv::imread(segFilePath, cv::IMREAD_UNCHANGED);
        if (segImage.empty()) {
			std::cerr << "Failed to load segmentation image: " << segFilePath << std::endl;
			return;
		}
        // Load image
        // Load RGB image
        cv::Mat rgbImage = cv::imread(imagePath);
        if (rgbImage.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2RGB);

        // Load depth image
        cv::Mat depthImage = cv::imread(depthFilePath, cv::IMREAD_UNCHANGED);
        if (depthImage.empty()) {
            throw std::runtime_error("Failed to load depth image: " + depthFilePath);
        }

        // Initialize RotateEstimator
        PoseHyperParams poseHyper;
        BoxPoseEstimator rotateEstimator(poseHyper.templateDimBox, poseHyper.objectCoordinateSystem, poseHyper.cameraAxisOrder);

        // Perform rotation detection
        DepthImage depthImageStruct(depthImage, SIZEUNIT::MM, 0.1f);
        CameraParams cameraParams = readCameraParams(jsonPath);
        Pose3D poseDetected;
        rotateEstimator.detectPose(rgbImage, segImage , depthImageStruct, cameraParams, poseDetected);

        // Save to file 
        std::string out_path = input_base_path;
        poseDetected.dump(out_path);
    }
}


#ifdef UNITTEST
int main(int argc, char** argv) {


    std::string mode = "pose";
    int numOfImage = 11;
    if (mode == "seg") {
        testSegmentationModel(numOfImage);
    }
    else if (mode == "rot") {
        testRotationEstimation(numOfImage);
    }
    else if (mode == "pose") {
        testBrickPoseDetectionModel(numOfImage);
    }
    else if (mode == "both") {
        testSegmentationModel(numOfImage);
        testRotationEstimation(numOfImage);
    }
    else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return -1;
    }

    return 0;
}
#endif //  UNITTEST

