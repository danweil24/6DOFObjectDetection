#include "crow.h"
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "CommonTypes.h"
#include "BrickPoseDetection3D.h"
#include <mutex>

// Singleton class for BrickPoseDetection3D
class BrickPoseDetection3DWrapper {
public:
    static BrickPoseDetection3D& getInstance(const std::string& base_models_path) {
        static BrickPoseDetection3D instance(base_models_path);
        return instance;
    }

private:
    BrickPoseDetection3DWrapper() = default;
    ~BrickPoseDetection3DWrapper() = default;
    BrickPoseDetection3DWrapper(const BrickPoseDetection3DWrapper&) = delete;
    BrickPoseDetection3DWrapper& operator=(const BrickPoseDetection3DWrapper&) = delete;
};

void runBrickPoseDetectionModel(const std::string& base_models_path, const cv::Mat& rgbImage, const cv::Mat& depthImage, const std::string& camParamsPath, std::string& result) {
    BrickPoseDetection3D& brickPoseModel = BrickPoseDetection3DWrapper::getInstance(base_models_path);

    // Load camera parameters
    CameraParams cameraParams = readCameraParams(camParamsPath);

    // Process the images
    DepthImage depthImageStruct(depthImage, SIZEUNIT::MM, 0.1f);
    Pose3D brickPose;
    brickPoseModel.process(rgbImage, depthImageStruct, cameraParams, brickPose);

    // Convert the pose to JSON
    nlohmann::json poseJson;
    poseJson["position"] = { brickPose.translation[0], brickPose.translation[1], brickPose.translation[2] };
    poseJson["orientation"] = { brickPose.rotation[0], brickPose.rotation[1], brickPose.rotation[2] };
    poseJson["path"] = rgbImage;

    result = poseJson.dump();
}

VerbosityLevel stringToVerbosityLevel(const std::string& level) {
    if (level == "NONE") return VerbosityLevel::NONE;
    if (level == "LOW") return VerbosityLevel::LOW;
    if (level == "MEDIUM") return VerbosityLevel::MEDIUM;
    if (level == "HIGH") return VerbosityLevel::HIGH;
    if (level == "ERR") return VerbosityLevel::ERR;
    return VerbosityLevel::LOW;  // Default
}

void processSingleRequest(const std::string& base_models_path, const crow::json::rvalue& bodyArgs, crow::response& res) {
    std::string imagePath, depthPath, camParamsPath;
    if (bodyArgs.has("basePath")) {
        std::string basePath = bodyArgs["basePath"].s();
        basePath = basePath + "\\";
        imagePath = basePath + "color.png";
        depthPath = basePath + "depth.png";
        camParamsPath = basePath + "cam.json";
    }
    else {
        imagePath = bodyArgs["imagePath"].s();
        depthPath = bodyArgs["depthPath"].s();
        camParamsPath = bodyArgs["camParamsPath"].s();
    }

    // Load RGB image
    cv::Mat rgbImage = cv::imread(imagePath);
    if (rgbImage.empty()) {
        res = crow::response(400, "Failed to load image: " + imagePath);
        return;
    }
    cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2RGB);

    // Load depth image
    cv::Mat depthImage = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
    if (depthImage.empty()) {
        res = crow::response(400, "Failed to load depth image: " + depthPath);
        return;
    }

    std::string result;
    try {
        runBrickPoseDetectionModel(base_models_path, rgbImage, depthImage, camParamsPath, result);
    }
    catch (const std::exception& e) {
        res = crow::response(500, e.what());
        return;
    }

    res = crow::response(200, result);
}

void processManyRequests(const std::string& base_models_path, const std::string& manyBasePath, crow::response& res) {
    std::string responseContent;
    BrickPoseDetection3D& brickPoseModel = BrickPoseDetection3DWrapper::getInstance(base_models_path);

    for (const auto& entry : std::filesystem::directory_iterator(manyBasePath)) {
        if (entry.is_directory()) {
            std::string basePath = entry.path().string() + "\\";
            crow::json::wvalue subRequest;
            subRequest["basePath"] = basePath;
            crow::json::rvalue subRequestRvalue = crow::json::load(subRequest.dump());
            crow::response subRes;

            // Process the request with the single instance
            std::string imagePath = basePath + "color.png";
            std::string depthPath = basePath + "depth.png";
            std::string camParamsPath = basePath + "cam.json";

            // Load RGB image
            cv::Mat rgbImage = cv::imread(imagePath);
            if (rgbImage.empty()) {
                subRes = crow::response(400, "Failed to load image: " + imagePath);
                responseContent += subRes.body + "\n";
                continue;
            }
            cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2RGB);

            // Load depth image
            cv::Mat depthImage = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
            if (depthImage.empty()) {
                subRes = crow::response(400, "Failed to load depth image: " + depthPath);
                responseContent += subRes.body + "\n";
                continue;
            }

            std::string result;
            try {
                CameraParams cameraParams = readCameraParams(camParamsPath);
                DepthImage depthImageStruct(depthImage, SIZEUNIT::MM, 0.1f);
                Pose3D brickPose;
                brickPoseModel.process(rgbImage, depthImageStruct, cameraParams, brickPose);

                nlohmann::json poseJson;
                poseJson["path"] = basePath;
                poseJson["position"] = { brickPose.translation[0], brickPose.translation[1], brickPose.translation[2] };
                poseJson["orientation"] = { brickPose.rotation[0], brickPose.rotation[1], brickPose.rotation[2] };
                
                result = poseJson.dump();
                subRes = crow::response(200, result);
            }
            catch (const std::exception& e) {
                subRes = crow::response(500, e.what());
            }

            responseContent += subRes.body + "\n";
        }
    }

    res = crow::response(200, responseContent);
}



int main(int argc, char* argv[]) {
    crow::SimpleApp app;

    std::string default_models_path = ".\\models\\";
    std::string models_path = default_models_path;

    CROW_ROUTE(app, "/process").methods(crow::HTTPMethod::POST)([&models_path](const crow::request& req) {
        auto bodyArgs = crow::json::load(req.body);
        if (!bodyArgs) {
            return crow::response(400, "Invalid JSON");
        }

        // Set models path if provided
        if (bodyArgs.has("modelsPath")) {
            models_path = bodyArgs["modelsPath"].s();
            models_path = models_path + "\\";
        }

        // Set verbosity level if provided
        if (bodyArgs.has("verbosity")) {
            std::string verbosity = bodyArgs["verbosity"].s();
            globalLogger.setVerbosityLevel(stringToVerbosityLevel(verbosity));
        }

        bool processMany = false;
        if (bodyArgs.has("multi")) {
            processMany = bodyArgs["multi"].b();
        }

        crow::response res;

        if (processMany) {
            std::string basePath = bodyArgs["basePath"].s();
            basePath = basePath + "\\";
            processManyRequests(models_path, basePath, res);
        }
        else {
            processSingleRequest(models_path, bodyArgs, res);
        }

        return res;
        });

    app.port(8080).multithreaded().run();
}

