#ifndef SEGMENT_DETECTION_H
#define SEGMENT_DETECTION_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <optional>
#include <vector>
#include "Utils.h"

enum class SEG_LABELS {
    BACKGROUND = 0,
    BRICK = 1,
    WALL = 2,
    UNKNOWN = 3
};

struct SegOutput {
    std::vector<cv::Mat> mSegmentations;
    std::vector<float> mMaxScores;
    std::vector<SEG_LABELS> mLabels;
};

// Abstract base class for segmentation models
class SegDetectionModel2D {
public:
    virtual SegOutput process(cv::Mat& image, std::optional<cv::Point> pointOfInterest = std::nullopt, std::optional<std::string> dumpPath = std::nullopt);
    virtual ~SegDetectionModel2D() = default;
    virtual cv::Size getModelInputSize() const = 0;

protected:
    Preprocessor mPreprocessor;
    PreprocessParams mPreprocessParams; // Define PreprocessParams as a member
    cv::Size mResizedInputSize;
    cv::Size mOriginalInputSize;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    virtual SegOutput processInternal(const cv::Mat& image, std::optional<cv::Point> pointOfInterest, std::optional<std::string> dumpPath = std::nullopt) = 0;
};

// Derived class for ONNX segmentation model
class SegAnythingONNX : public SegDetectionModel2D {
public:
    SegAnythingONNX(const std::string& encoderPath, const std::string& decoderPath);
    virtual cv::Size getModelInputSize() const override { return  cv::Size(mPreprocessParams.mOutputSize, mPreprocessParams.mOutputSize); }
private:
    Ort::Env mEnv;
    Ort::SessionOptions mSessionOptions;
    Ort::Session mEncoderSession;
    Ort::Session mDecoderSession;
    void initializePreprocessParams(); // Function to initialize preprocess params based on the model input size
    virtual SegOutput processInternal(const cv::Mat& image, std::optional<cv::Point> pointOfInterest, std::optional<std::string> dumpPath) override;
};

#endif // SEGMENT_DETECTION_H
