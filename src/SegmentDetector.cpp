#include "SegmentDetector.h"
#include <chrono>
#include <codecvt>
#include <locale>
#include <fstream>

constexpr int CHANNELS = 3;
constexpr int DEFAULT_OUTPUT_SIZE = 640;
constexpr float DEFAULT_POINT_LABEL = 1.0f;
constexpr int BATCH_SIZE = 1;
constexpr int POINT_DIM = 2;
constexpr int SINGLE_DIM = 1;

constexpr std::array<int64_t, 4> ENCODER_INPUT_SHAPE = { BATCH_SIZE, CHANNELS, DEFAULT_OUTPUT_SIZE, DEFAULT_OUTPUT_SIZE };
constexpr std::array<int64_t, 3> POINT_INPUT_SHAPE = { BATCH_SIZE, SINGLE_DIM, POINT_DIM };
constexpr std::array<int64_t, 2> LABEL_INPUT_SHAPE = { BATCH_SIZE, SINGLE_DIM };
constexpr std::array<int64_t, SINGLE_DIM> ORIG_IM_SIZE_INPUT_SHAPE = { POINT_DIM };

constexpr const char* ENCODER_INPUT_NAME = "input_image";
constexpr const char* ENCODER_OUTPUT_NAME = "image_embeddings";
constexpr const char* DECODER_INPUT_NAMES[] = { "image_embeddings", "point_coords", "point_labels", "orig_im_size" };
constexpr const char* DECODER_OUTPUT_NAMES[] = { "masks", "iou_predictions" };

SegAnythingONNX::SegAnythingONNX(const std::string& encoderPath, const std::string& decoderPath)
    : mEnv(ORT_LOGGING_LEVEL_WARNING, "BrickSegmentation"),
    mSessionOptions(),
    mEncoderSession(mEnv, to_wstring(encoderPath).c_str(), mSessionOptions),  // check creation 
    mDecoderSession(mEnv, to_wstring(decoderPath).c_str(), mSessionOptions) {  // check creation

    initializePreprocessParams();
}

void SegAnythingONNX::initializePreprocessParams() {
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputShape = mEncoderSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inputShape.size() == 4 && inputShape[2] == inputShape[3]) {
        mPreprocessParams.mOutputSize = inputShape[2];
    }
}

SegOutput SegDetectionModel2D::process(cv::Mat& image, std::optional<cv::Point> pointOfInterest, std::optional<std::string> dumpPath) {
    cv::Mat inputRgb;
    startTime = std::chrono::high_resolution_clock::now();
    int newH, newW;
    mOriginalInputSize = image.size();
    mPreprocessor.preprocessImage(image, inputRgb, newH, newW, mPreprocessParams);
    mResizedInputSize = cv::Size(newW, newH);
    auto endTimePreProcess = std::chrono::high_resolution_clock::now();
    double preprocessTime = std::chrono::duration<double, std::milli>(endTimePreProcess - startTime).count();
    return processInternal(inputRgb, pointOfInterest, dumpPath);
}

SegOutput SegAnythingONNX::processInternal(const cv::Mat& resizedImage, std::optional<cv::Point> pointOfInterest, std::optional<std::string> dumpPath) {
    if (dumpPath) {
        std::string dumpInputP = dumpPath.value() + "preprocessed_image_" + std::to_string(mPreprocessParams.mOutputSize) + ".raw";
        std::ofstream dumpInput(dumpInputP, std::ios::binary);
        if (dumpInput.is_open()) {
            dumpInput.write(reinterpret_cast<char*>(resizedImage.data), resizedImage.total() * resizedImage.channels() * sizeof(float));
            dumpInput.close();
        }
        else {
            LOG(VerbosityLevel::ERR, "Failed to open file for writing: " + dumpInputP);
        }
    }

    std::vector<float> pointLabels = { DEFAULT_POINT_LABEL };
    std::array<int64_t, 4> inputShape = ENCODER_INPUT_SHAPE;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(resizedImage.ptr<float>()), resizedImage.total() * resizedImage.channels(), inputShape.data(), inputShape.size());

    auto starttime = std::chrono::high_resolution_clock::now();
    const char* inputNames[] = { ENCODER_INPUT_NAME };
    const char* outputNames[] = { ENCODER_OUTPUT_NAME };
    auto imageEmbeddings = mEncoderSession.Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);

    auto endTimeEnd = std::chrono::high_resolution_clock::now();
    double encoderTime = std::chrono::duration<double, std::milli>(endTimeEnd - starttime).count();
    LOG(VerbosityLevel::HIGH, "Encoder inference time: " + std::to_string(encoderTime) + " ms");

    if (dumpPath) {
        std::string dumpInputE = dumpPath.value() + "encoder_output_" + std::to_string(mPreprocessParams.mOutputSize) + ".raw";
        std::ofstream dumpInput(dumpInputE, std::ios::binary);
        if (dumpInput.is_open()) {
            dumpInput.write(reinterpret_cast<char*>(imageEmbeddings[0].GetTensorMutableData<float>()), imageEmbeddings[0].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
            dumpInput.close();
        }
        else {
            LOG(VerbosityLevel::ERR, "Failed to open file for writing: " + dumpInputE);
        }
    }

    SegOutput output;
    if (pointOfInterest) {
        cv::Point2f adjPoint(pointOfInterest->x * (static_cast<float>(mResizedInputSize.width) / mOriginalInputSize.width), pointOfInterest->y * (static_cast<float>(mResizedInputSize.height) / mOriginalInputSize.height));
        std::vector<float> pointData = { adjPoint.x, adjPoint.y };
        std::array<int64_t, 3> pointShape = POINT_INPUT_SHAPE;
        Ort::Value pointTensor = Ort::Value::CreateTensor<float>(memoryInfo, pointData.data(), pointData.size(), pointShape.data(), pointShape.size());

        std::array<int64_t, 2> labelShape = LABEL_INPUT_SHAPE;
        Ort::Value labelTensor = Ort::Value::CreateTensor<float>(memoryInfo, pointLabels.data(), pointLabels.size(), labelShape.data(), labelShape.size());

        std::array<int64_t, 1> origImSizeShape = ORIG_IM_SIZE_INPUT_SHAPE;
        std::vector<float> origImSizeData = { static_cast<float>(mOriginalInputSize.height), static_cast<float>(mOriginalInputSize.width) };
        Ort::Value origImSizeTensor = Ort::Value::CreateTensor<float>(memoryInfo, origImSizeData.data(), origImSizeData.size(), origImSizeShape.data(), origImSizeShape.size());

        // Decoder inference
        try {
            startTime = std::chrono::high_resolution_clock::now();
            const char* decoderInputNames[] = { DECODER_INPUT_NAMES[0], DECODER_INPUT_NAMES[1], DECODER_INPUT_NAMES[2], DECODER_INPUT_NAMES[3] };
            const char* decoderOutputNames[] = { DECODER_OUTPUT_NAMES[0], DECODER_OUTPUT_NAMES[1] };

            // Make sure to pass all the required inputs to the Run function
            std::array<Ort::Value, 4> inputTensors = { std::move(imageEmbeddings[0]), std::move(pointTensor), std::move(labelTensor), std::move(origImSizeTensor) };

            auto masksOnnx = mDecoderSession.Run(Ort::RunOptions{ nullptr }, decoderInputNames, inputTensors.data(), inputTensors.size(), decoderOutputNames, std::size(decoderOutputNames));
            auto decoderEndTime = std::chrono::high_resolution_clock::now();
            double decoderTime = std::chrono::duration<double, std::milli>(decoderEndTime - endTimeEnd).count();

            LOG(VerbosityLevel::HIGH, "Decoder inference time: " + std::to_string(decoderTime) + " ms");

            // Correctly create the cv::Mat from the ONNX output tensors
            auto masksShape = masksOnnx[0].GetTensorTypeAndShapeInfo().GetShape();
            auto iouPredictionsShape = masksOnnx[1].GetTensorTypeAndShapeInfo().GetShape();

            // Assuming masksShape is [batch, channels, height, width]
            cv::Mat masks(masksShape[2], masksShape[3], CV_32F, masksOnnx[0].GetTensorMutableData<void>());
            float iouPrediction = masksOnnx[1].At<float>({ 0, 0 });

            output.mSegmentations.push_back(masks);
            output.mMaxScores.push_back(iouPrediction);
            output.mLabels.push_back(SEG_LABELS::BRICK);  // Assuming a label value of 1 for now
        }
        catch (const Ort::Exception& e) {
            LOG(VerbosityLevel::ERR, "Error during decoder inference: " + std::string(e.what()));
        }
    }

    return output;
}
