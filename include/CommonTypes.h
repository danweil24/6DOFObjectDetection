#pragma once
#ifndef COMMONTYPES_H
#define COMMONTYPES_H
#include <opencv2/opencv.hpp>

/// needed types 


 
struct Pose3D { 

    std::vector<double> translation;
    std::vector<double> rotation;
    void dump(const std::string& filePath) const;
};

struct PreprocessParams {
    bool mKeepAR = true;           // Keep Aspect Ratio
    bool mPadImage = true;         // Pad Image
    cv::Scalar mMean = cv::Scalar(123.675, 116.28, 103.53); // Mean for normalization
    cv::Scalar mStd = cv::Scalar(58.395, 57.12, 57.375);    // Standard deviation for normalization
    int mOutputSize = 640;         // Output size for the image
    bool mChangeToCWH = true;         // Transpose the image
};

struct CameraParams {
    double fx;
    double fy;
    double px;
    double py;
};

/**
 * @enum Axis
 * @brief Represents the coordinate axes.
 */
enum class Axis {
    X = 0,  ///< X-axis
    Y,      ///< Y-axis
    Z       ///< Z-axis
};

/**
 * @enum Dimension
 * @brief Represents the dimensions of the brick.
 */
enum class Dimension {
    Width,  ///< Width dimension
    Height, ///< Height dimension
    Depth   ///< Depth dimension
};

/**
 * @struct AxisDimension
 * @brief Associates an axis with a dimension.
 */
struct AxisDimension {
    Axis axis;         ///< The coordinate axis
    Dimension dimension; ///< The brick dimension
};

/**
 * @enum SIZEUNIT
 * @brief Enum to represent the size unit
 */
enum class SIZEUNIT : int {
    MM = 0,
    CM = 1,
    M = 2, // can add more units here
    UNDEFINED
};

struct TemplateDimBOX {
    std::vector<float> dims;
    std::vector<Dimension> dimOrder;
    SIZEUNIT unit;
};

struct DepthImage {
    cv::Mat depthImageMat;
    SIZEUNIT unit;
    float depthScale;
    DepthImage(const cv::Mat& depthImage, SIZEUNIT unit, float depthScale) : depthImageMat(depthImage), unit(unit), depthScale(depthScale) {}

};

///using 
using ObjectCoordinateSystem = std::map<Dimension, Axis>;
using CameraAxisOrder = std::map<Axis, std::vector<float>>;

constexpr std::array<std::pair<Dimension, Axis>, 3> OBJECT_COORDINATE_SYSTEM = { {
    {Dimension::Width, Axis::X},
    {Dimension::Height, Axis::Y},
    {Dimension::Depth, Axis::Z}
} };

constexpr std::array<std::pair<Axis, std::array<float, 3>>, 3> CAMERA_AXIS_ORDER = { {
    {Axis::X, {1.0, 0.0, 0.0}},
    {Axis::Y, {0.0, 1.0, 0.0}},
    {Axis::Z, {0.0, 0.0, 1.0}}
} };

constexpr const char* DECODER_FILE_NAME = "sam_vit_b_01ec64_640_decoder_simplified.onnx";
constexpr const char* ENCODER_FILE_NAME = "sam_vit_b_01ec64_640_encoder_simplified.onnx";
constexpr int WALL_DEPTH_THRESHOLD_LOW = 250; // mm
constexpr int WALL_DEPTH_THRESHOLD_HIGH = 400; //mm
constexpr SIZEUNIT WORKING_SIZE_UNIT = SIZEUNIT::MM;
constexpr int POINT_OFFSET_Y_DIR = 50;
constexpr std::array<float, 3> BRICK_SIZE_MM{ 50.0, 210.0, 100.0 }; // mm 
constexpr std::array<Dimension, 3> TEMPLATE_DIMS_ORDER{ Dimension::Height, Dimension::Width, Dimension::Depth };

/**
 * @brief Structure to hold the hyperparameters for pose estimation.
 */
struct PoseHyperParams {
    std::string encoderFileName; ///< File name of the encoder model
    std::string decoderFileName; ///< File name of the decoder model
    int wallDepthThresholdLow; ///< Lower threshold for wall depth in mm
    int wallDepthThresholdHigh; ///< Upper threshold for wall depth in mm
    SIZEUNIT workingSizeUnit; ///< Unit of measurement for sizes (e.g., mm, cm)
    int pointOffsetYDir; ///< After finding the centered brick upper point, offset in the Y direction for points
    TemplateDimBOX templateDimBox; // Size of the brick in millimeters (length, width, height) and order of dimensions for the template
    ObjectCoordinateSystem objectCoordinateSystem; ///< Mapping of dimensions to axes for the object coordinate system
    CameraAxisOrder cameraAxisOrder; ///< Mapping of camera axes to 3D vectors representing their directions

    /**
     * @brief Constructor to initialize the pose hyperparameters.
     * @param encoderFileName File name of the encoder model
     * @param decoderFileName File name of the decoder model
     * @param wallDepthThresholdLow Lower threshold for wall depth in mm
     * @param wallDepthThresholdHigh Upper threshold for wall depth in mm
     * @param workingSizeUnit Unit of measurement for sizes (e.g., mm, cm)
     * @param pointOffsetYDir Offset in the Y direction for points
     * @param brickSizeMM Size of the brick in millimeters (length, width, height)
     * @param templateDimsOrder Order of dimensions for the template (Height, Width, Depth)
     * @param objectCoordinateSystem Mapping of dimensions to axes for the object coordinate system
     * @param cameraAxisOrder Mapping of camera axes to 3D vectors representing their directions
     */
    PoseHyperParams(
        const std::string& encoderFileName,
        const std::string& decoderFileName,
        int wallDepthThresholdLow,
        int wallDepthThresholdHigh,
        SIZEUNIT workingSizeUnit,
        int pointOffsetYDir,
        TemplateDimBOX templateDimBox,
        const ObjectCoordinateSystem& objectCoordinateSystem,
        const CameraAxisOrder& cameraAxisOrder)
        : encoderFileName(encoderFileName),
        decoderFileName(decoderFileName),
        wallDepthThresholdLow(wallDepthThresholdLow),
        wallDepthThresholdHigh(wallDepthThresholdHigh),
        workingSizeUnit(workingSizeUnit),
        pointOffsetYDir(pointOffsetYDir),
        templateDimBox(templateDimBox),
        objectCoordinateSystem(objectCoordinateSystem),
        cameraAxisOrder(cameraAxisOrder) {}

    /**
    * @brief Default constructor initializing hyperparameters with default values.
    */
    PoseHyperParams()
        : encoderFileName(ENCODER_FILE_NAME),
        decoderFileName(DECODER_FILE_NAME),
        wallDepthThresholdLow(WALL_DEPTH_THRESHOLD_LOW),
        wallDepthThresholdHigh(WALL_DEPTH_THRESHOLD_HIGH),
        workingSizeUnit(WORKING_SIZE_UNIT),
        pointOffsetYDir(POINT_OFFSET_Y_DIR),
        objectCoordinateSystem(OBJECT_COORDINATE_SYSTEM.begin(), OBJECT_COORDINATE_SYSTEM.end()),
        cameraAxisOrder([] {
        CameraAxisOrder order;
        for (const auto& pair : CAMERA_AXIS_ORDER) {
            order[pair.first] = std::vector<float>(pair.second.begin(), pair.second.end());
        }
        return order;
            }()) {
        std::vector<Dimension> templateDims = { TEMPLATE_DIMS_ORDER[0], TEMPLATE_DIMS_ORDER[1], TEMPLATE_DIMS_ORDER[2] };
        std::vector <float> templateSize = { BRICK_SIZE_MM[0], BRICK_SIZE_MM[1],BRICK_SIZE_MM[2] };
        templateDimBox = TemplateDimBOX{ templateSize,templateDims, WORKING_SIZE_UNIT };
    }
    std::string toString() const;
        
};


#endif // COMMONTYPES_H