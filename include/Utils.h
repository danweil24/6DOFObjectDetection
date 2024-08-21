#pragma once
#include "CommonTypes.h"


/// Helper functions

std::string sizeUnitToString(SIZEUNIT unit);
std::string dimensionToString(Dimension dim);
std::string axisToString(Axis axis);
std::string templateDimBOXToString(TemplateDimBOX toSrc);

float getScaleFactorBetweenUnits(SIZEUNIT from, SIZEUNIT to);


/**
 * @class Preprocessor
 * @brief A class to preprocess images by resizing, normalizing, and padding while optionally converting the image to CWH format.
 */
class Preprocessor {
public:
    /**
     * @brief Preprocess the input image by resizing, normalizing, and padding.
     * @param inputRgb The input RGB image.
     * @param outputRgb The preprocessed output RGB image.
     * @param newH The new height of the image after resizing.
     * @param newW The new width of the image after resizing.
     * @param params The preprocessing parameters.
     */
    static void preprocessImage(const cv::Mat& inputRgb, cv::Mat& outputRgb, int& newH, int& newW, const PreprocessParams &params);

private:
    /**
     * @brief Resize the image while keeping the aspect ratio and pad if necessary.
     * @param inputRgb The input RGB image.
     * @param outputRgb The resized and padded output RGB image.
     * @param newH The new height of the image after resizing.
     * @param newW The new width of the image after resizing.
     * @param params The preprocessing parameters.
     */
    static void resizeKeepAspectAndPad(const cv::Mat& inputRgb, cv::Mat& outputRgb, int& newH, int& newW, const PreprocessParams& params);

    /**
     * @brief Normalize and pad the image channels individually.
     * @param x The input image.
     * @param params The preprocessing parameters.
     * @return The normalized and padded image.
     */
    static cv::Mat normPad(const cv::Mat& x, const PreprocessParams& params);
};



// simple helper function to get the value from cv::Mat and cast it to the desired type
template <typename CAST_TO>
CAST_TO getCVMatValue(const cv::Mat& mat, const cv::Point& point) {
    switch (mat.type()) {
		case CV_16U:
			return static_cast<CAST_TO>(mat.at<uint16_t>(point));
		case CV_32F:
			return static_cast<CAST_TO>(mat.at<float>(point));
        case CV_64F:
			return static_cast<CAST_TO>(mat.at<double>(point));
		default:
			throw std::runtime_error("Unsupported depth image type");
	}   
}

enum class VerbosityLevel {
    NONE,       ///< No debug prints
    LOW,        ///< Low verbosity debug prints
    MEDIUM,     ///< Medium verbosity debug prints
    HIGH,       ///< High verbosity debug prints
    ERR       ///< Error prints
};

class Logger {
public:
    Logger(VerbosityLevel level) : verbosityLevel(level) {}

    void setVerbosityLevel(VerbosityLevel level) {
        verbosityLevel = level;
    }

    void log(VerbosityLevel level, const std::string& message) {
        if (level <= verbosityLevel) {
            std::cout << message << std::endl;
        }
        if (level == VerbosityLevel::ERR) {
            std::cerr << message << std::endl;
        }
    }

private:
    VerbosityLevel verbosityLevel;
};

// Define the global logger instance
extern Logger globalLogger;

// Define the logging macro
#define LOG(level, message) globalLogger.log(level, message)


/// helper function 
std::wstring to_wstring(const std::string& str);
CameraParams readCameraParams(const std::string& cameraParamsJson);
void uvToXyz(const cv::Point2d& uv, double depth, const CameraParams& cameraParams, cv::Point3d& xyz);
void xyzToUv(const cv::Point3d& xyz, double depth, const CameraParams& cameraParams, cv::Point2d& uv);