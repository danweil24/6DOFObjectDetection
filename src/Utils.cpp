#include <iostream>
#include <windows.h>
#include <fstream>
#include <json/json.h>
#include "Utils.h"

// Conversion factors to meters
const float TO_METER_FACTOR[] = {
    0.001f,  // MM to meters
    0.01f,   // CM to meters
    1.0f     // M to meters
};
// Initialize the global logger with the default verbosity level
Logger globalLogger(VerbosityLevel::LOW);

// Preprocessor implementation
void Preprocessor::preprocessImage(const cv::Mat& inputRgb, cv::Mat& outputRgb, int& newH, int& newW, const PreprocessParams& params) {
    if (params.mKeepAR && params.mPadImage) {
        resizeKeepAspectAndPad(inputRgb, outputRgb, newH, newW, params);
    }
}

void Preprocessor::resizeKeepAspectAndPad(const cv::Mat& inputRgb, cv::Mat& outputRgb, int& newH, int& newW, const PreprocessParams& params) {
    double scale = static_cast<double>(params.mOutputSize) / std::max(inputRgb.rows, inputRgb.cols);
    newH = static_cast<int>(inputRgb.rows * scale + 0.5);
    newW = static_cast<int>(inputRgb.cols * scale + 0.5);
    cv::Mat resized;
    cv::resize(inputRgb, resized, { newW, newH }, 0, 0, cv::INTER_LINEAR);
    outputRgb = normPad(resized, params);
}

cv::Mat Preprocessor::normPad(const cv::Mat& x, const PreprocessParams& params) {
    cv::Mat normX;
    x.convertTo(normX, CV_32F); // Convert to float

    // Subtract mean and divide by std
    std::vector<cv::Mat> channels(3);
    cv::split(normX, channels);

    // Normalize each channel and pad if necessary
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - params.mMean[c]) / params.mStd[c];
        if (params.mPadImage) {
            int padH = params.mOutputSize - channels[c].rows;
            int padW = params.mOutputSize - channels[c].cols;
            cv::copyMakeBorder(channels[c], channels[c], 0, padH, 0, padW, cv::BORDER_CONSTANT, 0);
        }
    }

    if (params.mChangeToCWH) {
        cv::Mat chwImage;
        cv::vconcat(channels, chwImage); // Concatenate along the channels dimension
        normX = chwImage;
    }
    else {
        cv::merge(channels, normX);
    }
    return normX;
}

float getScaleFactorBetweenUnits(SIZEUNIT from, SIZEUNIT to)
{
    // size TO meter factor
    int sizeTable = sizeof(TO_METER_FACTOR) / sizeof(TO_METER_FACTOR[0]);
    if (from == SIZEUNIT::UNDEFINED || to == SIZEUNIT::UNDEFINED || static_cast<int>(from) >= sizeTable || static_cast<int>(to) >= sizeTable) {
        throw std::invalid_argument("Invalid SIZEUNIT provided.");
    }

    float fromFactor = TO_METER_FACTOR[static_cast<int>(from)];
    float toFactor = TO_METER_FACTOR[static_cast<int>(to)];

    // Return the scale factor
    return fromFactor / toFactor;
}

// Convert string to wstring
std::wstring to_wstring(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

CameraParams readCameraParams(const std::string& cameraParamsJson) {
    std::ifstream file(cameraParamsJson);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + cameraParamsJson);
    }

    Json::CharReaderBuilder readerBuilder;
    Json::Value root;
    std::string errs;

    if (!Json::parseFromStream(readerBuilder, file, &root, &errs)) {
        throw std::runtime_error("Failed to parse JSON file: " + errs);
    }

    if (!root.isObject() || !root.isMember("fx") || !root.isMember("fy") || !root.isMember("px") || !root.isMember("py")) {
        throw std::runtime_error("Invalid JSON format: missing required fields");
    }

    CameraParams cameraParams;
    cameraParams.fx = root["fx"].asDouble();
    cameraParams.fy = root["fy"].asDouble();
    cameraParams.px = root["px"].asDouble();
    cameraParams.py = root["py"].asDouble();

    return cameraParams;
}

void Pose3D::dump(const std::string& filePath) const
{
    std::ofstream out_file(filePath);
	out_file << "Translation: " << translation[0] << " " << translation[1] << " " << translation[2] << std::endl;
	out_file << "Rotation: " << rotation[0] << " " << rotation[1] << " " << rotation[2] << std::endl;
}

void uvToXyz(const cv::Point2d& uv, double depth, const CameraParams& cameraParams, cv::Point3d& xyz) {
    xyz.x = (uv.x - cameraParams.px) * depth / cameraParams.fx;
    xyz.y = (uv.y - cameraParams.py) * depth / cameraParams.fy;
    xyz.z = depth;
}

void xyzToUv(const cv::Point3d& xyz, double depth, const CameraParams& cameraParams, cv::Point2d& uv) {
    uv.x = xyz.x * cameraParams.fx / depth + cameraParams.px;
	uv.y = xyz.y * cameraParams.fy / depth + cameraParams.py;
}

std::string sizeUnitToString(SIZEUNIT unit) {
    switch (unit) {
    case SIZEUNIT::MM: return "MM";
    case SIZEUNIT::CM: return "CM";
    case SIZEUNIT::M: return "M";
    default: return "UNDEFINED";
    }
}
std::string dimensionToString(Dimension dim)
{
    std::string res = "";
    switch (dim)
    {
       case Dimension::Width:
		res = "Width";
		break;
        case Dimension::Height:
            res = "Height";
        break;
        case Dimension::Depth:
			res = "Depth";
            break;
    }
    return res;
}

std::string axisToString(Axis axis)
{
    std::string res = "";
    switch (axis)
    {
		case Axis::X:
			res = "X";
			break;
		case Axis::Y:
			res = "Y";
			break;
		case Axis::Z:
			res = "Z";
			break;
	}
	return res;
}

std::string templateDimBOXToString(TemplateDimBOX toSrc)
{
    std::string str = "Dims: ";
    for (int i = 0; i < toSrc.dims.size(); i++)
    {
		str += std::to_string(toSrc.dims[i]) + " ";
	}
	str += "\n";
	str += "DimOrder: ";
    for (int i = 0; i < toSrc.dimOrder.size(); i++)
    {
		str += dimensionToString(toSrc.dimOrder[i]) + " ";
	}
	str += "\n";
	str += "Unit: " + sizeUnitToString(toSrc.unit) + "\n";
	return str;
    
}

std::string PoseHyperParams::toString() const {
    // object coordinate system
    std::string str = "Object coordinate system: ";
    for (const auto& [dim, axis] : objectCoordinateSystem) {
        str += dimensionToString(dim) + " -> " + axisToString(axis) + ", ";
    }
	str += "Wall depth threshold low: " + std::to_string(wallDepthThresholdLow) + "\n";
	str += "Wall depth threshold high: " + std::to_string(wallDepthThresholdHigh) + "\n";
	str += "Working size unit: " + sizeUnitToString(workingSizeUnit) + "\n";
    // template 
    str += "Template dimensions: " + templateDimBOXToString(templateDimBox) + "\n";
    // camera axis order
    str += "Camera axis order: ";
    for (const auto& [axis, values] : cameraAxisOrder) {
		str += axisToString(axis) + " -> ";
        for (const auto& value : values) {
			str += std::to_string(value) + " ";
		}
		str += ", ";
	}
    

	return str;
}