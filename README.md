# 6D Brick Pose Detection from RGBD Images

## Overview

This project aims to detect the 6D pose of a brick from RGBD images using a combination of 2D segmentation and 3D pose estimation. The approach focuses on creating a baseline solution that runs efficiently and provides a modular framework for ongoing algorithm improvements.

## Pipeline

### 1. Object Segmentation (2D)

- **Objective:** Detect the pixels in the 2D image where the brick lies.
- **Approach:** Initially, classic vision filters and 2D CNNs were tested, but Meta’s open-source Segment Anything model (v2) was used for better results. The model was modified for faster performance and converted to ONNX format.

![2D Segmentation](path/to/Fig1_image.png)  
*Figure 1: Detecting point on the center brick by heuristics.*

### 2. Pose Estimation

- **Objective:** Detect the rotation and translation of the segmented data using camera world coordinates.
- **Approach:** The 3D point cloud is generated using camera intrinsics and depth information. PCA is performed to align the point cloud to the principal axes, and the final rotation is determined by aligning these axes to the desired world coordinates.

![Pose Estimation](path/to/Fig2_image.png)  
*Figure 2: PCA on point cloud before and after rotation to camera axis.*

## Code and Implementation

- **Language:** C++ with Python prototyping
- **Dependencies:** OpenCV, ONNX Runtime, Eigen, Crow (HTTP)

For detailed steps on running the project, please refer to the instructions below.

## Environment Setup

### 1. Install Dependencies with vcpkg

```bash
# Clone the repository
git clone https://github.com/your-repo/6D-Brick-Pose-Detection.git

# Navigate to the project directory
cd 6D-Brick-Pose-Detection

# Install vcpkg if not already installed
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install

# Install dependencies listed in vcpkg.json
./vcpkg install --triplet x64-windowss
```
## Future Improvements

- **Segmentation:** Training a UNet model specifically for wall data can improve runtime and accuracy.
- **Pose Estimation:** Handling camera distortion and improving the algorithm for partial visibility of the brick can enhance robustness.



# BrickDetection HTTP Server

## Overview

The BrickDetection HTTP server processes RGB-D images to detect the pose of bricks. This guide provides instructions on how to use the HTTP server.

## Starting the Server

1. **Navigate to BrickDetection\x64\Debug

2. **Run the Server**:
   Start the server by running the executable:
   ```sh
   BrickDetection.exe
   ```

3. **Server Running**:
   The server will start and listen on port `8080` by default. You will see a message indicating that the server is running.

### Sending Requests

1. **Endpoint**:
   Send HTTP POST requests to the endpoint:
   ```
   http://localhost:8080/process
   ```

2. **Request Payload**:
   The server accepts JSON payloads with the following fields:

   - `imagePath`: The path to the RGB image file.
   - `depthPath`: The path to the depth image file.
   - `camParamsPath`: The path to the camera parameters file.
   - `verbosity` (optional): The verbosity level of the logger. Possible values: `NONE`, `LOW`, `MEDIUM`, `HIGH`, `ERR`.
   - `basePath` (optional): A base path containing `color.png`, `depth.png`, and `cam.json`. If provided, this overrides individual paths.

3. **Example Requests**:

   - **With Individual Paths**:
     ```sh
     curl -X POST -H "Content-Type: application/json" -d "{\"imagePath\": \"C:\\path\\to\\color.png\", \"depthPath\": \"C:\\path\\to\\depth.png\", \"camParamsPath\": \"C:\\path\\to\\cam.json\", \"verbosity\": \"HIGH\"}" http://localhost:8080/process
     ```

   - **With Base Path**:
     ```sh
     curl -X POST -H "Content-Type: application/json" -d "{\"basePath\": \"C:\\path\\to\\base\\\\\", \"verbosity\": \"LOW\"}" http://localhost:8080/process
     ```

4. **Response**:
   The server will respond with the detected brick pose in JSON format. An example response might look like this:
   ```json
   {
       "position": [x, y, z],
       "orientation": [roll, pitch, yaw]
   }
   ```

### Configuration

- **Port Configuration**:
  By default, the server runs on port `8080`. To change the port, modify the source code in the `main()` function and recompile the application.

- **Verbosity Levels**:
  - `NONE`: No debug prints.
  - `LOW`: Low verbosity debug prints.
  - `MEDIUM`: Medium verbosity debug prints.
  - `HIGH`: High verbosity debug prints.
  - `ERR`: Error prints.

