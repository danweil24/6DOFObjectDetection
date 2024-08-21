
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
