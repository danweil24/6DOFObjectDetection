@echo off
set modelPath=C:\\development\\Repository\\Brick3DPose\\models
set basePath=C:\\development\\Repository\\Brick3DPose\\inputs
set verbosity=LOW
set multi=true

set jsonPayload={"\"modelsPath\":\"%modelPath%\",\"basePath\":\"%basePath%\",\"verbosity\":\"%verbosity%\",\"multi\":%multi%"}

curl -X POST -H "Content-Type: application/json" -d %jsonPayload% http://localhost:8080/process
pause