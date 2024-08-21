import time
import os
import json
from brick_seg_detect_2d import BrickSegDetect2D
from detect_rotation import DetectRotation

if __name__ == '__main__':
    img_dir = "inputs"
    output_dir = "outputs"
    current_output_dir = time.strftime("%Y%m%d_640")
    output_dir = os.path.join(output_dir, current_output_dir)
    brick_size_mm = (500, 2100, 1000)
    depth_scale = 0.1
    pathes = [os.path.join(img_dir, _dir) for _dir in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, _dir))]
    input_size = 640
    skip_network_exe = False

    onnx_encoder_path = f"models/sam_vit_b_01ec64_{input_size}_encoder_simplified.onnx"
    onnx_decoder_path = f"models/sam_vit_b_01ec64_{input_size}_decoder_simplified.onnx"

    brick_seg_detector = BrickSegDetect2D(onnx_encoder_path, onnx_decoder_path, input_size)
    rotation_detector = DetectRotation(brick_size_mm, depth_scale, debug_mode=True)

    total_encoder_time = 0
    total_decoder_time = 0
    num_images = 0

    for p in pathes:
        if p == "inputs\\10" or True:
            current_output_dir = os.path.join(output_dir, os.path.basename(p))
            os.makedirs(current_output_dir, exist_ok=True)

            image, full_mask, depth_image, wall_mask, center_point_brick = brick_seg_detector.detect_by_segment(
                os.path.join(p, "color.png"),
                os.path.join(p, "depth.png"),
                os.path.join(p, "cam.json"),
                dump_path=current_output_dir,
                brick_size_mm=brick_size_mm,
                depth_scale=depth_scale
            )
            rotation_detector.set_dump_path(current_output_dir)
            translation, euler_angles = rotation_detector.detect_rotation(
                image,
                full_mask,
                depth_image,
                json.load(open(os.path.join(p, "cam.json"))),
                wall_mask=wall_mask,

            )

