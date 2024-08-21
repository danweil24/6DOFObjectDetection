import onnxruntime
import time
import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

from segment_onnx import run_onnx
from utils import get_wall_mask, show_mask, show_points, calculate_crop_for_network

class BrickSegDetect2D:
    def __init__(self, onnx_encoder_path=None, onnx_decoder_path=None, input_size=640):
        self.onnx_encoder = None
        self.onnx_decoder = None
        self.input_size = input_size

        if onnx_encoder_path and onnx_decoder_path:
            self.onnx_encoder = onnxruntime.InferenceSession(onnx_encoder_path)
            self.onnx_decoder = onnxruntime.InferenceSession(onnx_decoder_path)

    def filter_full_mask(self, full_mask, input_point):
        if not (0 <= input_point[0] < full_mask.shape[1] and 0 <= input_point[1] < full_mask.shape[0]):
            raise ValueError("Input point is outside the mask boundaries")
        num_labels, labels_im = cv2.connectedComponents(full_mask.astype(np.uint8))
        input_label = labels_im[input_point[1], input_point[0]]
        filtered_mask = (labels_im == input_label).astype(np.uint8)
        return filtered_mask

    def detect_by_segment(self, color_path, depth_path, cam_path, dump_path=None, brick_size_mm=(500, 2100, 1000), depth_scale=0.1):
        image = cv2.imread(color_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_image = self.read_depth_image(depth_path)

        wall_mask, center_point_brick = get_wall_mask(image, depth_image, dump_path=dump_path)
        if center_point_brick is not None:
            center_point_brick = center_point_brick[::-1]
        print(f"Center Point Brick: {center_point_brick}")
        input_point = np.array([center_point_brick])
        input_label = np.array([1])
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.onnx_encoder and self.onnx_decoder:
            crop_x, crop_y, crop_w, crop_h = calculate_crop_for_network(image.shape, self.input_size, center_point_brick)
            center_cropped_image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            brick_point_centered = center_point_brick - np.array([crop_x, crop_y])
            max_mask, max_score, prepocess_time, encoder_time, decoder_time = run_onnx(self.onnx_encoder, self.onnx_decoder, center_cropped_image, np.array(brick_point_centered), input_size=self.input_size,dump_path = dump_path)
            # dump the mask in float
            max_mask = max_mask > 0
            full_mask = full_mask > 0
            full_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = max_mask
            filter_t = time.time()
            full_mask = self.filter_full_mask(full_mask, center_point_brick)
            filter_t = time.time() - filter_t
            print(f"Filter Time: {filter_t:.3f}ms")
            if dump_path:
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(full_mask, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Model Input size {self.input_size} Filtered: {True}\nEncoder Time: {encoder_time:.3f}ms, Decoder Time: {decoder_time:.3f}ms", fontsize=18)
                plt.axis('off')
                plt.savefig(os.path.join(dump_path, f"seg_brick_{self.input_size}.png"))
                plt.close()
                np.save(os.path.join(dump_path, f"seg_brick_{self.input_size}.npy"), full_mask)
                #
                full_mask.tofile(os.path.join(dump_path, 'seg.raw'))
        else:
            print(f"Could not find the brick point, checking if seg_brick_{self.input_size}.npy exists")
            if os.path.join(dump_path, f"seg_brick_{self.input_size}.npy"):
                path_to_dump_seg = os.path.join(dump_path, f"seg_brick_{self.input_size}.npy")
                full_mask = np.load(path_to_dump_seg)
                print(f"Loaded {path_to_dump_seg}")

        return image, full_mask, depth_image, wall_mask, center_point_brick

    def read_depth_image(self, depth_image_path, depth_scale=0.1):
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        return depth_image
