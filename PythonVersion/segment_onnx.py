import os

import cv2
import numpy as np
import onnxruntime
import time
INPUT_TRAN_SIZE = 640


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def norm_pad(x,input_size):
    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])

    # Normalize each channel
    x = (x - pixel_mean) / pixel_std

    h, w = x.shape[:2]
    padh = input_size - h
    padw = input_size - w

    # Pad the image to 1024x1024
    x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant')

    return x


def preprocess_for_onnx(input_rgb,input_size):
    newh, neww = get_preprocess_shape(input_rgb.shape[0], input_rgb.shape[1], input_size)
    # debug
    input_rgb = cv2.resize(input_rgb, (neww, newh), interpolation=cv2.INTER_LINEAR)
    input_rgb = norm_pad(input_rgb,input_size)
    input_rgb = input_rgb.transpose(2, 0, 1)  # Change to CHW format
    return input_rgb, newh, neww

def run_onnx(encoder_session, decoder_session, image, point,input_size=640, dump_path=None):
    """
    Run inference using ONNX models for the given image and point.

    Args:
        encoder_session (onnxruntime.InferenceSession): ONNX runtime session for the encoder model.
        decoder_session (onnxruntime.InferenceSession): ONNX runtime session for the decoder model.
        image (np.array): Input image.
        point (np.array): Point coordinates.

    Returns:
        np.array: Mask.
        np.array: Scores.
    """
    point_labels = np.array([1])
    inputs = {
        "image": image,
        "point": point,
        "point_labels": point_labels
    }

    # Preprocess the image
    t = time.time()
    oldh, oldw = image.shape[:2]
    image, newh, neww = preprocess_for_onnx(image,input_size)
    image = image.astype(np.float32)
    time_preprocess = time.time() - t
    time_preprocess_ms = time_preprocess * 1000
    # Run the encoder model
    time_encoder = time.time()
    image_embeddings = encoder_session.run(None, {"input_image": image[np.newaxis, :]})[0]
    time_encoder = time.time() - time_encoder
    time_encoder_ms = time_encoder * 1000
    # Adjust the point to the new shape
    point = point * [neww / oldw, newh / oldh]
    # make 1,1,2
    point = point[np.newaxis, :]
    point = point[np.newaxis, :]
    time_decoder = time.time()
    # Run the decoder model
    masks_onnx = decoder_session.run(None, {
        "image_embeddings": image_embeddings,
        "point_coords": point.astype(np.float32),
        "point_labels": point_labels[np.newaxis, :].astype(np.float32),
        "orig_im_size": np.array([oldh, oldw], dtype=np.float32)
    })
    time_decoder = time.time() - time_decoder
    masks = masks_onnx[0][0]
    scores = masks_onnx[1][0]
    #if > 0 True else False


    # resize the mask to the original size
    #masks = masks > 0
    print(f"encoder time: {time_encoder_ms:.3f}ms, decoder time: {time_decoder * 1000:.3f}ms")

    return masks, scores, time_preprocess_ms, time_encoder_ms, time_decoder * 1000