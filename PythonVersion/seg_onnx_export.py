import cv2
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import warnings
import onnx
from onnxsim import simplify
from torch.nn import functional as F
import numpy as np
from brick_detector_segment import show_mask

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


def export_image_encoder(sam, output_path, opset):
    print("Exporting image encoder...")
    image_encoder = sam.image_encoder

    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float)  # Adjusted for encoder's expected input size
    output = image_encoder(dummy_input)
    with open(output_path, "wb") as f:
        torch.onnx.export(
            image_encoder,
            dummy_input,
            f,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["image_embeddings"]
        )
    print(f"Image encoder exported to {output_path}")


def export_full_model(sam, output_path, opset, return_single_mask, gelu_approximate, use_stability_score,
                      return_extra_metrics):
    print("Exporting full model...")
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 1, 2), dtype=torch.float),  # One point
        "point_labels": torch.randint(low=0, high=4, size=(1, 1), dtype=torch.float),  # One class
        "orig_im_size": torch.tensor([480, 848], dtype=torch.float),  # Nearest crop with the aspect ratio 3:2
    }

    with open(output_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks"]
        )
    print(f"Full model exported to {output_path}")


def simplify_onnx_model(model_path, output_path):
    print("Simplifying the ONNX model...")
    onnx_model = onnx.load(model_path)
    model_simp, check = simplify(onnx_model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, output_path)
    print(f"Simplified model saved to {output_path}")


def run_export(model_type, checkpoint, output, opset, return_single_mask, gelu_approximate, use_stability_score,
               return_extra_metrics):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    encoder_output = output.replace(".onnx", "_encoder.onnx")
    decoder_output = output.replace(".onnx", "_decoder.onnx")

    export_image_encoder(sam, encoder_output, opset)
    export_full_model(sam, decoder_output, opset, return_single_mask, gelu_approximate, use_stability_score,
                      return_extra_metrics)

    simplify_onnx_model(encoder_output, encoder_output.replace(".onnx", "_simplified.onnx"))
    simplify_onnx_model(decoder_output, decoder_output.replace(".onnx", "_simplified.onnx"))


def to_numpy(tensor):
    return tensor.cpu().numpy()



def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def norm_pad(x):
    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])

    # Normalize each channel
    x = (x - pixel_mean) / pixel_std

    h, w = x.shape[:2]
    padh = 1024 - h
    padw = 1024 - w

    # Pad the image to 1024x1024
    x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant')

    return x


def preprocess_for_onnx(input_rgb):
    newh, neww = get_preprocess_shape(input_rgb.shape[0], input_rgb.shape[1], 1024)
    input_rgb = cv2.resize(input_rgb, (neww, newh), interpolation=cv2.INTER_LINEAR)
    input_rgb = norm_pad(input_rgb)
    input_rgb = input_rgb.transpose(2, 0, 1)  # Change to CHW format
    return input_rgb, newh, neww

def inference_input_onnx(inputs, encoder_model, decoder_model):
    point = inputs["point"]
    point_labels = inputs["point_labels"]
    image = inputs["image"]
    oldh, oldw = (480, 848)
    image, newh, neww = preprocess_for_onnx(image)
    image = image.astype(np.float32)

    encoder_session = onnxruntime.InferenceSession(encoder_model)
    image_embeddings = encoder_session.run(None, {"input_image": image[np.newaxis, :]})[0]
    # change point to the new shape

    point = point * [neww / oldw, newh / oldh]
    decoder_session = onnxruntime.InferenceSession(decoder_model)
    masks = decoder_session.run(None, {
        "image_embeddings": image_embeddings,
        "point_coords": point[np.newaxis, :, :].astype(np.float32),
        "point_labels": point_labels[np.newaxis, :].astype(np.float32),
        "orig_im_size": np.array([480, input_size], dtype=np.float32)
    })

    return masks,image_embeddings


def inference_input_normal(inputs, model):
    predictor = SamPredictor(model)
    image = inputs["image"]
    predictor.set_image(image)
    point = inputs["point"]
    point_labels = inputs["point_labels"]
    masks, scores, logits = predictor.predict(
        point_coords=point,
        point_labels=point_labels,
        multimask_output=False,
    )
    return masks, scores, logits,predictor.get_image_embedding()


def run_onnx(encoder_model, decoder_model, image, point):
    """
    Run inference using ONNX models for the given image and point.

    Args:
        encoder_model (str): Path to the encoder ONNX model.
        decoder_model (str): Path to the decoder ONNX model.
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
    oldh, oldw = image.shape[:2]
    image, newh, neww = preprocess_for_onnx(image)
    image = image.astype(np.float32)

    # Load the encoder model
    encoder_session = onnxruntime.InferenceSession(encoder_model)
    image_embeddings = encoder_session.run(None, {"input_image": image[np.newaxis, :]})[0]

    # Adjust the point to the new shape
    point = point * [neww / oldw, newh / oldh]

    # Load the decoder model
    decoder_session = onnxruntime.InferenceSession(decoder_model)
    masks_onnx = decoder_session.run(None, {
        "image_embeddings": image_embeddings,
        "point_coords": point[np.newaxis, :, :].astype(np.float32),
        "point_labels": point_labels[np.newaxis, :].astype(np.float32),
        "orig_im_size": np.array([oldh, oldw], dtype=np.float32)
    })

    masks = masks_onnx[0][0]
    scores = masks_onnx[1][0]

    return masks, scores


def test_onnx_model():
    # Dummy input
    image = cv2.imread(r"C:\development\Repository\PoseEstimation6DOF\inputs\0\color.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    copy_=image.copy()
    point = np.array([[70, 70]])
    point_labels = np.array([1])

    inputs = {
        "image": image,
        "point": point,
        "point_labels": point_labels
    }
    input_full = {
        "image": copy_,
        "point": point,
        "point_labels": point_labels
    }

    encoder_model = "sam_vit_b_01ec64_encoder_simplified.onnx"
    decoder_model = "sam_vit_b_01ec64_decoder_simplified.onnx"
    masks, scores, logits,image_emb= inference_input_normal(input_full, sam_model_registry["vit_b"]("sam_vit_b_01ec64.pth"))
    masks_onnx,onnx_image_emb = inference_input_onnx(inputs, encoder_model, decoder_model)
    normal_embedding_numpy = to_numpy(image_emb)
    diff = np.abs(normal_embedding_numpy - onnx_image_emb)
    print("Difference between embeddings mean and max: ", diff.mean(), diff.max())
    mask_onnx = masks_onnx[0][0]
    scores_onnx = masks_onnx[1][0]
    # if > 0 True else False
    mask_onnx = mask_onnx > 0
    # plot the masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask_onnx, plt.gca())
    plt.title(f"ONNX Inference Score: {np.max(scores_onnx):.3f}", fontsize=18)
    # Save the output image
    plt.show()
    # fig the normal model
    plt.figure(figsize=(10, 10))
    plt.imshow(copy_)
    show_mask(masks, plt.gca())
    plt.title(f"Normal Inference Score: {np.max(scores):.3f}", fontsize=18)
    plt.show()
    print("ONNX Inference output: ", masks_onnx)


if __name__ == '__main__':
    model_path = "sam_vit_b_01ec64.pth"
    type_m = "vit_b"
    input_size = 704
    return_single_mask = True
    opset = 17
    output = f"models\sam_vit_b_01ec64_{input_size}.onnx"
    run_export(model_type=type_m, checkpoint=model_path, output=output, opset=opset,
               return_single_mask=return_single_mask, gelu_approximate=False, use_stability_score=False,
               return_extra_metrics=False)
    test_onnx_model()
