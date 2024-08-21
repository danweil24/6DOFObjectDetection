import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_camera_params(camera_params_json):
    fx = camera_params_json['fx']
    fy = camera_params_json['fy']
    cx = camera_params_json['px']
    cy = camera_params_json['py']
    return {'fx': float(fx), 'fy': float(fy), 'px': float(cx), 'py': float(cy)}


def uv_to_xyz(uv, depth, camera_params):
    fx, fy, cx, cy = camera_params['fx'], camera_params['fy'], camera_params['px'], camera_params['py']
    x = (uv[0] - cx) * depth / fx
    y = (uv[1] - cy) * depth / fy
    z = depth
    return np.array([x, y, z])


def xyz_to_uv(xyz, camera_params):
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['px']
    cy = camera_params['py']
    u = (fx * xyz[0] / xyz[2]) + cx
    v = (fy * xyz[1] / xyz[2]) + cy
    return np.array([u, v])


def calculate_wall_depth(wall_mask, depth_image):
    return np.mean(depth_image[wall_mask > 0])


def project_brick_size_to_image(brick_size_mm, depth, camera_params):
    brick_height_mm, brick_width_mm, side_size = brick_size_mm
    fx = camera_params['fx']
    fy = camera_params['fy']
    brick_height_px = (brick_height_mm * fx) / depth
    brick_width_px = (brick_width_mm * fy) / depth
    return brick_height_px, brick_width_px


def calculate_brick_dimensions_in_camera_space(brick_size_mm, average_wall_depth, camera_params):
    fx = camera_params['fx']
    fy = camera_params['fy']
    brick_height_m = brick_size_mm[0]
    brick_width_m = brick_size_mm[1]
    brick_depth_m = brick_size_mm[2]
    brick_height_px = (brick_height_m * fx) / average_wall_depth
    brick_width_px = (brick_width_m * fy) / average_wall_depth
    brick_depth_px = (brick_depth_m * fy) / average_wall_depth
    return brick_height_px, brick_width_px, brick_depth_px


def calculate_rotation_matrix_iterative(pca_components):
    def align_to_axis(main_axis, target_axis):
        rotation_axis = np.cross(main_axis, target_axis)
        angle = np.arccos(np.dot(main_axis, target_axis))

        if np.linalg.norm(rotation_axis) != 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        I = np.eye(3)
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    rotation_matrix_x = align_to_axis(pca_components[0], np.array([1, 0, 0]))
    pca_components = np.dot(rotation_matrix_x, pca_components.T).T
    rotation_matrix_y = align_to_axis(pca_components[1], np.array([0, 1, 0]))
    rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_x)
    return rotation_matrix


def determine_side_axis(template_dims_3D, rotated_brick_3D, tolerance=0.8):
    min_coords = np.min(rotated_brick_3D, axis=0)
    max_coords = np.max(rotated_brick_3D, axis=0)
    lengths = max_coords - min_coords
    height_t, width_t, depth_t = template_dims_3D

    if lengths[0] == 0 or lengths[1] == 0 or height_t == 0 or width_t == 0 or depth_t == 0:
        print(
            f"Zero dimension detected: Brick Lengths: X: {lengths[0]}, Y: {lengths[1]}, Template Lengths: Width: {width_t}, Height: {height_t}, Depth: {depth_t}")
        return False, None

    def is_within_tolerance(length, template_dim):
        ar = length / template_dim
        if ar > 1:
            ar = 1 / ar
        return ar >= tolerance

    if is_within_tolerance(lengths[0], width_t) and is_within_tolerance(lengths[1], height_t):
        matched_dims = {'X': 'Width', 'Y': 'Height'}
    elif is_within_tolerance(lengths[0], height_t) and is_within_tolerance(lengths[1], width_t):
        matched_dims = {'X': 'Height', 'Y': 'Width'}
    elif is_within_tolerance(lengths[0], depth_t) and is_within_tolerance(lengths[1], width_t):
        matched_dims = {'X': 'Depth', 'Y': 'Width'}
    elif is_within_tolerance(lengths[0], width_t) and is_within_tolerance(lengths[1], depth_t):
        matched_dims = {'X': 'Width', 'Y': 'Depth'}
    elif is_within_tolerance(lengths[0], height_t) and is_within_tolerance(lengths[1], depth_t):
        matched_dims = {'X': 'Height', 'Y': 'Depth'}
    elif is_within_tolerance(lengths[0], depth_t) and is_within_tolerance(lengths[1], height_t):
        matched_dims = {'X': 'Depth', 'Y': 'Height'}
    else:
        print(
            f"Axes need rotation: Brick Lengths: X: {lengths[0]}, Y: {lengths[1]}, Template Lengths: Width: {width_t}, Height: {height_t}, Depth: {depth_t}")
        return None

    print(
        f"Axes are ok: Brick Length X: {lengths[0]} ({matched_dims['X']}), Y: {lengths[1]} ({matched_dims['Y']}), Template Lengths: Width: {width_t}, Height: {height_t}, Depth: {depth_t}")
    return matched_dims


def adjust_rotation_to_align_axes(matched_dims, rotation_matrix):
    is_changed = False
    target_orientations = {
        'Width': np.array([1, 0, 0]),
        'Height': np.array([0, 1, 0]),
        'Depth': np.array([0, 0, 1])
    }

    current_orientations = {
        'X': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'Z': np.array([0, 0, 1])
    }

    rotation_align = np.eye(3)
    for axis, orientation in matched_dims.items():
        current_axis = current_orientations[axis]
        target_axis = target_orientations[orientation]

        if not np.allclose(current_axis, target_axis):
            is_changed = True
            print(current_axis)
            v = np.cross(current_axis, target_axis)
            c = np.dot(current_axis, target_axis)
            s = np.linalg.norm(v)

            if s != 0:
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
                rotation_align = rotation @ rotation_align

    adjusted_rotation_matrix = rotation_align @ rotation_matrix
    return adjusted_rotation_matrix,is_changed


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def calculate_crop_for_network(image_size, network_input_size, detected_brick__point):
    center = np.array([image_size[1] // 2, image_size[0] // 2])
    width = min(network_input_size, image_size[1])
    height = min(network_input_size, image_size[0])
    start_x = max(0, center[0] - width // 2)
    start_y = max(0, center[1] - height // 2)
    if detected_brick__point[0] < start_x or detected_brick__point[0] > start_x + width or detected_brick__point[
        1] < start_y or detected_brick__point[1] > start_y + height:
        start_x = max(0, detected_brick__point[0] - width // 2)
        start_y = max(0, detected_brick__point[1] - height // 2)
    return start_x, start_y, width, height


def get_wall_mask(rgb_image, depth_image, dump_path=None, depth_threshold_low=250,
                  depth_threshold_high=400, depth_scale=0.1):
    def apply_colormap(depth_image):
        depth_normalized = cv2.normalize(depth_image / depth_scale, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_normalized), cv2.COLORMAP_JET)
        return depth_colormap

    wall_mask = cv2.inRange(depth_image, depth_threshold_low /depth_scale, depth_threshold_high /depth_scale)
    wall_depth_mm = np.mean(depth_image[wall_mask > 0]) * depth_scale

    mid_y, mid_x = wall_mask.shape[0] // 2, wall_mask.shape[1] // 2
    non_zero_points = np.argwhere(wall_mask > 0)
    center_point = None

    if len(non_zero_points) > 0:
        distances = np.sqrt((non_zero_points[:, 0] - mid_y) ** 2 + (non_zero_points[:, 1] - mid_x) ** 2)
        min_index = np.argmin(distances)
        center_point = non_zero_points[min_index]
        center_point[0] += 50

    if dump_path:
        depth_colormap = apply_colormap(depth_image)
        os.makedirs(dump_path, exist_ok=True)
        wall_region = cv2.bitwise_and(rgb_image, rgb_image, mask=wall_mask)
        depth_colormap_path = os.path.join(dump_path, "depth_colormap.png")
        cv2.imwrite(depth_colormap_path, depth_colormap)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(depth_colormap)
        plt.title('Depth Image')

        plt.subplot(1, 2, 2)
        plt.imshow(wall_region, cmap='gray')
        if center_point is not None:
            plt.plot(center_point[1], center_point[0], 'ro')
        plt.title('Wall Region with Center Point')

        wall_region_plot_path = os.path.join(dump_path, "wall_region_with_center_point.png")
        plt.savefig(wall_region_plot_path)
        plt.close()

    return wall_mask, center_point
