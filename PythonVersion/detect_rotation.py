import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from utils import read_camera_params, uv_to_xyz, xyz_to_uv, calculate_wall_depth, project_brick_size_to_image, \
    calculate_rotation_matrix_iterative, determine_side_axis, \
    adjust_rotation_to_align_axes


class DetectRotation:
    def __init__(self, brick_size_mm=(500, 2100, 1000), depth_scale=0.1, debug_mode=False):
        self.brick_size_mm = brick_size_mm
        self.depth_scale = depth_scale
        self.debug_mode = debug_mode

    def set_dump_path(self, dump_path):
        self.dump_path = dump_path

    def detect_rotation(self, image, seg_brick, depth_image, camera_params_json, wall_mask=None):
        # Read camera parameters
        camera_params = read_camera_params(camera_params_json)

        # Extract 3D point cloud data from the depth image

        # Calculate the average wall depth
        if wall_mask is not None:
            average_wall_depth = calculate_wall_depth(wall_mask, depth_image)
        else:
            average_wall_depth = np.mean(depth_image[depth_image > 0])
        #####################Debug########
        segmented_brick_2D = np.array(
            [[j, i] for i in range(seg_brick.shape[0]) for j in range(seg_brick.shape[1]) if seg_brick[i, j]])
        segmented_brick_3D = np.array([uv_to_xyz(point, depth_image[point[1], point[0]], camera_params) for point in
                                        segmented_brick_2D])

        # Calculate the standard deviation of the Z values
        std_depth = np.std(segmented_brick_3D[:, 2])

        # Define a dynamic threshold based on the standard deviation
        # Here, we use 2 standard deviations as an example, you can adjust this factor if needed
        threshold_factor = 1
        threshold = threshold_factor * std_depth
        mean_z = np.mean(segmented_brick_3D[:, 2])
        # Filter out points that are too far from the median depth based on the dynamic threshold
        segmented_brick_3D_Z_filter = segmented_brick_3D[np.abs(segmented_brick_3D[:, 2] - mean_z) < threshold]

        centroid = np.mean(segmented_brick_3D_Z_filter, axis=0)
        centered_brick_3D = segmented_brick_3D_Z_filter - centroid

        #######Debugggggg########

        # Apply PCA to find the principal axes
        pca = PCA(n_components=3)
        pca.fit(centered_brick_3D)
        pca_components = pca.components_

        # Calculate rotation matrix using PCA components
        rotation_matrix = calculate_rotation_matrix_iterative(pca_components)
        print(f"Rotation Matrix:\n{rotation_matrix}")
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        print(f"Euler Angles (degrees): {euler_angles}")
        print("centroid", centroid)
        # Rotate the point cloud using the rotation matrix
        rotated_point_cloud = np.dot(centered_brick_3D, rotation_matrix.T) + centroid

        # Determine the correct side axis based on brick size
        matched_dims = determine_side_axis(self.brick_size_mm, rotated_point_cloud)
        is_changes = False

        final_translation = np.mean(rotated_point_cloud, axis=0)

        #if self.debug_mode:
            # Calculate the final translation vector

           # final_translation, rotated_point_cloud = calculate_final_translation(final_translation, matched_dims,
              #                                                                   self.brick_size_mm,
               #                                                                  points=rotated_point_cloud)
        #else:
            #final_translation, same_points = calculate_final_translation(final_translation, matched_dims,
             #                                                                    self.brick_size_mm)
        if matched_dims:
            rotation_matrix, is_changes = adjust_rotation_to_align_axes(matched_dims, rotation_matrix)
            adjusted_euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
            print(f"Adjusted Euler Angles (degrees): {adjusted_euler_angles}")
        else:
            print("No changes needed in the rotation matrix or coordinates")
            adjusted_euler_angles = euler_angles


        # Dump debugging information if debug_mode is enabled
        if self.debug_mode:
            # Project the rotated 3D points back onto the 2D image plane
            projected_points_2D = np.array([xyz_to_uv(point, camera_params) for point in rotated_point_cloud])

            # Calculate initial translation in the original orientation
            inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
            initial_translation = np.dot(inverse_rotation_matrix, final_translation - centroid)
            brick_height_px_2D, brick_width_px_2D = project_brick_size_to_image(self.brick_size_mm, average_wall_depth,
                                                                                camera_params)
            point_cloud_template_3D = np.array(
                [uv_to_xyz([j, i], average_wall_depth, camera_params) for i in range(int(brick_height_px_2D)) for j in
                 range(int(brick_width_px_2D))])
            self.dump_debug_info(image,segmented_brick_2D, projected_points_2D, final_translation, centroid, camera_params,
                                 centroid, centered_brick_3D, pca_components, rotation_matrix,
                                 euler_angles, adjusted_euler_angles, rotated_point_cloud,
                                 point_cloud_template_3D, self.dump_path)
            res_path = os.path.join(self.dump_path, "result.txt")
            with open(res_path, "w") as f:
                f.write(f"Translation: {final_translation[0],final_translation[1],final_translation[2]}\n")
                f.write(f"Euler Angles: {adjusted_euler_angles[0],adjusted_euler_angles[1],adjusted_euler_angles[2]}\n")

        print(f"Final Translation (mm): {final_translation}")
        print(f"Final Rotation (Euler Angles in degrees): {adjusted_euler_angles}")

        return final_translation, adjusted_euler_angles

    def dump_debug_info(self, image, segmented_brick_2D, projected_points_2D, final_translation, after_rotation_translation,
                        camera_params,
                        centroid, centered_brick_3D, pca_components, rotation_matrix,
                        euler_angles, adjusted_euler_angles, rotated_point_cloud, point_cloud_template_3D, dump_path):
        # Normalize the PCA components
        basis_vectors = np.eye(3)
        pca_axis_indices = np.argmax(np.abs(np.dot(pca_components, basis_vectors.T)), axis=0)
        pca_x = pca_components[pca_axis_indices[0]]
        pca_y = pca_components[pca_axis_indices[1]]
        pca_z = pca_components[pca_axis_indices[2]]
        pca_x /= np.linalg.norm(pca_x)
        pca_y /= np.linalg.norm(pca_y)
        pca_z /= np.linalg.norm(pca_z)

        # Plot and save the 2D projection of the points
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # First subplot: Projected points with translation point
        ax1.imshow(image)
        ax1.scatter(*zip(*projected_points_2D), color='red', s=1, label='Projected Points')
        translationxy = xyz_to_uv(final_translation, camera_params)
        ax1.scatter(translationxy[0], translationxy[1], color='blue', s=50, label='Translation Point')
        ax1.set_title('Projected 3D Points on Image Plane')
        ax1.legend()

        # Second subplot: Original segmentation and center
        ax2.imshow(image)
        ax2.scatter(*zip(*segmented_brick_2D), color='green', s=1, label='Original Segmentation')
        original_center_2D = xyz_to_uv(centroid, camera_params)
        ax2.scatter(original_center_2D[0], original_center_2D[1], color='yellow', s=50, label='Segment Center')
        ax2.set_title('Original Segmentation and Center on Image Plane')
        ax2.legend()

        plt.savefig(os.path.join(dump_path, "projected_and_original_segmentation.png"))

        # Plot and save the PCA components and rotated point cloud
        fig = plt.figure(figsize=(20, 10))

        ax1 = fig.add_subplot(121, projection='3d')
        nonCentered_brick_3D = centered_brick_3D + centroid
        ax1.scatter(nonCentered_brick_3D[:, 0], nonCentered_brick_3D[:, 1], nonCentered_brick_3D[:, 2], color='blue', s=1,
                    label='3D Segmented Points')
        ax1.scatter(final_translation[0], final_translation[1], final_translation[2], color='red', s=50,
                    label='Translation Point')
        origin = np.zeros(3)
        scale = 1000
        start_plot_c = centroid
        ax1.quiver(start_plot_c[0], start_plot_c[1], start_plot_c[2], pca_x[0], pca_x[1], pca_x[2], color='r',
                   length=scale, label='PCA_X')
        ax1.quiver(start_plot_c[0], start_plot_c[1], start_plot_c[2], pca_y[0], pca_y[1], pca_y[2], color='g',
                   length=scale, label='PCA_Y')
        ax1.quiver(start_plot_c[0], start_plot_c[1], start_plot_c[2], pca_z[0], pca_z[1], pca_z[2], color='b',
                   length=scale, label='PCA_Z')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D PCA Components with Point Clo')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(rotated_point_cloud[:, 0], rotated_point_cloud[:, 1], rotated_point_cloud[:, 2], color='blue',
                    s=1, label='Rotated Point Cloud')

        ax2.scatter(after_rotation_translation[0], after_rotation_translation[1], after_rotation_translation[2], color='red', s=50, label='Translation Point')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        ax2.set_title('Rotated 3D Point Cloud')

        ax3 = fig.add_subplot(244, projection='3d')
        ax3.scatter(point_cloud_template_3D[:, 0], point_cloud_template_3D[:, 1], point_cloud_template_3D[:, 2],
                    color='blue', s=1, label='Template Point Cloud')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()

        plt.savefig(os.path.join(dump_path, "pca_and_rotated_point_cloud.png"))

        # Dump rotation matrix and Euler angles for debugging
        with open(os.path.join(dump_path, "rotation_matrix.txt"), "w") as f:
            f.write(f"Rotation Matrix:\n{rotation_matrix}\n")
        with open(os.path.join(dump_path, "euler_angles.txt"), "w") as f:
            f.write(f"Euler Angles (degrees): {euler_angles}\n")
        with open(os.path.join(dump_path, "adjusted_euler_angles.txt"), "w") as f:
            f.write(f"Adjusted Euler Angles (degrees): {adjusted_euler_angles}\n")