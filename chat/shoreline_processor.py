# import os
# import numpy as np
# import cv2
# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from skimage.morphology import skeletonize
# import rasterio
# import geopandas as gpd
# from shapely.geometry import LineString
# from model import build_unet

# #############################################
# # 1. Preprocessing and Inference Functions  #
# #############################################

# def preprocess_image(image_path, img_size=(512, 512)):
#     """
#     Reads a georeferenced TIFF image with rasterio to preserve the transform and CRS,
#     resizes it for model inference, and returns the image tensor along with the original image,
#     the affine transform, and CRS.
#     """
#     with rasterio.open(image_path) as src:
#         # Read the first three bands (assumed to be R, G, B)
#         image = src.read([1, 2, 3]).transpose(1, 2, 0)
#         transform = src.transform
#         crs = src.crs

#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     # Keep the original image (for mapping and overlay) at full resolution
#     original_image = image.copy()
#     # Resize the image for model inference
#     resized_image = cv2.resize(image, img_size)

#     transform_pipeline = transforms.Compose([
#         transforms.ToTensor(),  # Convert to tensor
#     ])
#     image_tensor = transform_pipeline(resized_image).unsqueeze(0)  # Add batch dimension

#     return image_tensor, original_image, transform, crs

# def skeletonize_shoreline(binary_mask):
#     """
#     Converts a binary mask (values 0 and 1) to a skeleton (centerline) using skimage's skeletonize.
#     Returns a skeleton image scaled to 0-255.
#     """
#     binary_mask = (binary_mask > 0).astype(np.uint8)
#     skeleton = skeletonize(binary_mask)
#     return (skeleton * 255).astype(np.uint8)

# def overlay_centerline(original_image, skeleton, line_color=(255, 0, 0), thickness=2):
#     """
#     Overlays the skeletonized centerline onto the original image.
#     Draws each pixel of the skeleton as a small circle.
#     """
#     overlay_image = original_image.copy()
#     coords = np.column_stack(np.where(skeleton > 0))
#     for coord in coords:
#         cv2.circle(overlay_image, (coord[1], coord[0]), radius=1, color=line_color, thickness=thickness)
#     return overlay_image

# #############################################
# # 2. Vectorization Function                 #
# #############################################

# def skeleton_to_vector(skeleton, transform, crs):
#     """
#     Converts the skeleton (binary image) into a vector polyline using the provided affine transform.
#     The pixel coordinates of the skeleton are converted to real‑world coordinates.
#     Returns a GeoDataFrame with the polyline and the correct CRS.
#     """
#     # Extract pixel coordinates where skeleton > 0 (row, col)
#     coords = np.column_stack(np.where(skeleton > 0))
#     if len(coords) < 2:
#         raise ValueError("Not enough points to form a polyline.")

#     # For a simple conversion, we sort by row then column
#     sorted_coords = coords[np.lexsort((coords[:,1], coords[:,0]))]
#     # Convert each pixel coordinate (col, row) to real-world coordinate using the transform
#     geo_coords = [transform * (int(col), int(row)) for row, col in sorted_coords]
#     line = LineString(geo_coords)
#     gdf = gpd.GeoDataFrame({"geometry": [line]}, crs=crs)
#     return gdf

# #############################################
# # 3. Visualization Function                 #
# #############################################

# def visualize_results(original_image, binary_mask, skeleton, overlay_image):
#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 4, 1)
#     plt.imshow(original_image)
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 4, 2)
#     plt.imshow(binary_mask, cmap="gray")
#     plt.title("Binary Mask")
#     plt.axis("off")

#     plt.subplot(1, 4, 3)
#     plt.imshow(skeleton, cmap="gray")
#     plt.title("Skeletonized Centerline")
#     plt.axis("off")

#     plt.subplot(1, 4, 4)
#     plt.imshow(overlay_image)
#     plt.title("Overlay on Original Image")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# #############################################
# # 4. Main Workflow                          #
# #############################################

# if __name__ == "__main__":
#     # Paths
#     model_path = "/content/drive/MyDrive/boundary_drive_file/final_model.pth"
#     image_path = "/content/drive/MyDrive/boundary_drive_file/trails/r2023.tif"
#     output_shapefile = "/content/drive/MyDrive/boundary_drive_file/trails/centerline.shp"

#     # Model parameters
#     img_size = (512, 512)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load your U-Net model (make sure build_unet() is defined)
#     model = build_unet()  # Replace with your U-Net model definition
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # Preprocess the input image (using rasterio to retain georeferencing info)
#     image_tensor, original_image, transform, crs = preprocess_image(image_path, img_size)
#     image_tensor = image_tensor.to(device)

#     # Perform inference with the model
#     with torch.no_grad():
#         pred_mask = model(image_tensor)
#         pred_mask = torch.sigmoid(pred_mask).squeeze(0).cpu().numpy()
#         if pred_mask.ndim == 3:
#             pred_mask = pred_mask[0]

#     # Convert prediction to binary mask
#     binary_mask = (pred_mask > 0.5).astype(np.uint8)

#     # Resize the binary mask to the original image dimensions (if needed)
#     binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Skeletonize the shoreline mask
#     skeleton = skeletonize_shoreline(binary_mask_resized)

#     # Overlay the skeleton on the original image
#     overlay_image = overlay_centerline(original_image, skeleton)

#     # Convert the skeleton to a vector polyline using the georeferencing information
#     centerline_gdf = skeleton_to_vector(skeleton, transform, crs)

#     # Save the vector polyline as a Shapefile
#     centerline_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
#     print(f"Centerline saved to {output_shapefile}")

#     # Print the total length of the centerline (in the units of the CRS)
#     total_length = centerline_gdf.length.sum()
#     print(f"Total Length of Centerline: {total_length:.2f} meters")

#     # Visualize results
#     visualize_results(original_image, binary_mask_resized, skeleton, overlay_image)
