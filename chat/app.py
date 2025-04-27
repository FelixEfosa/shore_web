# import os
# from flask import Flask, request, render_template, redirect, url_for, send_from_directory
# from werkzeug.utils import secure_filename

# import torch
# import torch.nn as nn
# from torchvision import transforms
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# import rasterio
# import geopandas as gpd
# from shapely.geometry import LineString
# import leafmap.foliumap as leafmap  # using Leafmap with the folium backend

# # ----------------------------
# # Custom UNet Model Definition
# # ----------------------------
# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_c)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU()
#     def forward(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         return x

# class encoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = conv_block(in_c, out_c)
#         self.pool = nn.MaxPool2d(2)
#     def forward(self, inputs):
#         x = self.conv(inputs)
#         p = self.pool(x)
#         return x, p

# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
#         self.conv = conv_block(out_c + out_c, out_c)
#     def forward(self, inputs, skip):
#         x = self.up(inputs)
#         # Crop skip to match x spatial dimensions if needed
#         if x.size()[2:] != skip.size()[2:]:
#             diffY = skip.size(2) - x.size(2)
#             diffX = skip.size(3) - x.size(3)
#             skip = skip[:, :, diffY // 2: skip.size(2) - diffY // 2, diffX // 2: skip.size(3) - diffX // 2]
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         return x

# class build_unet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder
#         self.e1 = encoder_block(3, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)
#         self.e4 = encoder_block(256, 512)
#         # Bottleneck
#         self.b = conv_block(512, 1024)
#         # Decoder
#         self.d1 = decoder_block(1024, 512)
#         self.d2 = decoder_block(512, 256)
#         self.d3 = decoder_block(256, 128)
#         self.d4 = decoder_block(128, 64)
#         # Classifier
#         self.outputs = nn.Conv2d(64, 1, kernel_size=1)
#     def forward(self, inputs):
#         s1, p1 = self.e1(inputs)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)
#         b = self.b(p4)
#         d1 = self.d1(b, s4)
#         d2 = self.d2(d1, s3)
#         d3 = self.d3(d2, s2)
#         d4 = self.d4(d3, s1)
#         outputs = self.outputs(d4)
#         return outputs

# # ----------------------------
# # Helper Functions for Image Processing & Georeferencing
# # ----------------------------
# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {"tif", "tiff", "png", "jpg", "jpeg"}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image_cv(image_path):
#     """Preprocess image using OpenCV while keeping original dimensions."""
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")
#     original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     transform_pipeline = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform_pipeline(original_image).unsqueeze(0)
#     return image_tensor, original_image

# def skeletonize_shoreline(binary_mask):
#     """Skeletonize the binary mask to obtain the centerline."""
#     binary_mask = (binary_mask > 0).astype(np.uint8)
#     skeleton = skeletonize(binary_mask)
#     return (skeleton * 255).astype(np.uint8)

# def overlay_centerline(original_image, skeleton, line_color=(255, 0, 0), thickness=2):
#     """Overlay the skeleton (centerline) on the original image."""
#     overlay = original_image.copy()
#     coords = np.column_stack(np.where(skeleton > 0))
#     for coord in coords:
#         cv2.circle(overlay, (coord[1], coord[0]), radius=1, color=line_color, thickness=thickness)
#     return overlay

# def save_centerline_to_shapefile(skeleton, image_path, output_filename):
#     """
#     Converts the skeleton (in pixel coordinates) to geographic coordinates using the
#     georeferencing information from the input image, and saves it as a shapefile.
#     """
#     with rasterio.open(image_path) as src:
#         transform = src.transform
#         crs = src.crs
#     coords = np.column_stack(np.where(skeleton > 0))
#     if len(coords) < 2:
#         raise ValueError("No valid centerline detected.")
#     # Convert (row, col) pixel coordinates to (x, y) geographic coordinates.
#     real_coords = [transform * (int(x), int(y)) for y, x in coords]
#     centerline_geom = LineString(real_coords)
#     gdf = gpd.GeoDataFrame({"geometry": [centerline_geom]}, crs=crs)
#     output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
#     gdf.to_file(output_path, driver="ESRI Shapefile")
#     print(f"Centerline saved as {output_path}")
#     return output_path, gdf

# def create_leafmap(vector_shapefile):
#     """Create an interactive Leafmap map that displays the vector shapefile."""
#     gdf = gpd.read_file(vector_shapefile)
#     if gdf.empty:
#         raise ValueError("Shapefile is empty.")
#     gdf = gdf.to_crs(epsg=4326)
#     center_lat = gdf.geometry.centroid.y.mean()
#     center_lon = gdf.geometry.centroid.x.mean()
#     m = leafmap.Map(center=[center_lat, center_lon], zoom=15, basemap="Google Satellite")
#     m.add_gdf(gdf, layer_name="Shoreline Centerline", style={"color": "red", "weight": 2})
#     return m

# # ----------------------------
# # Flask App Setup
# # ----------------------------
# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "uploads"
# app.config["OUTPUT_FOLDER"] = "outputs"
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# # ----------------------------
# # Load the Model
# # ----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Change the model_path to your local path if needed.
# model_path = "C:/Users/ThinkPad/Desktop/GeoAI/New_Boundary_plotter/final_model.pth"
# model = build_unet()  # Your custom UNet model
# try:
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
# except TypeError:
#     model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()

# # ----------------------------
# # Flask Routes
# # ----------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return "No file part", 400
#         file = request.files["file"]
#         if file.filename == "":
#             return "No selected file", 400
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(filepath)
#             return redirect(url_for("process_image_route", filename=filename))
#     return render_template("index.html")

# @app.route("/process/<filename>")
# def process_image_route(filename):
#     filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     # Preprocess the image (keeping original dimensions)
#     image_tensor, original_image = preprocess_image_cv(filepath)
#     image_tensor = image_tensor.to(device)
#     # Run inference
#     with torch.no_grad():
#         pred_mask = model(image_tensor)
#         pred_mask = torch.sigmoid(pred_mask).squeeze(0).cpu().numpy()
#         if pred_mask.ndim == 3:
#             pred_mask = pred_mask[0]
#     binary_mask = (pred_mask > 0.5).astype(np.uint8)
#     # Assume binary_mask is already at original dimensions from OpenCV preprocessing.
#     skeleton = skeletonize_shoreline(binary_mask)
#     overlay_image = overlay_centerline(original_image, skeleton)
#     # Save the centerline as a shapefile
#     shapefile_name = f"{os.path.splitext(filename)[0]}_centerline.shp"
#     shp_path, _ = save_centerline_to_shapefile(skeleton, filepath, shapefile_name)
#     # Create a Leafmap map showing the shapefile
#     m = create_leafmap(shp_path)
#     map_output = os.path.join(app.config["OUTPUT_FOLDER"], f"{os.path.splitext(filename)[0]}_map.html")
#     m.to_html(map_output)
#     # Render result template (assume result.html exists in templates)
#     return render_template("result.html", 
#                            original_filename=filename,
#                            map_filename=os.path.basename(map_output),
#                            shapefile_name=shapefile_name)

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route("/outputs/<filename>")
# def output_file(filename):
#     return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

# if __name__ == "__main__":
#     try:
#         # Disable reloader to avoid WinError on Windows
#         app.run(debug=True, use_reloader=False)
#     except OSError as e:
#         if e.errno == 10038:
#             pass
#         else:
#             raise
