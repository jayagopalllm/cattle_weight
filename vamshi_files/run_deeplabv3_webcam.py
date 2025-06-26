import cv2
import torch
import torchvision
import numpy as np
import os
from torchvision import transforms
import pyrealsense2 as rs

# Create output directories for masks and depth
os.makedirs('test_cattle_weight/masks/deeplabv3', exist_ok=True)
os.makedirs('test_cattle_weight/depth/deeplabv3', exist_ok=True)
os.makedirs('test_cattle_weight/grayscale/deeplabv3', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the new weights API to avoid deprecation warnings
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()
model.to(device)

def get_palette():
    # Standard Pascal VOC palette
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)
    return palette

# For DeepLabV3
deeplab_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check for RealSense device availability
ctx = rs.context()
if len(ctx.devices) == 0:
    raise RuntimeError("No Intel RealSense device found. Please connect the camera and try again.")

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
# Lower FPS to 6 for better compatibility
config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 6)
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 6)
pipeline.start(config)

frame_count = 0
SKIP = 4  # Only process every 5th frame
mode = 'color'  # Modes: 'color', 'grayscale', 'edge'
print("Press 'q' to quit. Press 'c' for color, 'g' for grayscale, 'e' for edge mode.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit depth
        orig_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % (SKIP + 1) != 0:
            frame_count += 1
            continue

        # --- Mode selection ---
        if mode == 'color':
            input_img = frame_rgb
            display_img = frame_rgb.copy()
            window_title = 'DeepLabV3 Segmentation (Color)'
            save_gray = False
        elif mode == 'grayscale':
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_3ch = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            input_img = frame_gray_3ch
            display_img = frame_gray_3ch.copy()
            window_title = 'DeepLabV3 Segmentation (Grayscale)'
            save_gray = True
        elif mode == 'edge':
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(frame_gray, 100, 200)
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            input_img = edges_3ch
            display_img = edges_3ch.copy()
            window_title = 'DeepLabV3 Segmentation (Edge)'
            save_gray = True
        else:
            input_img = frame_rgb
            display_img = frame_rgb.copy()
            window_title = 'DeepLabV3 Segmentation (Color)'
            save_gray = False

        # --- DeepLabV3 ---
        input_tensor = deeplab_transform(input_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)['out'][0]  # [C, H, W]
        seg_mask = output.argmax(0).cpu().numpy().astype(np.uint8)  # [H, W]
        palette = get_palette()
        seg_mask_color = palette[seg_mask % len(palette)]
        seg_overlay = cv2.addWeighted(display_img, 0.5, seg_mask_color, 0.5, 0)
        seg_overlay_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

        # --- Display ---
        cv2.imshow(window_title, seg_overlay_bgr)

        # --- Save mask and depth ---
        cv2.imwrite(f'test_cattle_weight/masks/deeplabv3/mask_{frame_count:05d}.png', seg_mask)
        cv2.imwrite(f'test_cattle_weight/depth/deeplabv3/depth_{frame_count:05d}.png', depth_image)
        if save_gray:
            cv2.imwrite(f'test_cattle_weight/grayscale/deeplabv3/gray_{frame_count:05d}.png', frame_gray)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            mode = 'color'
        elif key == ord('g'):
            mode = 'grayscale'
        elif key == ord('e'):
            mode = 'edge'
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 