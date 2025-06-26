import cv2
import torch
import torchvision
import numpy as np
import os
from torchvision import transforms
import pyrealsense2 as rs

# Output directories
os.makedirs('masks', exist_ok=True)
os.makedirs('depth', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DeepLabV3 with MobileNetV3 backbone
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()
model.to(device)

def get_palette():
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)
    return palette

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# RealSense setup
ctx = rs.context()
if len(ctx.devices) == 0:
    raise RuntimeError("No Intel RealSense device found. Please connect the camera and try again.")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frame_count = 0
print("Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        seg_mask = output.argmax(0).cpu().numpy().astype(np.uint8)
        palette = get_palette()
        seg_mask_color = palette[seg_mask % len(palette)]
        seg_overlay = cv2.addWeighted(frame_rgb, 0.5, seg_mask_color, 0.5, 0)
        seg_overlay_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

        cv2.imshow('DeepLabV3-MobileNetV3 Segmentation', seg_overlay_bgr)
        cv2.imwrite(f'masks/mask_{frame_count:05d}.png', seg_mask)
        cv2.imwrite(f'depth/depth_{frame_count:05d}.png', depth_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 