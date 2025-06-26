import cv2
import torch
import numpy as np
import os
import pyrealsense2 as rs
from torchvision import transforms

# NOTE: Requires segmentation_models_pytorch and timm
# Install with: pip install segmentation-models-pytorch timm
import segmentation_models_pytorch as smp

# Print available encoders for user reference
print('Available encoders:', smp.encoders.get_encoder_names())

# Output directories
os.makedirs('masks', exist_ok=True)
os.makedirs('depth', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose backbone: 'mobilenet_v2' or 'efficientnet-b0' (mobilenet_v3_large is not supported)
BACKBONE = 'mobilenet_v2'  # or 'efficientnet-b0'
print(f'Using backbone: {BACKBONE}')
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,  # Binary segmentation (adjust if needed)
)
model.eval()
model.to(device)

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
            output = model(input_tensor)
        # Output shape: [1, 1, H, W] for binary, [1, C, H, W] for multi-class
        mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        seg_mask = (mask > 0.5).astype(np.uint8) * 255
        seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
        seg_overlay = cv2.addWeighted(frame_rgb, 0.5, seg_mask_color, 0.5, 0)
        seg_overlay_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

        cv2.imshow('U-Net Segmentation', seg_overlay_bgr)
        cv2.imwrite(f'masks/mask_{frame_count:05d}.png', seg_mask)
        cv2.imwrite(f'depth/depth_{frame_count:05d}.png', depth_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 