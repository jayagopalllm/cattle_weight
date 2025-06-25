import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
from torchvision import transforms
import pyrealsense2 as rs

# Create output directories for masks and depth
os.makedirs('test_cattle_weight/masks/unet', exist_ok=True)
os.makedirs('test_cattle_weight/depth/unet', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained U-Net (for demonstration, use a model trained on ImageNet)
unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
unet.eval()
unet.to(device)

# Simple normalization for U-Net (ImageNet stats)
unet_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# RealSense pipeline setup
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
        depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit depth
        orig_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- U-Net ---
        unet_input = unet_preprocess(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            unet_pred = torch.sigmoid(unet(unet_input))[0,0]
        unet_mask = (unet_pred > 0.5).cpu().numpy().astype(np.uint8)
        unet_mask_resized = cv2.resize(unet_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        unet_overlay = frame.copy()
        unet_overlay[unet_mask_resized==1] = [0,255,0]

        # --- Display ---
        cv2.imshow('U-Net Segmentation', unet_overlay)

        # --- Save mask and depth ---
        cv2.imwrite(f'test_cattle_weight/masks/unet/mask_{frame_count:05d}.png', unet_mask_resized*255)
        cv2.imwrite(f'test_cattle_weight/depth/unet/depth_{frame_count:05d}.png', depth_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 