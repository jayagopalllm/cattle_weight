import cv2
import torch
import torchvision
import numpy as np
import os
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import pyrealsense2 as rs

# Create output directories for masks and depth
os.makedirs('test_cattle_weight/masks/maskrcnn', exist_ok=True)
os.makedirs('test_cattle_weight/depth/maskrcnn', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained Mask R-CNN (COCO)
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn.eval()
maskrcnn.to(device)

# For Mask R-CNN
maskrcnn_transform = transforms.Compose([
    transforms.ToTensor()
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

        # --- Mask R-CNN ---
        input_tensor = maskrcnn_transform(frame_rgb).to(device)
        with torch.no_grad():
            outputs = maskrcnn([input_tensor])[0]
        masks = outputs['masks'] > 0.5  # [N, 1, H, W]
        if len(masks) > 0:
            maskrcnn_mask = torch.zeros_like(masks[0][0], dtype=torch.bool)
            for m in masks:
                maskrcnn_mask |= m[0]
            maskrcnn_mask_np = maskrcnn_mask.cpu().numpy().astype(np.uint8) * 255
            maskrcnn_overlay = draw_segmentation_masks(torch.from_numpy(frame_rgb).permute(2,0,1), maskrcnn_mask, alpha=0.5)
            maskrcnn_overlay = maskrcnn_overlay.permute(1,2,0).byte().cpu().numpy()
            maskrcnn_overlay = cv2.cvtColor(maskrcnn_overlay, cv2.COLOR_RGB2BGR)
        else:
            maskrcnn_mask_np = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            maskrcnn_overlay = orig_frame

        # --- Display ---
        cv2.imshow('Mask R-CNN Segmentation', maskrcnn_overlay)

        # --- Save mask and depth ---
        cv2.imwrite(f'test_cattle_weight/masks/maskrcnn/mask_{frame_count:05d}.png', maskrcnn_mask_np)
        cv2.imwrite(f'test_cattle_weight/depth/maskrcnn/depth_{frame_count:05d}.png', depth_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 