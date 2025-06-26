import cv2
import numpy as np
import os
import pyrealsense2 as rs
import onnxruntime as ort
from torchvision import transforms

# NOTE: Requires onnxruntime and a SegFormer-B0 ONNX model
# Install with: pip install onnxruntime
# Download or export SegFormer-B0 ONNX model and set the path below
ONNX_MODEL_PATH = 'segformer_b0.onnx'

# Output directories
os.makedirs('masks', exist_ok=True)
os.makedirs('depth', exist_ok=True)

# Load ONNX model
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

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

        input_tensor = transform(frame_rgb).unsqueeze(0).numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        # Assume output is [1, num_classes, H, W]
        seg_mask = np.argmax(ort_outs[0], axis=1)[0].astype(np.uint8)
        # Simple color map
        palette = np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128]], dtype=np.uint8)
        seg_mask_color = palette[seg_mask % len(palette)]
        seg_overlay = cv2.addWeighted(frame_rgb, 0.5, seg_mask_color, 0.5, 0)
        seg_overlay_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

        cv2.imshow('SegFormer-B0 Segmentation', seg_overlay_bgr)
        cv2.imwrite(f'masks/mask_{frame_count:05d}.png', seg_mask)
        cv2.imwrite(f'depth/depth_{frame_count:05d}.png', depth_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows() 