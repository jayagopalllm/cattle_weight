import cv2
import numpy as np
import os
import sys
import pyrealsense2 as rs

def enhance_image(img):
    # 1. Denoise
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    # 2. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # 3. Thresholding (Otsu)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 4. Morphological opening
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

input_dir = 'test_cattle_weight/masks/maskrcnn/'
output_dir = 'test_cattle_weight/enhanced/maskrcnn/'
os.makedirs(output_dir, exist_ok=True)

mode = 'folder'
if len(sys.argv) > 1:
    mode = sys.argv[1].lower()

if mode == 'folder':
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.png'):
            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path, 0)  # Load as grayscale
            enhanced = enhance_image(img)
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, enhanced)
    print(f"Enhanced images saved to {output_dir}")

elif mode == 'realsense':
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("Press 'q' to quit capturing.")
    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = enhance_image(frame_gray)
            out_path = os.path.join(output_dir, f'enhanced_{frame_count:05d}.png')
            cv2.imwrite(out_path, enhanced)
            cv2.imshow('Enhanced RealSense Grayscale', enhanced)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"Enhanced RealSense images saved to {output_dir}")
else:
    print("Unknown mode. Use 'folder' (default) or 'realsense'.") 