# Code for data loading and preprocessing
import os  # For interacting with the operating system, like file paths
import torch  # For deep learning and PyTorch framework
import numpy as np  # For numerical operations on arrays
import cv2  # For image processing using OpenCV
from PIL import Image  # For handling images using the Python Imaging Library
import matplotlib.pyplot as plt  # For plotting images and graphs
from torchvision import transforms  # For data transformations in image processing
from scipy.spatial import distance  # To calculate Euclidean distance

# Load the pre-trained DeepLabV3 model with a ResNet50 backbone for semantic segmentation
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Preprocessing pipeline for input images
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to 512x512 pixels
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
])

# Function to extract points on vertical and horizontal lines from the segmentation mask
def extract_line_points(mask, num_vertical=10, num_horizontal=5):
    height, width = mask.shape  # Get dimensions of the mask
    
    # Extract points along vertical lines
    vertical_lines_points = []
    for i in range(1, num_vertical + 1):
        x = width * i // (num_vertical + 1)  # Calculate the x-coordinate for vertical lines
        # Get all non-zero points along the vertical line
        line_points = [(x, y) for y in range(height) if mask[y, x] > 0]  
        vertical_lines_points.append(line_points)  # Append the line points
    
    # Extract points along horizontal lines
    horizontal_lines_points = []
    for i in range(1, num_horizontal + 1):
        y = height * i // (num_horizontal + 1)  # Calculate the y-coordinate for horizontal lines
        # Get all non-zero points along the horizontal line
        line_points = [(x, y) for x in range(width) if mask[y, x] > 0]
        horizontal_lines_points.append(line_points)  # Append the line points
    
    return vertical_lines_points, horizontal_lines_points  # Return the points

# Function to find the maximum Euclidean distance between pairs of points
def find_max_euclidean_distance(points):
    max_distance = 0  # Initialize maximum distance
    point_pair = None  # Initialize the point pair with maximum distance

    # Iterate over all pairs of points to calculate Euclidean distance
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]  # Get points
            dist = distance.euclidean(p1, p2)  # Calculate the distance between points
            if dist > max_distance:  # Update if found a new maximum distance
                max_distance = dist
                point_pair = (p1, p2)
    
    return point_pair, max_distance  # Return the point pair and the maximum distance

# Function to process a single image for cattle weight prediction
def process_image(image_path):
    # Set device for model processing (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move model to the appropriate device
    
    input_image = Image.open(image_path).convert("RGB")  # Load and convert image to RGB
    original_image = np.array(input_image)  # Convert image to a NumPy array
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)  # Preprocess image and add batch dimension

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_tensor)['out'][0]  # Get the model output
    output_predictions = output.argmax(0).cpu().numpy()  # Get the predicted class for each pixel

    # Create a binary segmentation mask (foreground vs background)
    segmentation_mask = (output_predictions > 0).astype(np.uint8)

    # Step 1: Extract points on vertical and horizontal lines from the mask
    vertical_lines_points, horizontal_lines_points = extract_line_points(segmentation_mask)

    # Step 2: Find the maximum Euclidean distance points for vertical and horizontal lines
    max_vertical_pair = None
    max_horizontal_pair = None
    max_vertical_distance = 0
    max_horizontal_distance = 0

    # Calculate maximum distance for vertical line points
    for vertical_points in vertical_lines_points:
        if len(vertical_points) > 1:  # Ensure there are at least two points to calculate distance
            vertical_pair, vertical_distance = find_max_euclidean_distance(vertical_points)
            if vertical_distance > max_vertical_distance:  # Update if a new max distance is found
                max_vertical_distance = vertical_distance
                max_vertical_pair = vertical_pair
    
    # Calculate maximum distance for horizontal line points
    for horizontal_points in horizontal_lines_points:
        if len(horizontal_points) > 1:  # Ensure there are at least two points to calculate distance
            horizontal_pair, horizontal_distance = find_max_euclidean_distance(horizontal_points)
            if horizontal_distance > max_horizontal_distance:  # Update if a new max distance is found
                max_horizontal_distance = horizontal_distance
                max_horizontal_pair = horizontal_pair

    # Step 3: Mark the key points on the image
    mask_colored = apply_color_map(output_predictions)  # Apply a color map to the mask for visualization

    # Draw circles on the image for the maximum distance points found
    if max_vertical_pair:
        cv2.circle(mask_colored, max_vertical_pair[0], radius=5, color=(0, 255, 0), thickness=-1)  # Mark first point in green
        cv2.circle(mask_colored, max_vertical_pair[1], radius=5, color=(0, 255, 0), thickness=-1)  # Mark second point in green

    if max_horizontal_pair:
        cv2.circle(mask_colored, max_horizontal_pair[0], radius=5, color=(0, 255, 0), thickness=-1)  # Mark first point in green
        cv2.circle(mask_colored, max_horizontal_pair[1], radius=5, color=(0, 255, 0), thickness=-1)  # Mark second point in green

    # Step 4: Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # Create a subplot for the original image
    plt.title('Original Image')
    plt.imshow(original_image)  # Show the original image
    plt.axis('off')  # Hide axis

    plt.subplot(1, 2, 2)  # Create a subplot for the segmented mask
    plt.title('Segmented Mask with Key Points')
    plt.imshow(mask_colored)  # Show the mask with key points
    plt.axis('off')  # Hide axis

    plt.show()  # Render the plots

    # Output the distances and points for user reference
    if max_vertical_pair:
        print(f"Max Vertical Euclidean Distance: {max_vertical_distance}")
        print(f"Points: {max_vertical_pair}")

    if max_horizontal_pair:
        print(f"Max Horizontal Euclidean Distance: {max_horizontal_distance}")
        print(f"Points: {max_horizontal_pair}")

# Function to process all images in a given folder, limiting to a specified number of images
def process_images_in_folder(folder_path, limit=3):
    count = 0  # Initialize a counter for the number of images processed
    for image_name in os.listdir(folder_path):  # Iterate through each file in the directory
        image_path = os.path.join(folder_path, image_name)  # Create the full image path
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
            print(f"Processing: {image_name}")  # Output the current processing image
            process_image(image_path)  # Process the image
            count += 1  # Increment the counter
            if count >= limit:  # Stop after processing a limited number of images
                break

# Specify the folder containing images to process
folder_path = r'D:\Office-Work\New folder\bbox-yolov5-cattle-top-view.v1i.coco\train'  
process_images_in_folder(folder_path)  # Call the function to process images in the specified folder
