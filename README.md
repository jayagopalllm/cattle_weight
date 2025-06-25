# Cattle Weight Prediction

This project focuses on predicting cattle weight based on keypoint data. By leveraging machine learning techniques, the model estimates the weight of cattle using key points from images or structured data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a deep learning regression model that predicts the weight of cattle using keypoint data. Key points represent specific parts of the cattle's body, such as the head, legs, and torso, and are used to estimate the weight.

## Dataset

We are gathering real-time data from a cattle farm, focusing on two main approaches. To support this, we are implementing a series of forms to capture image data of the cattle. For each cow, we will collect four images—front, back, side, and top views—along with the ground truth weight of the animal.






=====================================================================================================================================


#Test-claude.ai inputs:

STEP 1:

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install python3-pip python3-venv git cmake build-essential

# Create virtual environment
python3 -m venv cattle_weight_env
source cattle_weight_env/bin/activate

# Install RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev

# Install Python packages
pip install pyrealsense2 opencv-python numpy scipy scikit-learn pandas matplotlib


-----------------------------------------------------------------------------------------
STEP 2:Test Camera Setup


#!/usr/bin/env python3
"""
Test script for RealSense D435i camera
This script captures and displays depth and color frames
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        pipeline.start(config)
        print("Camera started successfully!")
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap to depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.imshow('RealSense - Color and Depth', images)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()

-----------------------------------------------------------------------------



----------------------------------------------------------------------------------------




--------------------------------------------------------------------------------------------------------

Step 6: Complete Integration System


#!/usr/bin/env python3
"""
Complete Cattle Weight Prediction System
Integrates camera measurement and weight prediction
"""

import sys
import os
from datetime import datetime
import argparse

# Import our custom modules (make sure they're in the same directory)
try:
    from cattle_measurement import CattleMeasurement
    from weight_prediction import CattleWeightPredictor
except ImportError:
    print("Error: Please ensure cattle_measurement.py and weight_prediction.py are in the same directory")
    sys.exit(1)

class CompleteCattleSystem:
    def __init__(self):
        self.measurer = CattleMeasurement()
        self.predictor = CattleWeightPredictor()
        self.results = []
    
    def setup_system(self):
        """Setup and validate system components"""
        print("Setting up Cattle Weight Prediction System...")
        
        # Check if model exists, if not, create sample data and train
        if not os.path.exists('cattle_weight_model.pkl'):
            print("No trained model found. Creating sample dataset and training model...")
            data = self.predictor.load_data()
            if data is not None:
                self.predictor.train_model(data)
            else:
                print("Failed to create training data")
                return False
        else:
            # Load existing model
            if not self.predictor.load_model():
                print("Failed to load existing model")
                return False
        
        print("System setup complete!")
        return True
    
    def live_measurement_and_prediction(self):
        """Run live measurement with real-time weight prediction"""
        if not self.measurer.start_camera():
            return
        
        print("Live Cattle Weight Prediction System")
        print("Controls:")
        print("  'c' - Capture measurement and predict weight")
        print("  's' - Save current session results")
        print("  'q' - Quit")
        
        session_results = []
        
        try:
            while True:
                # Capture measurement
                measurements, display_image, mask = self.measurer.capture_measurement()
                
                if display_image is not None:
                    # Show camera feed
                    cv2.imshow('Cattle Detection', display_image)
                    if mask is not None:
                        cv2.imshow('Depth Mask', mask)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c') and measurements:
                    # Predict weight from measurements
                    weight = self.predictor.predict_weight(measurements)
                    
                    if weight is not None:
                        predicted_weight = weight[0]
                        measurements['predicted_weight_kg'] = predicted_weight
                        measurements['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        session_results.append(measurements)
                        
                        print(f"\n--- New Measurement ---")
                        print(f"Timestamp: {measurements['timestamp']}")
                        print(f"Length: {measurements.get('estimated_length_m', 'N/A'):.2f}m")
                        print(f"Width: {measurements.get('estimated_width_m', 'N/A'):.2f}m")
                        print(f"Height: {measurements.get('estimated_height_m', 'N/A'):.2f}m")
                        print(f"Girth: {measurements.get('estimated_girth_m', 'N/A'):.2f}m")
                        print(f"PREDICTED WEIGHT: {predicted_weight:.1f} kg")
                        print("-" * 30)
                    else:
                        print("Failed to predict weight")
                
                elif key == ord('s') and session_results:
                    # Save session results
                    filename = f"cattle_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    import json
                    with open(filename, 'w') as f:
                        json.dump(session_results, f, indent=2)
                    print(f"Session results saved to {filename}")
                
                elif key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
        
        finally:
            self.measurer.stop_camera()
            
            # Final summary
            if session_results:
                print(f"\nSession Summary:")
                print(f"Total measurements: {len(session_results)}")
                weights = [r['predicted_weight_kg'] for r in session_results if 'predicted_weight_kg' in r]
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    min_weight = min(weights)
                    max_weight = max(weights)
                    print(f"Average weight: {avg_weight:.1f} kg")
                    print(f"Weight range: {min_weight:.1f} - {max_weight:.1f} kg")
    
    def batch_process_images(self, image_directory):
        """Process a batch of saved images"""
        print(f"Processing images from {image_directory}")
        # This would be implemented for offline processing of saved images
        # For now, we'll focus on live processing
        pass
    
    def calibrate_system(self):
        """Calibrate the measurement system with known objects"""
        print("System calibration mode")
        print("Place a reference object of known dimensions in view")
        print("This helps improve measurement accuracy")
        
        if not self.measurer.start_camera():
            return
        
        try:
            print("Press 'c' to capture reference measurement, 'q' to quit calibration")
            
            while True:
                measurements, display_image, mask = self.measurer.capture_measurement()
                
                if display_image is not None:
                    cv2.imshow('Calibration Mode', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c') and measurements:
                    print("Reference measurement captured:")
                    print(f"Length: {measurements.get('estimated_length_m', 'N/A'):.3f}m")
                    print(f"Width: {measurements.get('estimated_width_m', 'N/A'):.3f}m") 
                    print(f"Height: {measurements.get('estimated_height_m', 'N/A'):.3f}m")
                    
                    actual_length = float(input("Enter actual length (m): "))
                    actual_width = float(input("Enter actual width (m): "))
                    actual_height = float(input("Enter actual height (m): "))
                    
                    # Calculate calibration factors
                    length_factor = actual_length / measurements.get('estimated_length_m', 1)
                    width_factor = actual_width / measurements.get('estimated_width_m', 1)
                    height_factor = actual_height / measurements.get('estimated_height_m', 1)
                    
                    print(f"Calibration factors:")
                    print(f"Length: {length_factor:.3f}")
                    print(f"Width: {width_factor:.3f}")
                    print(f"Height: {height_factor:.3f}")
                    
                    # Save calibration data
                    calibration_data = {
                        'length_factor': length_factor,
                        'width_factor': width_factor,
                        'height_factor': height_factor,
                        'calibration_date': datetime.now().isoformat()
                    }
                    
                    import json
                    with open('calibration_data.json', 'w') as f:
                        json.dump(calibration_data, f, indent=2)
                    
                    print("Calibration data saved!")
                    break
                
                elif key == ord('q'):
                    break
        
        finally:
            self.measurer.stop_camera()
    
    def generate_report(self, session_file=None):
        """Generate a detailed report of measurements and predictions"""
        if session_file:
            import json
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            except Exception as e:
                print(f"Error loading session file: {e}")
                return
        else:
            print("No session file specified")
            return
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cattle Weight Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .measurement {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .weight {{ font-size: 1.2em; font-weight: bold; color: #2c5aa0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cattle Weight Prediction Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total measurements: {len(session_data)}</p>
            </div>
        """
        
        # Add measurements table
        html_content += """
            <h2>Measurements Summary</h2>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Length (m)</th>
                    <th>Width (m)</th>
                    <th>Height (m)</th>
                    <th>Girth (m)</th>
                    <th>Predicted Weight (kg)</th>
                </tr>
        """
        
        for measurement in session_data:
            html_content += f"""
                <tr>
                    <td>{measurement.get('timestamp', 'N/A')}</td>
                    <td>{measurement.get('estimated_length_m', 'N/A'):.2f}</td>
                    <td>{measurement.get('estimated_width_m', 'N/A'):.2f}</td>
                    <td>{measurement.get('estimated_height_m', 'N/A'):.2f}</td>
                    <td>{measurement.get('estimated_girth_m', 'N/A'):.2f}</td>
                    <td class="weight">{measurement.get('predicted_weight_kg', 'N/A'):.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Save report
        report_filename = f"cattle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='Cattle Weight Prediction System')
    parser.add_argument('--mode', choices=['live', 'train', 'calibrate', 'report'], 
                       default='live', help='System operation mode')
    parser.add_argument('--session-file', help='Session file for report generation')
    
    args = parser.parse_args()
    
    system = CompleteCattleSystem()
    
    if args.mode == 'live':
        # Live measurement and prediction mode
        if system.setup_system():
            system.live_measurement_and_prediction()
    
    elif args.mode == 'train':
        # Training mode - create new model
        print("Training new model...")
        data = system.predictor.load_data()
        if data is not None:
            X_test, y_test, y_pred = system.predictor.train_model(data)
            system.predictor.plot_predictions(X_test, y_test, y_pred)
        
    elif args.mode == 'calibrate':
        # Calibration mode
        system.calibrate_system()
        
    elif args.mode == 'report':
        # Report generation mode
        if args.session_file:
            system.generate_report(args.session_file)
        else:
            print("Please specify --session-file for report generation")

if __name__ == "__main__":
    # Add missing import for cv2
    import cv2
    main()



