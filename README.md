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

Step 4: Cattle Detection and Measurement

#!/usr/bin/env python3
"""
Cattle body measurement system using RealSense D435i
Extracts key body measurements for weight prediction
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
from datetime import datetime
import os

class CattleMeasurement:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.measurements = []
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
    def start_camera(self):
        """Start the RealSense camera"""
        try:
            self.pipeline.start(self.config)
            print("Camera started successfully!")
            return True
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the RealSense camera"""
        self.pipeline.stop()
        cv2.destroyAllWindows()
    
    def preprocess_frame(self, depth_frame, color_frame):
        """Preprocess frames for measurement"""
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply depth filtering
        depth_image = cv2.medianBlur(depth_image.astype(np.uint16), 5)
        
        return depth_image, color_image
    
    def detect_cattle_contour(self, depth_image, color_image):
        """Detect cattle contour using depth thresholding"""
        # Convert depth to 8-bit for processing
        depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        
        # Create mask for objects within cattle distance (1-4 meters)
        mask = cv2.inRange(depth_image, 1000, 4000)  # 1-4 meters in mm
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assuming it's the cattle)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area (cattle should be reasonably large)
            if cv2.contourArea(largest_contour) > 5000:  # Adjust threshold as needed
                return largest_contour, mask
        
        return None, mask
    
    def calculate_measurements(self, contour, depth_image, depth_frame):
        """Calculate key body measurements from contour and depth data"""
        measurements = {}
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate basic measurements
        measurements['bounding_width'] = w
        measurements['bounding_height'] = h
        measurements['contour_area'] = cv2.contourArea(contour)
        
        # Get depth intrinsics for real-world measurements
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Calculate real-world dimensions
        # Get depth at center of bounding box
        center_x, center_y = x + w//2, y + h//2
        depth_at_center = depth_image[center_y, center_x]
        
        if depth_at_center > 0:
            # Convert pixel measurements to real-world measurements (meters)
            # This is a simplified calculation - for more accuracy, use point cloud
            pixel_size = depth_at_center / depth_intrinsics.fx  # Approximate pixel size at that depth
            
            measurements['estimated_width_m'] = w * pixel_size / 1000  # Convert to meters
            measurements['estimated_height_m'] = h * pixel_size / 1000
            measurements['depth_at_center_m'] = depth_at_center / 1000
            
            # Calculate body length (assuming side view)
            # For more accuracy, you'd need to identify specific body parts
            measurements['estimated_length_m'] = measurements['estimated_width_m']
            
            # Calculate approximate girth (circumference)
            # This is a rough estimation - actual implementation would need more sophisticated methods
            measurements['estimated_girth_m'] = 2 * np.pi * (measurements['estimated_height_m'] / 2)
        
        return measurements
    
    def capture_measurement(self):
        """Capture a single measurement"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align frames
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None, None
            
            # Preprocess frames
            depth_image, color_image = self.preprocess_frame(depth_frame, color_frame)
            
            # Detect cattle contour
            contour, mask = self.detect_cattle_contour(depth_image, color_image)
            
            if contour is not None:
                # Calculate measurements
                measurements = self.calculate_measurements(contour, depth_image, depth_frame)
                
                # Add timestamp
                measurements['timestamp'] = datetime.now().isoformat()
                
                # Draw contour on color image for visualization
                display_image = color_image.copy()
                cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Add measurement text
                if 'estimated_width_m' in measurements:
                    text = f"W: {measurements['estimated_width_m']:.2f}m H: {measurements['estimated_height_m']:.2f}m"
                    cv2.putText(display_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                return measurements, display_image, mask
            
            return None, color_image, mask
            
        except Exception as e:
            print(f"Error capturing measurement: {e}")
            return None, None, None
    
    def save_measurement(self, measurements, filename="cattle_measurements.json"):
        """Save measurements to JSON file"""
        if measurements:
            self.measurements.append(measurements)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(self.measurements, f, indent=2)
            
            print(f"Measurement saved: {measurements}")
    
    def run_measurement_session(self):
        """Run interactive measurement session"""
        if not self.start_camera():
            return
        
        print("Cattle Measurement System Started")
        print("Press 'c' to capture measurement, 'q' to quit")
        
        try:
            while True:
                measurements, display_image, mask = self.capture_measurement()
                
                if display_image is not None:
                    # Show images
                    cv2.imshow('Cattle Detection', display_image)
                    cv2.imshow('Depth Mask', mask)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c') and measurements:
                    # Capture measurement
                    self.save_measurement(measurements)
                    print("Measurement captured!")
                
                elif key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Measurement session interrupted")
        
        finally:
            self.stop_camera()

def main():
    measurer = CattleMeasurement()
    measurer.run_measurement_session()

if __name__ == "__main__":
    main()


----------------------------------------------------------------------------------------

Step 5: Weight Prediction Model

#!/usr/bin/env python3
"""
Cattle weight prediction model using body measurements
Includes data collection, training, and prediction functionality
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os

class CattleWeightPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['estimated_length_m', 'estimated_width_m', 'estimated_height_m', 'estimated_girth_m']
        self.model_file = 'cattle_weight_model.pkl'
        self.scaler_file = 'cattle_weight_scaler.pkl'
    
    def create_sample_dataset(self, filename='sample_cattle_data.csv'):
        """Create a sample dataset for demonstration (replace with real data)"""
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic cattle measurements (in meters)
        # These are rough estimates based on typical cattle dimensions
        length = np.random.normal(2.0, 0.3, n_samples)  # 1.4-2.6m typical
        width = np.random.normal(0.6, 0.1, n_samples)   # 0.4-0.8m typical
        height = np.random.normal(1.4, 0.2, n_samples)  # 1.0-1.8m typical
        girth = np.random.normal(2.2, 0.3, n_samples)   # Chest circumference
        
        # Ensure positive values
        length = np.abs(length)
        width = np.abs(width)
        height = np.abs(height)
        girth = np.abs(girth)
        
        # Calculate weight using a realistic formula (simplified)
        # Real cattle weight estimation formulas exist but are more complex
        # This is a simplified version for demonstration
        weight = (length * width * height * 400) + (girth * 50) + np.random.normal(0, 20, n_samples)
        weight = np.clip(weight, 200, 800)  # Typical cattle weight range (kg)
        
        # Create DataFrame
        data = pd.DataFrame({
            'estimated_length_m': length,
            'estimated_width_m': width,
            'estimated_height_m': height,
            'estimated_girth_m': girth,
            'weight_kg': weight
        })
        
        # Save to CSV
        data.to_csv(filename, index=False)
        print(f"Sample dataset created: {filename}")
        return data
    
    def load_data(self, filename='sample_cattle_data.csv'):
        """Load cattle measurement data"""
        try:
            if not os.path.exists(filename):
                print(f"Data file {filename} not found. Creating sample dataset...")
                return self.create_sample_dataset(filename)
            
            data = pd.read_csv(filename)
            print(f"Loaded {len(data)} records from {filename}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def load_measurements_from_json(self, json_file='cattle_measurements.json'):
        """Load measurements from JSON file created by measurement system"""
        try:
            with open(json_file, 'r') as f:
                measurements = json.load(f)
            
            # Convert to DataFrame
            df_data = []
            for measurement in measurements:
                if all(key in measurement for key in self.feature_names):
                    df_data.append({
                        'estimated_length_m': measurement['estimated_length_m'],
                        'estimated_width_m': measurement['estimated_width_m'],
                        'estimated_height_m': measurement['estimated_height_m'],
                        'estimated_girth_m': measurement['estimated_girth_m']
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                print(f"Loaded {len(df)} measurements from {json_file}")
                return df
            else:
                print("No valid measurements found in JSON file")
                return None
                
        except Exception as e:
            print(f"Error loading measurements from JSON: {e}")
            return None
    
    def prepare_features(self, data):
        """Prepare features for training/prediction"""
        # Select feature columns
        X = data[self.feature_names].copy()
        
        # Add derived features
        X['body_volume'] = X['estimated_length_m'] * X['estimated_width_m'] * X['estimated_height_m']
        X['length_to_height_ratio'] = X['estimated_length_m'] / X['estimated_height_m']
        X['girth_to_length_ratio'] = X['estimated_girth_m'] / X['estimated_length_m']
        
        return X
    
    def train_model(self, data):
        """Train the weight prediction model"""
        print("Training cattle weight prediction model...")
        
        # Prepare features and target
        X = self.prepare_features(data)
        y = data['weight_kg']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('-inf')
        
        print("\nModel Comparison:")
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name}:")
            print(f"  MAE: {mae:.2f} kg")
            print(f"  RMSE: {rmse:.2f} kg")
            print(f"  R²: {r2:.3f}")
            
            # Select best model based on R²
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.model = model
        
        print(f"\nBest model selected with R² = {best_score:.3f}")
        
        # Save model and scaler
        self.save_model()
        
        return X_test_scaled, y_test, self.model.predict(X_test_scaled)
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.model and self.scaler:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print(f"Model saved to {self.model_file}")
            print(f"Scaler saved to {self.scaler_file}")
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)
            print("Model and scaler loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_weight(self, measurements):
        """Predict weight from measurements"""
        if not self.model or not self.scaler:
            if not self.load_model():
                print("No trained model available. Please train model first.")
                return None
        
        try:
            # Convert measurements to DataFrame
            if isinstance(measurements, dict):
                measurements = pd.DataFrame([measurements])
            
            # Prepare features
            X = self.prepare_features(measurements)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting weight: {e}")
            return None
    
    def predict_from_json_measurements(self, json_file='cattle_measurements.json'):
        """Predict weights for measurements in JSON file"""
        measurements_df = self.load_measurements_from_json(json_file)
        
        if measurements_df is not None:
            predictions = self.predict_weight(measurements_df)
            
            if predictions is not None:
                results = measurements_df.copy()
                results['predicted_weight_kg'] = predictions
                
                print("\nWeight Predictions:")
                for i, (_, row) in enumerate(results.iterrows()):
                    print(f"Measurement {i+1}: {predictions[i]:.1f} kg")
                    print(f"  Length: {row['estimated_length_m']:.2f}m")
                    print(f"  Width: {row['estimated_width_m']:.2f}m")
                    print(f"  Height: {row['estimated_height_m']:.2f}m")
                    print(f"  Girth: {row['estimated_girth_m']:.2f}m")
                    print()
                
                return results
        
        return None
    
    def plot_predictions(self, X_test, y_test, y_pred):
        """Plot prediction results"""
        plt.figure(figsize=(10, 6))
        
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Weight (kg)')
        plt.ylabel('Predicted Weight (kg)')
        plt.title('Actual vs Predicted Weight')
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Weight (kg)')
        plt.ylabel('Residuals (kg)')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('weight_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    predictor = CattleWeightPredictor()
    
    print("Cattle Weight Prediction System")
    print("1. Train new model")
    print("2. Predict from existing measurements")
    print("3. Single prediction")
    
    choice = input("Select option (1-3): ")
    
    if choice == '1':
        # Train new model
        data = predictor.load_data()
        if data is not None:
            X_test, y_test, y_pred = predictor.train_model(data)
            predictor.plot_predictions(X_test, y_test, y_pred)
    
    elif choice == '2':
        # Predict from JSON measurements
        results = predictor.predict_from_json_measurements()
        if results is not None:
            results.to_csv('weight_predictions.csv', index=False)
            print("Predictions saved to weight_predictions.csv")
    
    elif choice == '3':
        # Single prediction
        print("Enter cattle measurements:")
        try:
            length = float(input("Length (m): "))
            width = float(input("Width (m): "))
            height = float(input("Height (m): "))
            girth = float(input("Girth (m): "))
            
            measurements = {
                'estimated_length_m': length,
                'estimated_width_m': width,
                'estimated_height_m': height,
                'estimated_girth_m': girth
            }
            
            weight = predictor.predict_weight(measurements)
            if weight is not None:
                print(f"\nPredicted weight: {weight[0]:.1f} kg")
        
        except ValueError:
            print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()


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



