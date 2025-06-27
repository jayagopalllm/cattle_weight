import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# RECOMMENDED DATASET STRUCTURE
# =============================================================================

"""
PRODUCTION-READY DATASET STRUCTURE:

cattle_weight_dataset/
├── images/
│   ├── train/
│   │   ├── front/
│   │   │   ├── cattle_001_front.jpg
│   │   │   ├── cattle_002_front.jpg
│   │   │   └── ...
│   │   ├── side/
│   │   │   ├── cattle_001_side.jpg
│   │   │   ├── cattle_002_side.jpg
│   │   │   └── ...
│   │   └── rear/
│   │       ├── cattle_001_rear.jpg
│   │       ├── cattle_002_rear.jpg
│   │       └── ...
│   ├── val/
│   │   ├── front/
│   │   ├── side/
│   │   └── rear/
│   └── test/
│       ├── front/
│       ├── side/
│       └── rear/
├── labels/
│   ├── train_labels.csv
│   ├── val_labels.csv
│   ├── test_labels.csv
│   └── metadata.json
├── splits/
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── test_ids.txt
└── configs/
    └── dataset_config.yaml
"""

# =============================================================================
# DATASET CREATION UTILITIES
# =============================================================================

class DatasetOrganizer:
    """Utility class to organize multi-view cattle dataset"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Create the recommended directory structure"""
        dirs_to_create = [
            'images/train/front', 'images/train/side', 'images/train/rear',
            'images/val/front', 'images/val/side', 'images/val/rear',
            'images/test/front', 'images/test/side', 'images/test/rear',
            'labels', 'splits', 'configs'
        ]
        
        for dir_path in dirs_to_create:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset(self, num_samples=100):
        """Create sample dataset structure with dummy data"""
        
        # Generate sample data
        samples = []
        for i in range(1, num_samples + 1):
            cattle_id = f"cattle_{i:03d}"
            weight = np.random.normal(450, 80)  # Adult cattle weight
            age_months = np.random.randint(12, 60)
            breed = np.random.choice(['Holstein', 'Angus', 'Hereford', 'Simmental'])
            gender = np.random.choice(['Male', 'Female'])
            
            samples.append({
                'cattle_id': cattle_id,
                'weight_kg': round(weight, 1),
                'age_months': age_months,
                'breed': breed,
                'gender': gender,
                'front_image': f"{cattle_id}_front.jpg",
                'side_image': f"{cattle_id}_side.jpg",
                'rear_image': f"{cattle_id}_rear.jpg"
            })
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        
        # Split dataset
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save labels
        train_df.to_csv(self.base_path / 'labels/train_labels.csv', index=False)
        val_df.to_csv(self.base_path / 'labels/val_labels.csv', index=False)
        test_df.to_csv(self.base_path / 'labels/test_labels.csv', index=False)
        
        # Save splits
        with open(self.base_path / 'splits/train_ids.txt', 'w') as f:
            f.write('\n'.join(train_df['cattle_id'].tolist()))
        
        with open(self.base_path / 'splits/val_ids.txt', 'w') as f:
            f.write('\n'.join(val_df['cattle_id'].tolist()))
        
        with open(self.base_path / 'splits/test_ids.txt', 'w') as f:
            f.write('\n'.join(test_df['cattle_id'].tolist()))
        
        # Create metadata
        metadata = {
            'dataset_name': 'Cattle Weight Estimation Dataset',
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'weight_stats': {
                'mean': float(df['weight_kg'].mean()),
                'std': float(df['weight_kg'].std()),
                'min': float(df['weight_kg'].min()),
                'max': float(df['weight_kg'].max())
            },
            'image_format': 'jpg',
            'views': ['front', 'side', 'rear'],
            'breeds': df['breed'].unique().tolist(),
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        with open(self.base_path / 'labels/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Sample dataset structure created at: {self.base_path}")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

# =============================================================================
# PYTORCH DATASET CLASS
# =============================================================================

class MultiViewCattleDataset(Dataset):
    """PyTorch Dataset for multi-view cattle weight estimation"""
    
    def __init__(self, 
                 csv_file, 
                 images_dir, 
                 transform=None,
                 augment_views=False):
        """
        Args:
            csv_file (str): Path to CSV file with labels
            images_dir (str): Directory with images (should contain front/, side/, rear/ subdirs)
            transform (callable, optional): Optional transform to be applied on images
            augment_views (bool): Whether to apply different augmentations to different views
        """
        self.df = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.augment_views = augment_views
        
        # Verify all required columns exist
        required_cols = ['cattle_id', 'weight_kg', 'front_image', 'side_image', 'rear_image']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Verify image directories exist
        for view in ['front', 'side', 'rear']:
            view_dir = self.images_dir / view
            if not view_dir.exists():
                raise FileNotFoundError(f"View directory not found: {view_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        
        # Load images
        front_path = self.images_dir / 'front' / row['front_image']
        side_path = self.images_dir / 'side' / row['side_image']
        rear_path = self.images_dir / 'rear' / row['rear_image']
        
        try:
            front_img = Image.open(front_path).convert('RGB')
            side_img = Image.open(side_path).convert('RGB')
            rear_img = Image.open(rear_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading images for {row['cattle_id']}: {e}")
        
        # Apply transforms
        if self.transform:
            if self.augment_views:
                # Apply different augmentations to different views
                front_img = self.transform['front'](front_img)
                side_img = self.transform['side'](side_img)
                rear_img = self.transform['rear'](rear_img)
            else:
                # Apply same transform to all views
                front_img = self.transform(front_img)
                side_img = self.transform(side_img)
                rear_img = self.transform(rear_img)
        
        # Get weight (target)
        weight = torch.tensor(row['weight_kg'], dtype=torch.float32)
        
        # Additional metadata (optional)
        metadata = {
            'cattle_id': row['cattle_id'],
            'age_months': row.get('age_months', -1),
            'breed': row.get('breed', 'Unknown'),
            'gender': row.get('gender', 'Unknown')
        }
        
        return {
            'front_view': front_img,
            'side_view': side_img,
            'rear_view': rear_img,
            'weight': weight,
            'metadata': metadata
        }

# =============================================================================
# DATA TRANSFORMS
# =============================================================================

def get_train_transforms(img_size=384):
    """Get training transforms with augmentation"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size=384):
    """Get validation transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_view_specific_transforms(img_size=384):
    """Get view-specific transforms for different augmentation per view"""
    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    return {
        'front': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] + base_transform),
        
        'side': transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ] + base_transform),
        
        'rear': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(saturation=0.2, hue=0.1),
        ] + base_transform)
    }

# =============================================================================
# DATALOADER SETUP
# =============================================================================

def create_dataloaders(dataset_path, batch_size=16, num_workers=4, img_size=384):
    """Create PyTorch DataLoaders for training, validation, and testing"""
    
    dataset_path = Path(dataset_path)
    
    # Define transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = MultiViewCattleDataset(
        csv_file=dataset_path / 'labels/train_labels.csv',
        images_dir=dataset_path / 'images/train',
        transform=train_transform
    )
    
    val_dataset = MultiViewCattleDataset(
        csv_file=dataset_path / 'labels/val_labels.csv',
        images_dir=dataset_path / 'images/val',
        transform=val_transform
    )
    
    test_dataset = MultiViewCattleDataset(
        csv_file=dataset_path / 'labels/test_labels.csv',
        images_dir=dataset_path / 'images/test',
        transform=val_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# =============================================================================
# DATASET VALIDATION UTILITIES
# =============================================================================

def validate_dataset(dataset_path):
    """Validate dataset structure and integrity"""
    dataset_path = Path(dataset_path)
    issues = []
    
    # Check directory structure
    required_dirs = [
        'images/train/front', 'images/train/side', 'images/train/rear',
        'images/val/front', 'images/val/side', 'images/val/rear',
        'images/test/front', 'images/test/side', 'images/test/rear',
        'labels', 'splits'
    ]
    
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            issues.append(f"Missing directory: {dir_path}")
    
    # Check label files
    label_files = ['train_labels.csv', 'val_labels.csv', 'test_labels.csv']
    for label_file in label_files:
        label_path = dataset_path / 'labels' / label_file
        if not label_path.exists():
            issues.append(f"Missing label file: {label_file}")
        else:
            # Validate CSV structure
            try:
                df = pd.read_csv(label_path)
                required_cols = ['cattle_id', 'weight_kg', 'front_image', 'side_image', 'rear_image']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    issues.append(f"{label_file} missing columns: {missing_cols}")
            except Exception as e:
                issues.append(f"Error reading {label_file}: {e}")
    
    # Check for missing images
    for split in ['train', 'val', 'test']:
        label_path = dataset_path / 'labels' / f'{split}_labels.csv'
        if label_path.exists():
            df = pd.read_csv(label_path)
            for _, row in df.iterrows():
                for view in ['front', 'side', 'rear']:
                    img_path = dataset_path / 'images' / split / view / row[f'{view}_image']
                    if not img_path.exists():
                        issues.append(f"Missing image: {img_path}")
    
    if issues:
        print("Dataset validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset validation passed!")
        return True

def get_dataset_statistics(dataset_path):
    """Get statistics about the dataset"""
    dataset_path = Path(dataset_path)
    
    stats = {}
    for split in ['train', 'val', 'test']:
        label_path = dataset_path / 'labels' / f'{split}_labels.csv'
        if label_path.exists():
            df = pd.read_csv(label_path)
            stats[split] = {
                'num_samples': len(df),
                'weight_stats': {
                    'mean': df['weight_kg'].mean(),
                    'std': df['weight_kg'].std(),
                    'min': df['weight_kg'].min(),
                    'max': df['weight_kg'].max()
                }
            }
            
            if 'breed' in df.columns:
                stats[split]['breed_distribution'] = df['breed'].value_counts().to_dict()
    
    return stats

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of the dataset organization system"""
    
    # Create sample dataset
    print("Creating sample dataset structure...")
    organizer = DatasetOrganizer("cattle_weight_dataset")
    train_df, val_df, test_df = organizer.create_sample_dataset(num_samples=1000)
    
    # Validate dataset
    print("\nValidating dataset...")
    is_valid = validate_dataset("cattle_weight_dataset")
    
    if is_valid:
        # Get dataset statistics
        print("\nDataset statistics:")
        stats = get_dataset_statistics("cattle_weight_dataset")
        for split, split_stats in stats.items():
            print(f"\n{split.upper()} SET:")
            print(f"  Samples: {split_stats['num_samples']}")
            print(f"  Weight - Mean: {split_stats['weight_stats']['mean']:.1f}kg, "
                  f"Std: {split_stats['weight_stats']['std']:.1f}kg")
        
        # Create DataLoaders
        print("\nCreating DataLoaders...")
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                "cattle_weight_dataset", 
                batch_size=8, 
                num_workers=2
            )
            
            print(f"DataLoaders created successfully!")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Test loading a batch
            print("\nTesting batch loading...")
            batch = next(iter(train_loader))
            print(f"Batch keys: {batch.keys()}")
            print(f"Front view shape: {batch['front_view'].shape}")
            print(f"Side view shape: {batch['side_view'].shape}")
            print(f"Rear view shape: {batch['rear_view'].shape}")
            print(f"Weights shape: {batch['weight'].shape}")
            
        except Exception as e:
            print(f"Error creating DataLoaders: {e}")
            print("Note: This is expected if you don't have actual image files")

if __name__ == "__main__":
    main()