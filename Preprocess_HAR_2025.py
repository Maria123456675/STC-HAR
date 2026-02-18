import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SkeletonDataset(Dataset):
    """Dataset loader for skeleton-based action recognition"""
    def __init__(self, data_path, labels_path, num_frames=300, transform=None):
        self.data_path = data_path
        self.num_frames = num_frames
        self.transform = transform
        
        # Load data and labels
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        self.samples = self._preprocess_data()
    
    def _preprocess_data(self):
        samples = []
        for video_id, skeleton_data in self.data.items():
            # Convert to numpy array
            skeleton_array = np.array(skeleton_data)
            
            # Normalize skeleton coordinates
            skeleton_array = self._normalize_skeleton(skeleton_array)
            
            # Temporal sampling/padding
            skeleton_array = self._temporal_sampling(skeleton_array)
            
            label = self.labels[video_id]
            samples.append((skeleton_array, label))
        
        return samples
    
    def _normalize_skeleton(self, skeleton):
        """Normalize skeleton coordinates"""
        # Center around hip joint (joint 0)
        center = skeleton[:, 0:1, :]  # Hip joint
        skeleton = skeleton - center
        
        # Scale normalization
        max_val = np.max(np.abs(skeleton))
        if max_val > 0:
            skeleton = skeleton / max_val
            
        return skeleton
    
    def _temporal_sampling(self, skeleton):
        """Temporal sampling to fixed length"""
        T, V, C = skeleton.shape
        
        if T > self.num_frames:
            # Downsample
            indices = np.linspace(0, T-1, self.num_frames).astype(int)
            sampled_skeleton = skeleton[indices]
        else:
            # Pad with zeros
            sampled_skeleton = np.zeros((self.num_frames, V, C))
            sampled_skeleton[:T] = skeleton
            # Loop padding for shorter sequences
            if T < self.num_frames:
                for i in range(T, self.num_frames):
                    sampled_skeleton[i] = sampled_skeleton[i % T]
        
        return sampled_skeleton
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        skeleton, label = self.samples[idx]
        
        # Data augmentation
        if self.transform:
            skeleton = self.transform(skeleton)
        
        # Convert to tensor and reshape for model
        skeleton_tensor = torch.FloatTensor(skeleton)  # (T, V, C)
        skeleton_tensor = skeleton_tensor.permute(2, 0, 1)  # (C, T, V)
        
        return skeleton_tensor, label

class SkeletonTransform:
    """Data augmentation for skeleton data"""
    def __init__(self, apply_augmentation=True):
        self.apply_augmentation = apply_augmentation
    
    def __call__(self, skeleton):
        if not self.apply_augmentation:
            return skeleton
        
        # Random rotation
        if np.random.random() > 0.5:
            skeleton = self._random_rotation(skeleton)
        
        # Random scaling
        if np.random.random() > 0.5:
            skeleton = self._random_scaling(skeleton)
        
        # Random temporal shift
        if np.random.random() > 0.5:
            skeleton = self._temporal_shift(skeleton)
        
        return skeleton
    
    def _random_rotation(self, skeleton, max_angle=30):
        """Random rotation around Y-axis"""
        angle = np.random.uniform(-max_angle, max_angle)
        theta = np.radians(angle)
        
        rot_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        return np.dot(skeleton, rot_matrix.T)
    
    def _random_scaling(self, skeleton, scale_range=(0.8, 1.2)):
        """Random scaling"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return skeleton * scale
    
    def _temporal_shift(self, skeleton, max_shift=10):
        """Random temporal shifting"""
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            shifted = np.zeros_like(skeleton)
            shifted[shift:] = skeleton[:-shift]
        elif shift < 0:
            shifted = np.zeros_like(skeleton)
            shifted[:shift] = skeleton[-shift:]
        else:
            shifted = skeleton
        return shifted