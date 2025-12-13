# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Paired MR-CT dataset for image-conditioned training.
Dataset structure:
    dataset_dir/
        mr/
            train/
                image1.jpg
                image2.jpg
                ...
            test/
                image1.jpg
                ...
        ct/
            train/
                image1.jpg
                image2.jpg
                ...
            test/
                image1.jpg
                ...

Note: MR and CT images are paired by sorted order (not by filename).
Images are read in grayscale mode and normalized to (-1, 1).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MRCTDataset(Dataset):
    """
    Paired MR-CT dataset.
    
    Args:
        data_dir: Root directory containing 'mr' and 'ct' subdirectories
        split: 'train' or 'test'
        image_size: Target image size (will be resized if different)
        transform: Optional additional transforms
    """
    def __init__(self, data_dir, split='train', image_size=256, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        
        # Paths to MR and CT directories
        self.mr_dir = os.path.join(data_dir, 'mr', split)
        self.ct_dir = os.path.join(data_dir, 'ct', split)
        
        # Check directories exist
        if not os.path.exists(self.mr_dir):
            raise ValueError(f"MR directory not found: {self.mr_dir}")
        if not os.path.exists(self.ct_dir):
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        
        # Get sorted list of image files
        self.mr_files = sorted([f for f in os.listdir(self.mr_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        
        # Verify matching counts
        if len(self.mr_files) != len(self.ct_files):
            raise ValueError(f"Number of MR images ({len(self.mr_files)}) does not match "
                           f"number of CT images ({len(self.ct_files)})")
        
        print(f"Loaded {len(self.mr_files)} paired MR-CT images from {split} split")
        
        # Default transform: resize, to tensor, normalize to (-1, 1)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # Converts to [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ])
    
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        # Load MR image (grayscale)
        mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
        mr_image = Image.open(mr_path).convert('L')  # Convert to grayscale
        
        # Load CT image (grayscale)
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_image = Image.open(ct_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        mr_tensor = self.transform(mr_image)  # (1, H, W), range [-1, 1]
        ct_tensor = self.transform(ct_image)  # (1, H, W), range [-1, 1]
        
        return {
            'mr': mr_tensor,
            'ct': ct_tensor,
            'mr_path': mr_path,
            'ct_path': ct_path,
        }


class MRCTLatentDataset(Dataset):
    """
    Paired MR-CT dataset with precomputed VAE latents.
    For use with pretrained VAE encoder.
    
    Args:
        data_dir: Root directory containing 'mr' and 'ct' subdirectories
        split: 'train' or 'test'
        image_size: Target image size before VAE encoding
    """
    def __init__(self, data_dir, split='train', image_size=256):
        super().__init__()
        self.base_dataset = MRCTDataset(data_dir, split, image_size)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]


def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize tensor from [-1, 1] to [0, 1].
    
    Args:
        tensor: Tensor normalized with mean and std
        mean: Mean used for normalization (default: 0.5)
        std: Std used for normalization (default: 0.5)
    
    Returns:
        Denormalized tensor in range [0, 1]
    """
    return tensor * std + mean


def tensor_to_image(tensor, denorm=True):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Tensor of shape (1, H, W) or (H, W) in range [-1, 1] or [0, 1]
        denorm: Whether to denormalize from [-1, 1] to [0, 1]
    
    Returns:
        PIL Image
    """
    if denorm:
        tensor = denormalize(tensor)
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # (H, W)
    
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode='L')


def save_sample_images(mr_tensor, ct_tensor, pred_ct_tensor, save_path, denorm=True):
    """
    Save MR, GT CT, and predicted CT images side by side.
    
    Args:
        mr_tensor: MR input tensor (1, H, W)
        ct_tensor: Ground truth CT tensor (1, H, W)
        pred_ct_tensor: Predicted CT tensor (1, H, W)
        save_path: Path to save the combined image
        denorm: Whether to denormalize tensors
    """
    mr_img = tensor_to_image(mr_tensor, denorm)
    ct_img = tensor_to_image(ct_tensor, denorm)
    pred_img = tensor_to_image(pred_ct_tensor, denorm)
    
    # Create side-by-side image
    h, w = mr_img.size[1], mr_img.size[0]
    combined = Image.new('L', (w * 3, h))
    combined.paste(mr_img, (0, 0))
    combined.paste(ct_img, (w, 0))
    combined.paste(pred_img, (w * 2, 0))
    
    combined.save(save_path)


if __name__ == "__main__":
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    
    dataset = MRCTDataset(args.data_dir, args.split, args.image_size)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"MR shape: {sample['mr'].shape}, range: [{sample['mr'].min():.3f}, {sample['mr'].max():.3f}]")
    print(f"CT shape: {sample['ct'].shape}, range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
    print(f"MR path: {sample['mr_path']}")
    print(f"CT path: {sample['ct_path']}")
