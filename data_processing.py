"""
Data processing script for KITTI dataset for PWC-Net training
Following the PWC-Net paper's preprocessing strategy
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def read_flow(flow_file):
    """
    Read .flo file (KITTI optical flow format)
    """
    with open(flow_file, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = data.reshape((h, w, 2))
    return flow


def read_kitti_png_file(file_path):
    """
    Read KITTI flow from PNG file
    KITTI encodes flow as (flow_u, flow_v, mask) in 16-bit PNG
    """
    flow_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    flow_u = flow_data[:, :, 2].astype(np.float32)
    flow_v = flow_data[:, :, 1].astype(np.float32)
    mask = flow_data[:, :, 0].astype(np.float32)
    
    # Decode flow (subtract 32768 and divide by 64)
    flow_u = (flow_u - 32768) / 64.0
    flow_v = (flow_v - 32768) / 64.0
    
    # Create mask (1 where valid)
    mask = (mask > 0).astype(np.float32)
    
    flow = np.stack([flow_u, flow_v], axis=0)
    return flow, mask


def read_image(img_file):
    """
    Read image file
    """
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class KittiDataset(Dataset):
    """
    KITTI dataset loader for PWC-Net
    Dataset structure:
    KITTI/
    ├── training/
    │   ├── image_2/
    │   ├── image_3/
    │   ├── flow_occ/
    │   └── flow_noc/
    └── testing/
        ├── image_2/
        └── image_3/
    """
    
    def __init__(self, root_dir, split='training', use_occluded=False, max_samples=None):
        """
        Args:
            root_dir: KITTI dataset root directory
            split: 'training' or 'testing'
            use_occluded: Use occlusion flow or non-occlusion flow
            max_samples: Limit dataset size (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.use_occluded = use_occluded
        
        # Get flow subdirectory
        flow_dir = 'flow_occ' if use_occluded else 'flow_noc'
        
        self.img0_dir = self.root_dir / split / 'image_2'
        self.img1_dir = self.root_dir / split / 'image_3'
        self.flow_dir = self.root_dir / split / flow_dir
        
        # Get list of flow files
        self.flow_files = sorted(self.flow_dir.glob('*.png'))
        
        if max_samples is not None:
            self.flow_files = self.flow_files[:max_samples]
        
        print(f"Loaded {len(self.flow_files)} samples from {split} set")
        
        if len(self.flow_files) == 0:
            raise ValueError(f"No flow files found in {self.flow_dir}")
    
    def __len__(self):
        return len(self.flow_files)
    
    def __getitem__(self, idx):
        flow_file = self.flow_files[idx]
        base_name = flow_file.stem
        
        # Read images
        img0_path = self.img0_dir / f"{base_name}.png"
        img1_path = self.img1_dir / f"{base_name}.png"
        
        img0 = read_image(str(img0_path))
        img1 = read_image(str(img1_path))
        
        # Read flow and mask
        flow, mask = read_kitti_png_file(str(flow_file))
        
        # Stack images [6, H, W]
        imgs = np.concatenate([img0, img1], axis=2)   # [H, W, 6]
        # imgs = np.transpose(imgs, (2, 0, 1))          # [6, H, W]
        
        return {
            'images': imgs,
            'flow': flow,
            'mask': mask
        }


class KittiAugmentationPipeline:
    """
    Data augmentation following PWC-Net paper
    Augmentations:
    - Crop random patches
    - Horizontal flip (50% probability)
    - Rotate (-17 to 17 degrees)
    - Translate (small random shifts)
    - Brightness/contrast changes
    - Gaussian blur
    """
    
    def __init__(self, crop_size=(368, 768), augment=True):
        """
        Args:
            crop_size: (height, width) for cropping
            augment: Whether to apply augmentations
        """
        self.crop_size = crop_size
        self.augment = augment
    
    def __call__(self, sample):
        imgs = sample['images']  # [H, W, 6]
        flow = sample['flow']    # [H, W, 3]
        mask = sample['mask']    # [H, W]

        # Convert to [H, W, C] format for augmentation
        # imgs = np.transpose(imgs, (2, 0, 1))  # [H, W, 6]
        flow = np.transpose(flow, (1, 2, 0))  # [H, W, 2]
        
        H, W = imgs.shape[:2]

        # Random crop
        # if H > self.crop_size[0] or W > self.crop_size[1]:
        h_start = np.random.randint(0, H - self.crop_size[0] + 1) if H >= self.crop_size[0] else 0
        w_start = np.random.randint(0, W - self.crop_size[1] + 1) if W >= self.crop_size[1] else 0
        # h_start = 0
        # w_start = 0

        imgs = imgs[h_start:h_start+self.crop_size[0], 
                    w_start:w_start+self.crop_size[1]]
        flow = flow[h_start:h_start+self.crop_size[0], 
                    w_start:w_start+self.crop_size[1]]
        mask = mask[h_start:h_start+self.crop_size[0], 
                    w_start:w_start+self.crop_size[1]]

        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                imgs = np.ascontiguousarray(imgs[:, ::-1])
                flow = np.ascontiguousarray(flow[:, ::-1])
                flow[:, :, 0] *= -1  # Flip u component
                mask = np.ascontiguousarray(mask[:, ::-1])

            # Rotation (-17 to 17 degrees)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-17, 17)
                h, w = imgs.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation
                imgs = cv2.warpAffine(imgs, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                flow = cv2.warpAffine(flow, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                
                # Rotate flow vectors
                theta = np.radians(angle)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                u, v = flow[:, :, 0], flow[:, :, 1]
                flow[:, :, 0] = u * cos_t - v * sin_t
                flow[:, :, 1] = u * sin_t + v * cos_t

            # Random translation
            if np.random.rand() < 0.5:
                # imgs = imgs[:, ::-1].copy()
                # flow = flow[:, ::-1].copy()
                # mask = mask[:, ::-1].copy()
                # flow[:, :, 0] *= -1  # u 성분 부호 반전

                max_shift = 10
                tx = np.random.randint(-max_shift, max_shift + 1)
                ty = np.random.randint(-max_shift, max_shift + 1)
                
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                h, w = imgs.shape[:2]
                
                imgs = cv2.warpAffine(imgs, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                flow = cv2.warpAffine(flow, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
                
                # # Update flow for translation
                # flow[:, :, 0] += tx
                # flow[:, :, 1] += ty
            
            # Brightness/contrast
            if np.random.rand() < 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                contrast_factor = np.random.uniform(0.8, 1.2)
                
                imgs_rgb = imgs[:, :, :3]
                imgs_rgb = brightness_factor * contrast_factor * (
                    imgs_rgb - 127.5) + 127.5
                imgs_rgb = np.clip(imgs_rgb, 0, 255)
                
                imgs_rgb2 = imgs[:, :, 3:]
                imgs_rgb2 = brightness_factor * contrast_factor * (
                    imgs_rgb2 - 127.5) + 127.5
                imgs_rgb2 = np.clip(imgs_rgb2, 0, 255)
                
                imgs = np.concatenate([imgs_rgb, imgs_rgb2], axis=-1)
            
            # Gaussian blur
            if np.random.rand() < 0.5:
                sigma = np.random.uniform(0.5, 1.5)
                kernel_size = int(np.ceil(4 * sigma))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                imgs[:, :, :3] = cv2.GaussianBlur(
                    imgs[:, :, :3].astype(np.uint8), 
                    (kernel_size, kernel_size), sigma)
                imgs[:, :, 3:] = cv2.GaussianBlur(
                    imgs[:, :, 3:].astype(np.uint8), 
                    (kernel_size, kernel_size), sigma)
        
        # Convert back to [C, H, W]
        imgs = np.transpose(imgs, (2, 0, 1))
        flow = np.transpose(flow, (2, 0, 1))
        # print("\n***********************************final***********************************", imgs.shape, flow.shape, mask.shape)

        # Normalize images to [0, 1]
        imgs = imgs / 255.0
        return {
            'images': torch.from_numpy(imgs).float(),
            'flow': torch.from_numpy(flow).float(),
            'mask': torch.from_numpy(mask).float()
        }


def create_kitti_loaders(kitti_root, batch_size=8, num_workers=4, 
                        crop_size=(368, 768), augment=True, max_samples=None):
    """
    Create KITTI train/val data loaders
    
    Args:
        kitti_root: Root directory of KITTI dataset
        batch_size: Batch size
        num_workers: Number of workers
        crop_size: Crop size for augmentation
        augment: Whether to apply augmentations
        max_samples: Limit dataset size
    
    Returns:
        train_loader, val_loader
    """
    
    # For KITTI, we typically use all training data
    # You can split manually if needed
    train_dataset = KittiDataset(
        kitti_root, 
        split='training',
        use_occluded=False,
        max_samples=max_samples
    )
    
    augmentation = KittiAugmentationPipeline(
        crop_size=crop_size, 
        augment=augment
    )
    
    def collate_fn(batch):
        out = [augmentation(sample) for sample in batch]
        images = torch.stack([o['images'] for o in out])
        flows  = torch.stack([o['flow']   for o in out])
        masks  = torch.stack([o['mask']   for o in out])
        return images, flows, masks
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader


if __name__ == '__main__':
    # Test data loading
    kitti_root = '../Dataset/kitti_flow2015/training'
    
    dataset = KittiDataset(kitti_root, split='training', max_samples=10)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Images shape: {sample['images'].shape}")
    print(f"Flow shape: {sample['flow'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    
    # Test augmentation
    aug_pipeline = KittiAugmentationPipeline(crop_size=(368, 768), augment=True)
    augmented = aug_pipeline(sample)
    print(f"\nAfter augmentation:")
    print(f"Images shape: {augmented['images'].shape}")
    print(f"Flow shape: {augmented['flow'].shape}")