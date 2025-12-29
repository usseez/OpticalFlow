"""
PWC-Net Inference on KITTI Dataset
Calculates EPE (End-Point Error) and FL (Flow outlier percentage)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse

from models.PWCNet import PWCDCNet


class KITTIFlowDataset:
    """KITTI Optical Flow Dataset Loader"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to KITTI dataset (e.g., 'data_scene_flow')
            transform: Optional transform for images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # KITTI structure (flat directory)
        self.image_dir = self.root_dir / 'image_2'
        self.flow_dir = self.root_dir / 'flow_occ'
        
        # Collect image pairs
        self.samples = []
        if self.image_dir.exists():
            image_files = sorted(self.image_dir.glob('*_10.png'))
            for img1_path in image_files:
                img2_path = str(img1_path).replace('_10.png', '_11.png')
                
                if Path(img2_path).exists():
                    sample = {'img1': img1_path, 'img2': Path(img2_path)}
                    
                    # Add ground truth flow if available
                    if self.flow_dir.exists():
                        flow_path = self.flow_dir / img1_path.name.replace('_10.png', '_10.png')
                        if flow_path.exists():
                            sample['flow'] = flow_path
                    
                    self.samples.append(sample)
        
        print(f"Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        img1 = Image.open(sample['img1']).convert('RGB')
        img2 = Image.open(sample['img2']).convert('RGB')
        
        # Store original size
        original_size = img1.size  # (W, H)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        result = {
            'img1': img1,
            'img2': img2,
            'original_size': original_size,
            'filename': sample['img1'].name
        }
        
        # Load ground truth flow if available
        if 'flow' in sample:
            flow_gt, valid = self.read_flow(sample['flow'])
            result['flow_gt'] = torch.from_numpy(flow_gt).float()
            result['valid'] = torch.from_numpy(valid).bool()
        
        return result
    
    @staticmethod
    def read_flow(flow_path):
        """Read KITTI .png flow file"""
        flow_img = Image.open(flow_path)
        flow_data = np.array(flow_img).astype(np.float32)
        
        # KITTI flow format: encoded in 16-bit PNG
        flow_u = (flow_data[:, :, 2] - 2**15) / 64.0
        flow_v = (flow_data[:, :, 1] - 2**15) / 64.0
        valid = flow_data[:, :, 0]
        
        flow = np.stack([flow_u, flow_v], axis=-1)
        valid = valid.astype(bool)
        
        return flow, valid


def compute_epe(flow_pred, flow_gt, valid_mask):
    """
    Compute End-Point Error (EPE)
    
    Args:
        flow_pred: Predicted flow [H, W, 2]
        flow_gt: Ground truth flow [H, W, 2]
        valid_mask: Valid pixels [H, W]
    
    Returns:
        EPE value (float)
    """
    if valid_mask.sum() == 0:
        return 0.0
    
    # Compute Euclidean distance
    epe = np.sqrt(np.sum((flow_pred - flow_gt)**2, axis=-1))
    
    # Average over valid pixels
    epe = epe[valid_mask].mean()
    
    return float(epe)


def compute_fl(flow_pred, flow_gt, valid_mask, threshold=3.0, relative_threshold=0.05):
    """
    Compute FL (Percentage of erroneous pixels / outliers)
    A pixel is considered an outlier if EPE > threshold AND EPE > relative_threshold * ||flow_gt||
    
    Args:
        flow_pred: Predicted flow [H, W, 2]
        flow_gt: Ground truth flow [H, W, 2]
        valid_mask: Valid pixels [H, W]
        threshold: Absolute error threshold (default: 3.0 pixels)
        relative_threshold: Relative error threshold (default: 0.05 = 5%)
    
    Returns:
        FL percentage (float)
    """
    if valid_mask.sum() == 0:
        return 0.0
    
    # Compute endpoint error
    epe = np.sqrt(np.sum((flow_pred - flow_gt)**2, axis=-1))
    
    # Compute ground truth magnitude
    flow_gt_mag = np.sqrt(np.sum(flow_gt**2, axis=-1))
    
    # Outlier criteria: EPE > threshold AND EPE > relative_threshold * ||flow_gt||
    outliers = (epe > threshold) & (epe > relative_threshold * flow_gt_mag)
    
    # Calculate percentage over valid pixels
    fl = (outliers[valid_mask].sum() / valid_mask.sum()) * 100.0
    
    return float(fl)


def resize_flow(flow, target_size):
    """
    Resize flow field and scale flow vectors accordingly
    
    Args:
        flow: [2, H, W] or [H, W, 2]
        target_size: (H_new, W_new)
    
    Returns:
        Resized flow [H_new, W_new, 2]
    """
    if flow.ndim == 3 and flow.shape[0] == 2:
        # Convert from [2, H, W] to [H, W, 2]
        flow = flow.transpose(1, 2, 0)
    
    H, W = flow.shape[:2]
    H_new, W_new = target_size
    
    # Resize using PIL
    flow_u = Image.fromarray(flow[:, :, 0]).resize((W_new, H_new), Image.BILINEAR)
    flow_v = Image.fromarray(flow[:, :, 1]).resize((W_new, H_new), Image.BILINEAR)
    
    flow_resized = np.stack([np.array(flow_u), np.array(flow_v)], axis=-1)
    
    # Scale flow vectors
    flow_resized[:, :, 0] *= (W_new / W)
    flow_resized[:, :, 1] *= (H_new / H)
    
    return flow_resized


@torch.no_grad()
def inference(model, dataloader, device, output_dir=None):
    """
    Run inference and compute metrics
    
    Args:
        model: PWC-Net model
        dataloader: DataLoader for KITTI dataset
        device: torch device
        output_dir: Optional directory to save flow predictions
    
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    
    epe_list = []
    fl_list = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Inference')
    
    for batch in pbar:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        
        # Forward pass
        input_pair = torch.cat([img1, img2], dim=1)
        flow_pred = model(input_pair)
        
        # Handle multi-scale output (take finest resolution)
        if isinstance(flow_pred, (list, tuple)):
            flow_pred = flow_pred[0]
        
        # Process each sample in batch
        for i in range(flow_pred.shape[0]):
            flow = flow_pred[i].cpu().numpy()  # [2, H, W]
            original_size = batch['original_size']
            
            # Resize flow to original image size
            H_orig, W_orig = original_size[1][i].item(), original_size[0][i].item()
            flow_resized = resize_flow(flow, (H_orig, W_orig))
            
            # Compute metrics if ground truth is available
            if 'flow_gt' in batch:
                flow_gt = batch['flow_gt'][i].numpy()  # [H, W, 2]
                valid = batch['valid'][i].numpy()  # [H, W]
                
                epe = compute_epe(flow_resized, flow_gt, valid)
                fl = compute_fl(flow_resized, flow_gt, valid)
                print(f"Sample {batch['filename'][i]} - EPE: {epe:.3f}, FL: {fl:.2f}%")
                epe_list.append(epe)
                fl_list.append(fl)
                
                pbar.set_postfix({'EPE': f'{epe:.3f}', 'FL': f'{fl:.2f}%'})
            
            # Save flow if output directory is specified
            if output_dir:
                filename = batch['filename'][i]
                save_flow(flow_resized, output_dir, filename)
    
    # Compute average metrics
    results = {}
    if epe_list:
        results['EPE'] = np.mean(epe_list)
        results['FL'] = np.mean(fl_list)
        results['num_samples'] = len(epe_list)
    
    return results


def save_flow(flow, output_dir, filename):
    """Save flow field as KITTI format .png file"""
    # Convert to KITTI format
    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]
    
    # Encode flow
    flow_u_encoded = (flow_u * 64.0 + 2**15).astype(np.uint16)
    flow_v_encoded = (flow_v * 64.0 + 2**15).astype(np.uint16)
    valid = np.ones_like(flow_u, dtype=np.uint16)
    
    # Stack as RGB image
    flow_img = np.stack([valid, flow_v_encoded, flow_u_encoded], axis=-1)
    
    # Save
    output_path = Path(output_dir) / filename.replace('_10.png', '_10_flow.png')
    Image.fromarray(flow_img.astype(np.uint16)).save(output_path)


def main():
    parser = argparse.ArgumentParser(description='PWC-Net Inference on KITTI')
    parser.add_argument('--kitti_dir', type=str, required=True,
                        help='Path to KITTI dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save flow predictions')
    parser.add_argument('--image_size', type=int, nargs=2, default=[384, 1280],
                        help='Input image size (H W)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize(tuple(args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = KITTIFlowDataset(
        root_dir=args.kitti_dir,
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = PWCDCNet()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("Running inference...")
    results = inference(model, dataloader, device, args.output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("KITTI Evaluation Results")
    print("="*50)
    if results:
        print(f"Number of samples: {results['num_samples']}")
        print(f"Average EPE: {results['EPE']:.3f} pixels")
        print(f"Average FL: {results['FL']:.2f}%")
    else:
        print("No ground truth available for evaluation")
    print("="*50)
    
    if args.output_dir:
        print(f"\nFlow predictions saved to: {args.output_dir}")


if __name__ == '__main__':
    main()