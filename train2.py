"""
Fine-tuning script for PWC-Net on KITTI dataset
Transfer learning from Chairs pretrained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import datetime 
import os
from models.PWCNet import PWCDCNet
from data_processing import (
    KittiDataset, 
    KittiAugmentationPipeline,
    create_kitti_loaders
)


def load_pretrained_model(model, checkpoint_path, device):
    """
    Load pretrained model from checkpoint
    Handles both .pth and .pth.tar formats
    """
    if checkpoint_path.endswith('.tar'):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained model from {checkpoint_path}")
    return model


def warp_image(im2, flow):
    """
    Warp im2 according to flow
    """
    B, C, H, W = im2.size()
    xx = torch.arange(0, W, device=im2.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=im2.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    # print("*********************************************grid and flow shape", grid.shape, flow.shape)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    output = nn.functional.grid_sample(im2, vgrid, align_corners=True)
    return output


def photometric_loss(im1, im2_warp, mask=None):
    """
    Photometric loss (L1)
    mask: occlusion mask, areas with mask=0 are invalid
    """
    loss = torch.abs(im1 - im2_warp)
    
    if mask is not None:
        loss = loss * mask.unsqueeze(1)
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss


def smoothness_loss(flow, imgs=None):
    """
    Smoothness loss with edge-aware weighting
    Uses image gradients to preserve edges
    """
    # Compute flow gradients
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    
    if imgs is not None:
        # Edge-aware: weight by image gradients
        img_dx = torch.mean(torch.abs(imgs[:, :3, :, :-1] - imgs[:, :3, :, 1:]), dim=1, keepdim=True)
        img_dy = torch.mean(torch.abs(imgs[:, :3, :-1, :] - imgs[:, :3, 1:, :]), dim=1, keepdim=True)
        
        dx = dx * torch.exp(-img_dx)
        dy = dy * torch.exp(-img_dy)
    
    return dx.mean() + dy.mean()


def compute_epe(flow_pred, flow_gt, mask=None):
    """
    Compute End-Point Error
    """
    epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1))
    
    if mask is not None:
        epe = epe * mask
        epe = epe.sum() / (mask.sum() + 1e-8)
    else:
        epe = epe.mean()
    
    return epe

class MaskedCharbonnier(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, gt, mask):  # [B,2,H,W], [B,2,H,W], [B,H,W]
        valid = (mask > 0.5).float().unsqueeze(1)
        epe = torch.sqrt(((pred - gt)**2).sum(dim=1, keepdim=True) + self.eps**2)
        denom = valid.sum().clamp(min=1.0)
        return (epe * valid).sum() / denom

def supervised_multiscale_loss(flow_preds, images, flows_gt, masks, w=None,
                               lambda_photo=0.0, lambda_smooth=0.0):
    """
    flow_preds: list of predicted flows at different scales (fine->coarse or vice versa)
    We supervise each prediction at its **own** spatial size.
    """
    if not isinstance(flow_preds, (list, tuple)):
        flow_preds = [flow_preds]
    if w is None:
        w = [0.32, 0.08, 0.02, 0.01, 0.005]

    charbonnier = MaskedCharbonnier()
    B, _, H, W = flows_gt.shape
    im1, im2 = images[:, :3], images[:, 3:]

    total = 0.0
    for i, pred in enumerate(flow_preds):
        h, w_ = pred.shape[-2:]
        # Downsample GT **to pred size**
        gt_s = nn.functional.interpolate(flows_gt, size=(h, w_), mode='bilinear', align_corners=False)
        mask_s = nn.functional.interpolate(masks.unsqueeze(1).float(), size=(h, w_), mode='nearest').squeeze(1)

        # Scale GT vectors when resizing (preserve pixel units)
        scale_x = W / float(w_)
        scale_y = H / float(h)
        gt_s[:, 0] /= scale_x
        gt_s[:, 1] /= scale_y

        lvl_loss = charbonnier(pred, gt_s, mask_s)

        # tiny regularizers (optional)
        if lambda_photo > 0.0 or lambda_smooth > 0.0:
            im1_s = nn.functional.interpolate(im1, size=(h, w_), mode='bilinear', align_corners=False)
            im2_s = nn.functional.interpolate(im2, size=(h, w_), mode='bilinear', align_corners=False)
            im2_w = warp_image(im2_s, pred)
            photo = photometric_loss(im1_s, im2_w, mask_s) if lambda_photo > 0.0 else 0.0
            smooth = smoothness_loss(pred, im1_s) if lambda_smooth > 0.0 else 0.0
            lvl_loss = lvl_loss + lambda_photo * photo + lambda_smooth * smooth

        # pick weight safely even if fewer preds than weights
        wi = w[i] if i < len(w) else w[-1]
        total += wi * lvl_loss

    return total

def train_epoch(model, train_loader, optimizer, device, epoch, lambda_smooth=0.01):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
    
    for batch_idx, (images, flows, masks) in enumerate(progress_bar):
        images = images.to(device)
        flows = flows.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        flow_predictions = model(images)
        
        flow_predictions = flow_predictions if isinstance(flow_predictions, (list, tuple)) else [flow_predictions]
        loss = supervised_multiscale_loss(flow_predictions, images, flows, masks, w=None)  # or pass your list

        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def upsample_flow_to(flow, H, W):
    """
    flow: [B,2,h,w]  -> upsampled: [B,2,H,W]
    Scales vector magnitudes to remain in pixels.
    """
    b, c, h, w = flow.shape
    sx = W / float(w)
    sy = H / float(h)
    up = nn.functional.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    up[:, 0, :, :] *= sx  # u (x)
    up[:, 1, :, :] *= sy  # v (y)
    return up

def validate(model, val_loader, device, loss_weights=None):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0.0
    total_epe = 0.0
    num_batches = 0
    progress_bar = tqdm(val_loader, desc="Validation")
    # torch.set_grad_enabled(False)
    with torch.no_grad():
        for images, flows, masks in progress_bar:
            images = images.to(device)
            flows = flows.to(device)
            masks = masks.to(device)
            
            # Forward pass (returns only flow2 in eval mode)
            # out = 
            out = model(images)
            flow_preds = out if isinstance(out, (list, tuple)) else [out]

            
            # Warp and compute photometric loss
            loss = supervised_multiscale_loss(flow_preds, images, flows, masks, w = loss_weights)

            B, _, H, W = images.shape
            finest = flow_preds[0]
            flow_pred_full = upsample_flow_to(finest, H, W)
            epe = compute_epe(flow_pred_full, flows, masks)
            
            total_loss += loss.item()
            total_epe += epe.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item(), 'epe': epe.item()})
    
    avg_loss = total_loss / num_batches
    avg_epe = total_epe / num_batches
    return avg_loss, avg_epe



def plot_metrics(train_losses, val_losses, val_epes, save_path='training_metrics.png'):
    """
    Plot training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('PWC-Net Loss on KITTI', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # EPE plot
    ax2.plot(val_epes, label='Validation EPE', marker='s', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('End-Point Error (px)', fontsize=12)
    ax2.set_title('Optical Flow Accuracy (EPE)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune PWC-Net on KITTI')
    parser.add_argument('--kitti_root', type=str, required=True, 
                       help='Path to KITTI dataset root')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained model (pwc_net_chairs.pth.tar)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (smaller for KITTI due to resolution)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (smaller for fine-tuning)')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                       help='Smoothness loss weight')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[320, 896],
                       help='Crop size (height width) - KITTI is 370x1226, use smaller value')
    parser.add_argument('--save_dir', type=str, default='checkpoints_kitti',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Setup
    Path(args.save_dir).mkdir(exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Fine-tuning on KITTI dataset")
    
    # Load pretrained model
    print("Loading pretrained PWC-Net...")
    model = PWCDCNet()
    model = load_pretrained_model(model, args.pretrained, device)
    model = model.to(device)
    
    # Load KITTI dataset
    print("Loading KITTI dataset...")
    train_dataset = KittiDataset(
        args.kitti_root,
        split='training',
        use_occluded=False
    )
    valid_dataset = KittiDataset(
        args.kitti_root,
        split='validation',
        use_occluded=False
    )


    
    # Apply augmentation on-the-fly
    augmentation = KittiAugmentationPipeline(
        crop_size=tuple(args.crop_size),
        augment=True
    )

    def collate_fn(batch):
        items = [augmentation(item) for item in batch]
        images = torch.stack([item['images'] for item in items])
        flows = torch.stack([item['flow'] for item in items])
        masks = torch.stack([item['mask'] for item in items])
        return images, flows, masks
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # For validation, you can use a separate validation split if available
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    # For now, we'll use the test set (if available) or create a split
    # val_loader = train_loader  # Replace with actual val set if available
    
    # Optimizer (lower learning rate for fine-tuning)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,           # try 1e-4 for a few epochs then 5e-5
        betas=(0.9, 0.999),
        weight_decay=1e-6
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3, verbose=True
    )
    # Training loop
    train_losses = []
    val_losses = []
    val_epes = []
    best_epe = float('inf')
    
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}, Batch size: {args.batch_size}")

    # set save pth path
    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    os.mkdir(os.path.join(args.save_dir, suffix))
    
    loss_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.lambda_smooth)
        val_loss, val_epe = validate(model, val_loader, device, loss_weights=loss_weights)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_epes.append(val_epe)
        
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val EPE:    {val_epe:.4f} px")
        
        scheduler.step(val_epe)
        
        # Save best model (based on EPE)
        if val_epe < best_epe:
            best_epe = val_epe
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_epe': val_epe,
                'val_loss': val_loss
            }
            save_path = os.path.join(args.save_dir, f"{suffix}/best_model{suffix}.pth")
            torch.save(checkpoint, save_path)
            print(f"  Best model saved! EPE: {val_epe:.4f}")
        
        # Save checkpoint every 10 epochs
        # if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        ckpt_path = os.path.join(args.save_dir, f"{suffix}/checkpoint_epoch_{epoch+1:04d}.pth")
        # save_path = Path(args.save_dir) / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, ckpt_path)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, val_epes, 
                ckpt_path / 'training_metrics.png')
    
    # Save final model
    torch.save(model.state_dict(), Path(args.save_dir) / 'final_model.pth')
    print(f"\nFine-tuning completed!")
    print(f"Best validation EPE: {best_epe:.4f} px")
    print(f"Models saved to {args.save_dir}")


if __name__ == '__main__':
    main()