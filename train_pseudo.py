"""
PWC-Net Fine-tuning with Proxy-Label Method
Self-supervised training using consecutive frames
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import torch.nn.functional

from models.PWCNet import PWCDCNet

# from pwc_net import pwc_dc_net

#dataset:loading consecutive frames
class ConsecutiveFrameDataset(Dataset):
    """Dataset for loading consecutive image pairs from folders"""
    
    def __init__(self, root_dir, transform=None, frame_gap=1):#pairs of consecutive frames
        """
        Args:
            root_dir: Root directory containing subfolders with images
            transform: Optional transform to be applied on images
            frame_gap: Gap between consecutive frames (1 = adjacent frames)
        """
        self.root_dir = Path(root_dir) #images subfolders
        self.transform = transform #preprocessing pipeline
        self.frame_gap = frame_gap #frame_gap btw pair members
        self.image_pairs = []
        
        # Collect all image pairs from subfolders
        for subfolder in sorted(self.root_dir.iterdir()):
            if subfolder.is_dir():
                images = sorted(list(subfolder.glob('*.png')))
                # Create pairs with frame_gap
                for i in range(len(images) - frame_gap):
                    self.image_pairs.append((images[i], images[i + frame_gap]))
        
        print(f"Found {len(self.image_pairs)} image pairs")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2


class ProxyLabelLoss(nn.Module):#photometric + smoothness
    """
    Proxy-label loss for self-supervised optical flow training
    Combines photometric loss with smoothness regularization
    """
    
    def __init__(self, alpha_photo=1.0, alpha_smooth=0.1):
        super(ProxyLabelLoss, self).__init__()
        self.alpha_photo = alpha_photo
        self.alpha_smooth = alpha_smooth
    
    def photometric_loss(self, img1, img2_warped):
        """Photometric loss (L1 + SSIM)"""
        # L1 loss
        l1_loss = torch.abs(img2_warped - img1).mean()
        
        # SSIM loss
        ssim_loss = self.ssim_loss(img1, img2_warped)
        
        
        return 0.85 * ssim_loss + 0.15 * l1_loss
    
    def ssim_loss(self, x, y, C1=0.01**2, C2=0.03**2):
        """Structural similarity loss"""
        mu_x = nn.functional.avg_pool2d(x, 3, 1, 1)
        mu_y = nn.functional.avg_pool2d(y, 3, 1, 1)
        
        sigma_x = nn.functional.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sigma_y = nn.functional.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sigma_xy = nn.functional.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        
        ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return torch.clamp((1 - ssim) / 2, 0, 1).mean()
    
    def smoothness_loss(self, flow):
        """First-order smoothness regularization"""
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return dx.mean() + dy.mean()
    
    def forward(self, flow, img1, img2):
        """
        Args:
            flow: Predicted optical flow [B, 2, H, W]
            img1: First image [B, 3, H, W]
            img2: Second image [B, 3, H, W]
        """
        # Warp img2 using predicted flow
        img2_warped = self.warp(img2, flow)
        
        # Photometric loss
        photo_loss = self.photometric_loss(img1, img2_warped)
        
        # Smoothness loss
        smooth_loss = self.smoothness_loss(flow)
        
        # Total loss
        total_loss = self.alpha_photo * photo_loss + self.alpha_smooth * smooth_loss
        
        return total_loss, photo_loss, smooth_loss
    
    def warp(self, img, flow):
        """
        img:  [N, C, H, W]
        flow: [N, 2, h, w] pixel flow (dx, dy) in the same units as img pixels
        returns: img warped by flow to size [N, C, H, W]
        """
        N, C, H, W = img.shape

        # 1) If flow is not at image resolution, upsample it (and scale the vectors)
        if flow.shape[-2:] != (H, W):
            h, w = flow.shape[-2:]
            scale_y = H / h
            scale_x = W / w
            flow = torch.nn.functional.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
            flow[:, 0, :, :] *= scale_x  # dx
            flow[:, 1, :, :] *= scale_y  # dy

        # 2) Build base grid in normalized coordinates [-1, 1]
        # grid_sample expects grid layout [N, H, W, 2] with order (x, y)
        # x in [-1,1] left→right, y in [-1,1] top→bottom
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype),
            torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype),
            indexing='ij'
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)  # [N, H, W, 2]

        # 3) Convert pixel flow → normalized flow
        # dx_norm = 2*dx/(W-1), dy_norm = 2*dy/(H-1)
        flow_norm = torch.empty((N, H, W, 2), device=img.device, dtype=img.dtype)
        flow_norm[..., 0] = 2.0 * flow[:, 0, :, :] / max(W - 1, 1)
        flow_norm[..., 1] = 2.0 * flow[:, 1, :, :] / max(H - 1, 1)

        # 4) Create sampling grid and sample
        vgrid = base_grid + flow_norm  # [N, H, W, 2]
        img_warped = torch.nn.functional.grid_sample(img, vgrid, align_corners=True, mode='bilinear', padding_mode='border')
        return img_warped
    

    
def _select_finest_flow(outputs):
    """Return the highest-resolution flow from model outputs."""
    if isinstance(outputs, (list, tuple)):
        # pick the flow with the largest H*W
        flows = [f for f in outputs if isinstance(f, torch.Tensor)]
        flows.sort(key=lambda t: (t.shape[-2] * t.shape[-1]), reverse=True)
        return flows[0]
    return outputs

@torch.no_grad()
def _forward_backward_consistency(model, img1, img2, warp_fn):
    model.eval()
    inp12 = torch.cat([img1, img2], dim=1)
    flow12 = _select_finest_flow(model(inp12))

    inp21 = torch.cat([img2, img1], dim=1)
    flow21 = _select_finest_flow(model(inp21))

    # (Optional) ensure both are at img resolution
    B, _, H, W = img1.shape
    flow12 = _upsample_flow_to(flow12, H, W)
    flow21 = _upsample_flow_to(flow21, H, W)

    flow21_warped = warp_fn(flow21, flow12)
    cycle = flow12 + flow21_warped
    return cycle.abs().mean()

def _upsample_flow_to(flow, H, W):
    """
    Resize flow [B,2,h,w] to (H,W) and scale vectors accordingly.
    """
    b, c, h, w = flow.shape
    if (h, w) == (H, W):
        return flow
    scale_y = H / h
    scale_x = W / w
    flow_up = torch.nn.functional.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
    flow_up[:, 0] *= scale_x  # dx
    flow_up[:, 1] *= scale_y  # dy
    return flow_up

@torch.no_grad()
def _oob_ratio(flow, H, W, device, dtype):
    """
    Fraction of sampling locations that fall outside [-1,1] after adding flow.
    Works at IMAGE resolution by upsampling flow to (H,W) with proper scaling.
    """
    # ensure flow matches image size
    flow = _upsample_flow_to(flow, H, W)  # [B,2,H,W]

    B = flow.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing='ij'
    )
    base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)   # [B,H,W,2]

    flow_norm = torch.empty((B, H, W, 2), device=device, dtype=dtype)
    flow_norm[..., 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
    flow_norm[..., 1] = 2.0 * flow[:, 1] / max(H - 1, 1)

    vgrid = base_grid + flow_norm
    x = vgrid[..., 0]; y = vgrid[..., 1]
    oob = (x < -1) | (x > 1) | (y < -1) | (y > 1)
    return oob.float().mean()




def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_photo_loss = 0.0
    running_smooth_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (img1, img2) in enumerate(pbar):
        img1, img2 = img1.to(device), img2.to(device)
        
        # Concatenate images for PWC-Net input
        input_pair = torch.cat([img1, img2], dim=1)
        
        # Forward pass
        optimizer.zero_grad()
        
        if model.training:
            # During training, model returns multiple flow predictions
            flows = model(input_pair)
            flow2 = flows[0]  # Use finest resolution flow
        else:
            flow2 = model(input_pair)
        
        # Compute loss
        total_loss, photo_loss, smooth_loss = criterion(flow2, img1, img2)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_photo_loss += photo_loss.item()
        running_smooth_loss += smooth_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss.item(),
            'photo': photo_loss.item(),
            'smooth': smooth_loss.item()
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_photo = running_photo_loss / len(dataloader)
    avg_smooth = running_smooth_loss / len(dataloader)
    
    return avg_loss, avg_photo, avg_smooth

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    No-GT validation:
      - photometric & smoothness from ProxyLabelLoss
      - forward-backward cycle consistency
      - out-of-bounds ratio
    Returns a dict of averages.
    """
    model.eval()
    photo_sum = 0.0
    smooth_sum = 0.0
    fb_sum = 0.0
    oob_sum = 0.0
    n = 0

    for img1, img2 in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        # forward pass (img2 -> img1)
        inp = torch.cat([img1, img2], dim=1)
        outputs = model(inp)
        flow = _select_finest_flow(outputs)  # [B,2,H,W]

        # Proxy losses
        total_loss, photo_loss, smooth_loss = criterion(flow, img1, img2)
        photo_sum += float(photo_loss.item())
        smooth_sum += float(smooth_loss.item())

        # Forward–Backward cycle consistency
        fb = _forward_backward_consistency(model, img1, img2, criterion.warp)
        fb_sum += float(fb.item())

        # Out-of-bounds ratio
        B, _, H, W = img1.shape

        # make explicit we use image-sized flow for OOB
        flow_full = _upsample_flow_to(flow, H, W)
        oob = _oob_ratio(flow_full, H, W, device=img1.device, dtype=img1.dtype)


        oob_sum += float(oob.item())

        n += 1

    n = max(n, 1)
    return {
        "val_photo": photo_sum / n,
        "val_smooth": smooth_sum / n,
        "val_fb": fb_sum / n,
        "val_oob": oob_sum / n,
    }


def main():
    # Configuration
    config = {
        'data_dir': '../../../dataset',
        'train_dir': '../../../dataset/train',
        'val_dir': '../../../dataset/val',
        'checkpoint_path': 'pwc_net_chairs.pth.tar',  # Your pretrained model
        'output_dir': './checkpoints_pseudo',
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-7,
        'weight_decay': 4e-4,
        'frame_gap': 1,  # Gap between consecutive frames
        'alpha_photo': 1.0,
        'alpha_smooth': 0.1,
        'save_interval': 5,  # Save checkpoint every N epochs
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((384, 512)),  # Adjust to your needs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and dataloader
    t_dataset = ConsecutiveFrameDataset(
        root_dir=config['train_dir'],
        transform=transform,
        frame_gap=config['frame_gap']
    )
    v_dataset = ConsecutiveFrameDataset(
        root_dir=config['val_dir'],
        transform=transform,
        frame_gap=config['frame_gap']
    )
    print
    train_loader = DataLoader(
        t_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        v_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    
    # Load model
    # Uncomment and adjust based on your model file
    model = PWCDCNet()
    
    # For demonstration, assuming model is defined
    model = PWCDCNet()
    checkpoint = torch.load(config['checkpoint_path'], weights_only='True')
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    model = model.to(device)
    
    # Loss function
    criterion = ProxyLabelLoss(
        alpha_photo=config['alpha_photo'],
        alpha_smooth=config['alpha_smooth']
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        avg_loss, avg_photo, avg_smooth = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Photo={avg_photo:.4f}, Smooth={avg_smooth:.4f}")
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"val_photo={val_metrics['val_photo']:.4f} | "f"val_fb={val_metrics['val_fb']:.4f} | "f"val_oob={val_metrics['val_oob']:.4f}")

        scheduler.step()
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['output_dir'],
                f'pwcnet_proxy_epoch_{epoch}.tar'
            )
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Training configuration loaded. Uncomment model and training loop to start training.")


if __name__ == '__main__':
    main()