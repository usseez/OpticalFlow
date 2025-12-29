# train.py
# Fine-tuning PWCDCNet on KITTI with masked full-res loss:
# - batch size 4
# - compare upsampled quarter-res prediction with full-res GT
# - exclude invalid pixels via KITTI mask

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime 
from models.PWCNet import PWCDCNet
from data_processing import KittiFlowDataset, upsample_flow_to


# --------------------------
# Repro & speed
# --------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True  # faster on fixed input size (320x896)


# --------------------------
# Masked Charbonnier EPE at full resolution
# --------------------------
class MaskedCharbonnier(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred_flow_full, gt_flow_full, valid_mask):
        """
        pred_flow_full: [B,2,H,W]
        gt_flow_full  : [B,2,H,W]
        valid_mask    : [B,1,H,W] (0/1)
        """
        # EPE per-pixel
        epe = torch.sqrt(torch.sum((pred_flow_full - gt_flow_full) ** 2, dim=1, keepdim=True) + self.eps ** 2)  # [B,1,H,W]
        # Mask invalid pixels
        valid = (valid_mask > 0.5).float()
        num = valid.sum().clamp(min=1.0)
        loss = (epe * valid).sum() / num
        return loss


# --------------------------
# Train loop
# --------------------------
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, log_every=50):
    model.train()
    running = 0.0
    t0 = time.time()

    for step, (x, flow_gt, valid) in enumerate(loader):
        x = x.to(device, non_blocking=True)              # [B,6,320,896]
        flow_gt = flow_gt.to(device, non_blocking=True)  # [B,2,320,896]
        valid = valid.to(device, non_blocking=True)      # [B,1,320,896]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # PWC-DC in train() returns (flow2, flow3, flow4, flow5, flow6)
            flow2, *_ = model(x)  # quarter-res prediction [B,2,80,224]
            # Upsample to full resolution and scale vectors
            H, W = flow_gt.shape[2], flow_gt.shape[3]
            flow_pred_full = upsample_flow_to(flow2, H, W)  # [B,2,320,896]
            loss = loss_fn(flow_pred_full, flow_gt, valid)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item()
        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  step {step+1:5d}/{len(loader):5d} | loss {running/log_every:.4f} | {dt:.1f}s")
            running = 0.0
            t0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default = '../Dataset/kitti_flow2015/training', help='KITTI root (or parent of image_2 & flow_occ)')
    ap.add_argument('--list_txt', default=None, help='Optional list file: <img1> <img2> <flow_png> per line')
    ap.add_argument('--auto_scan', action='store_true', help='Auto-scan KITTI-style tree')
    ap.add_argument('--batch_size', type=int, default=4)          # paper: 4
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save_dir', default='checkpoints_kitti')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--amp', action='store_true', help='enable mixed precision')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pretrained', default=None,
                help='Path to pwc_net_chairs.tar (or other pretrained weights). Ignored if --resume is set.')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    # Dataset / Loader (crops to exactly 320x896, reduced aug)
    ds = KittiFlowDataset(
        root=args.data_root,
        list_txt=args.list_txt,
        auto_scan=True,
        crop_hw=(320, 896),
        apply_aug=True
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    # Model
    model = PWCDCNet().to(device)
    model.train()

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = MaskedCharbonnier()
    scaler = torch.amp.GradScaler() if (args.amp and device == 'cuda') else None

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        start_epoch = 0
        if args.pretrained is not None:
            print(f"Loading pretrained weights from: {args.pretrained}")
            ckpt = torch.load(args.pretrained, map_location='cpu')
            # Accept common layouts: state_dict, model, or flat state dict
            state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
            # Strip DP prefix if present
            state = {k.replace('module.', ''): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            print("Loaded pretrained. Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            # (Optional) lower LR for fine-tune
            for g in optimizer.param_groups:
                g['lr'] = args.lr


    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    os.mkdir(os.path.join(args.save_dir, suffix))
    # Train
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, dl, optimizer, scaler, loss_fn, device)

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"{suffix}/pwc_epoch_{epoch+1:04d}.pth")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'args': vars(args)
        }, ckpt_path)
        print(f"Saved {ckpt_path}")


if __name__ == '__main__':
    main()
