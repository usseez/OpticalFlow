import os
import sys
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
# -------------------------
#  Model
# -------------------------
from models.PWCNet import PWCDCNet


# =========================
# Utilities
# =========================

def load_flow_kitti_png(path):
    """
    Read KITTI 16-bit PNG optical flow with OpenCV (keeps 16-bit).
    Encoding:
      u = (R - 2^15) / 64
      v = (G - 2^15) / 64
      valid = (B != 0)
    Returns: flow (H,W,2) float32, valid (H,W) bool
    """
    arr_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # [H,W,3], uint16
    if arr_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    if arr_bgr.ndim != 3 or arr_bgr.shape[2] != 3:
        raise RuntimeError(f"Unexpected PNG shape for {path}: {arr_bgr.shape}")
    if arr_bgr.dtype != np.uint16:
        raise RuntimeError(f"{path} PNG dtype is {arr_bgr.dtype} (expected uint16)")

    # BGR -> RGB
    arr = arr_bgr[..., ::-1]  # uint16

    R = arr[..., 0].astype(np.float32)
    G = arr[..., 1].astype(np.float32)
    B = arr[..., 2].astype(np.uint16)

    u = (R - 32768.0) / 64.0
    v = (G - 32768.0) / 64.0
    valid = (B != 0)

    flow = np.stack([u, v], axis=-1).astype(np.float32)
    return flow, valid
def pad_to_64(x):
    """
    x: [B,C,H,W]
    pad to multiples of 64 using replicate padding
    returns: x_pad, (pad_h, pad_w)
    """
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (64 - (H % 64)) % 64
    pad_w = (64 - (W % 64)) % 64
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_pad, pad_h, pad_w


def unpad(x, pad_h, pad_w):
    """remove bottom/right padding"""
    if pad_h == 0 and pad_w == 0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    return x[..., : H - pad_h, : W - pad_w]


def select_finest_flow(outputs):
    """Return highest-res flow regardless of list order."""
    if isinstance(outputs, (list, tuple)):
        flows = [f for f in outputs if isinstance(f, torch.Tensor)]
        flows.sort(key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        return flows[0]
    return outputs


def flow_resize(flow, new_h, new_w):
    """Resize flow [B,2,H,W] to (new_h,new_w) and scale vectors accordingly."""
    B, C, H, W = flow.shape
    if (H, W) == (new_h, new_w):
        return flow
    flow_rs = F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)
    flow_rs[:, 0] *= new_w / W
    flow_rs[:, 1] *= new_h / H
    return flow_rs


def epe_metric(flow_pred, flow_gt, valid):
    """
    flow_pred, flow_gt: [H,W,2] float32
    valid: [H,W] boolean
    Returns average EPE over valid pixels.
    """
    du = flow_pred[..., 0] - flow_gt[..., 0]
    dv = flow_pred[..., 1] - flow_gt[..., 1]
    epe = np.sqrt(du * du + dv * dv)
    if valid is not None:
        epe = epe[valid]
    if epe.size == 0:
        return np.nan
    return float(np.mean(epe))

def fl_all_metric(flow_pred, flow_gt, valid):
    """
    KITTI 'Fl-all' outlier rate:
      pixel is outlier if EPE > max(3 px, 0.05 * |flow_gt|)
    Returns percentage (0..100).
    """
    du = flow_pred[..., 0] - flow_gt[..., 0]
    dv = flow_pred[..., 1] - flow_gt[..., 1]
    epe = np.sqrt(du * du + dv * dv)
    mag = np.sqrt(flow_gt[..., 0] ** 2 + flow_gt[..., 1] ** 2)
    thresh = np.maximum(3.0, 0.05 * mag)
    outlier = epe > thresh
    if valid is not None:
        outlier = outlier & valid
        denom = np.count_nonzero(valid)
    else:
        denom = outlier.size
    if denom == 0:
        return np.nan
    return 100.0 * float(np.count_nonzero(outlier)) / float(denom)


# =========================
# KITTI Dataset
# =========================
class KITTIPairsFlow(Dataset):
    """
    Expects folder structure:
      root/
        image_2/  (or 'colored_0' for 2012)  contains frame_10.png, frame_11.png pairs per sample
        flow_occ/ or flow_noc/ or flow/     contains corresponding flow_noc/out files (.png)

    You can also pass explicit subfolders via args.
    """

    def __init__(self,
                 root,
                 images_dir=None,
                 flow_dir=None,
                 kitti_year=2015,
                 normalize=True):
        self.root = Path(root)
        self.kitti_year = kitti_year
        # Common defaults
        if images_dir is None:
            images_dir = "image_2" if kitti_year == 2015 else "colored_0"
        if flow_dir is None:
            # 2015: training/flow_occ or flow_noc; 2012: flow_noc
            flow_dir = "flow_occ" if kitti_year == 2015 else "flow_noc"

        self.img_dir = self.root / images_dir
        self.flow_dir = self.root / flow_dir

        # Pair detection: frame_10/frame_11 pattern
        lefts = sorted(self.img_dir.glob("*_10.png"))
        rights = [self.img_dir / p.name.replace("_10.png", "_11.png") for p in lefts]
        flows = [self.flow_dir / p.name.replace("_10.png", "_10.png") for p in lefts]  # same stem as 10

        # Filter missing
        pairs = []
        for l, r, f in zip(lefts, rights, flows):
            if l.exists() and r.exists() and f.exists():
                pairs.append((l, r, f))
        self.samples = pairs

        # Transform: to tensor + normalize like training
        tfs = [T.ToTensor()]
        if normalize:
            tfs.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]))
        self.transform = T.Compose(tfs)

        print(f"[KITTI{self.kitti_year}] Found {len(self.samples)} pairs at {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_p, img2_p, flow_p = self.samples[idx]

        img1 = Image.open(img1_p).convert("RGB")
        img2 = Image.open(img2_p).convert("RGB")

        flow, valid = load_flow_kitti_png(flow_p)  # (H,W,2), (H,W)

        # tensors
        img1_t = self.transform(img1)  # [3,H,W]
        img2_t = self.transform(img2)

        H, W = flow.shape[0], flow.shape[1]
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).float()  # [2,H,W]
        valid_t = torch.from_numpy(valid.astype(np.uint8))        # [H,W]

        return img1_t, img2_t, flow_t, valid_t, str(img1_p.name).replace("_10.png", "")


# =========================
# Inference / Evaluation
# =========================
@torch.no_grad()
def model_infer(model, img1, img2):
    """
    Inference with 64-multiple padding, select finest flow, unpad,
    THEN upsample flow to original (H,W) if needed.
    """
    model.eval()
    B, C, H, W = img1.shape
    x = torch.cat([img1, img2], dim=1)          # [1,6,H,W]
    x_pad, ph, pw = pad_to_64(x)                # pad bottom/right
    out = model(x_pad)
    flow = select_finest_flow(out)              # [1,2,Hp,Wp] (often 1/4 res)
    flow = unpad(flow, ph, pw)                  # still might be < (H,W)
    # >>> IMPORTANT: upsample to (H,W) for metrics <<<
    if flow.shape[-2:] != (H, W):
        flow = flow_resize(flow, H, W)
    return flow   


def evaluate_kitti(model, loader, device):
    model.to(device).eval()

    epe_list = []
    fl_list = []

    with torch.no_grad():
        for img1, img2, flow_gt, valid, stem in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow_gt = flow_gt.to(device)   # [B,2,H,W]
            valid = valid.to(device).bool()  # [B,H,W]

            B = img1.shape[0]
            assert B == 1, "Use batch_size=1 for evaluation to simplify bookkeeping."

            flow_pred = model_infer(model, img1, img2)  # [1,2,H,W]

            # Convert to numpy for metrics
            fp = flow_pred[0].permute(1, 2, 0).cpu().numpy()   # [H,W,2]
            fg = flow_gt[0].permute(1, 2, 0).cpu().numpy()
            vm = valid[0].cpu().numpy()

            epe = epe_metric(fp, fg, vm)
            fl = fl_all_metric(fp, fg, vm)

            epe_list.append(epe)
            fl_list.append(fl)

            print(f"{stem[0]} | EPE: {epe:.3f} | Fl-all: {fl:.2f}%")

    mean_epe = float(np.nanmean(epe_list)) if len(epe_list) else np.nan
    mean_fl  = float(np.nanmean(fl_list)) if len(fl_list) else np.nan
    print("=" * 60)
    print(f"Mean EPE:   {mean_epe:.3f}")
    print(f"Mean Fl-all:{mean_fl:.2f}%")
    return mean_epe, mean_fl


# =========================
# Main
# =========================
def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only='True')
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    return model


def parse_args():
    ap = argparse.ArgumentParser("PWC-Net KITTI Evaluation")
    ap.add_argument("--kitti_root", type=str, required=True,
                    help="Path to KITTI 2012/2015 training root (contains image_2/ and flow_* dirs)")
    ap.add_argument("--kitti_year", type=int, default=2015, choices=[2012, 2015],
                    help="Which KITTI split to evaluate (affects default folder names)")
    ap.add_argument("--images_dir", type=str, default=None,
                    help="Override image folder (default: image_2 for 2015, colored_0 for 2012)")
    ap.add_argument("--flow_dir", type=str, default=None,
                    help="Override flow folder (default: flow_occ for 2015, flow_noc for 2012)")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Checkpoint path (e.g., pwcnet_proxy_epoch_50.tar)")
    ap.add_argument("--batch_size", type=int, default=1, help="Prefer 1 for exact metrics")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable ImageNet normalization if your training didn't use it")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Model
    model = PWCDCNet()
    model = load_checkpoint_into_model(model, args.ckpt, device)

    # Dataset / Loader
    ds = KITTIPairsFlow(root=args.kitti_root,
                        images_dir=args.images_dir,
                        flow_dir=args.flow_dir,
                        kitti_year=args.kitti_year,
                        normalize=(not args.no_normalize))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Evaluate
    evaluate_kitti(model, dl, device)


if __name__ == "__main__":
    main()
