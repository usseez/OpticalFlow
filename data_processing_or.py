# data_processing.py
# Preprocessing & dataset for KITTI fine-tuning (PWC-Net)
# - 896x320 crops
# - reduced rotation/zoom/squeeze augmentations
# - KITTI 2012/2015 semi-dense flow PNG reader (mask-aware)

import os
import math
import random
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------
# KITTI .png flow I/O
# --------------------------
def read_kitti_flow_png(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read KITTI 2012/2015 flow ground truth PNG.

    Returns:
        flow:  float32 [H,W,2] in pixels
        valid: uint8   [H,W,1] with values {0,1}

    The PNG has 3x 16-bit channels. Two encode (u,v) with:
        real_value = (value - 2^15) / 64.0
    The remaining channel is a binary valid mask.

    We autodetect which channel is the mask by finding the one
    that only contains {0,1}. This is robust against BGR/RGB order.
    """
    png = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWx3, uint16
    if png is None:
        raise FileNotFoundError(path)
    if png.dtype != np.uint16 or png.ndim != 3 or png.shape[2] != 3:
        raise ValueError(f"Unexpected KITTI flow format: {path} -> {png.shape}, {png.dtype}")

    ch = [png[..., 0], png[..., 1], png[..., 2]]
    # Identify mask channel (only 0/1 values)
    mask_idx = None
    for i in range(3):
        vals = np.unique(ch[i])
        if vals.size <= 2 and set(vals.tolist()).issubset({0, 1}):
            mask_idx = i
            break
    if mask_idx is None:  # Fallback: assume last channel is mask
        mask_idx = 2

    valid = ch[mask_idx].astype(np.uint8)[..., None]  # [H,W,1]
    uv_idx = [i for i in range(3) if i != mask_idx]
    u_raw = ch[uv_idx[0]].astype(np.float32)
    v_raw = ch[uv_idx[1]].astype(np.float32)

    # Decode to pixels
    flow_u = (u_raw - 32768.0) / 64.0
    flow_v = (v_raw - 32768.0) / 64.0
    flow = np.stack([flow_u, flow_v], axis=-1).astype(np.float32)  # [H,W,2]
    return flow, valid


# --------------------------
# Geometry helpers
# --------------------------
def _affine_params_reduced():
    """
    Reduced augmentation for KITTI fine-tuning, as per paper:
    - small rotation, mild anisotropic scaling (squeeze), mild zoom.
    """
    # Rotation in degrees (very small)
    rot_deg = random.uniform(-2.0, 2.0)

    # Zoom (isotropic) ~ [0.95, 1.05]
    zoom = random.uniform(0.95, 1.05)

    # Squeeze: mild anisotropic scaling around 1.0
    squeeze_x = random.uniform(0.97, 1.03)
    squeeze_y = random.uniform(0.97, 1.03)

    sx = zoom * squeeze_x
    sy = zoom * squeeze_y
    return rot_deg, sx, sy


def _cv2_affine_matrix(center_xy, rot_deg, sx, sy, translate_xy=(0.0, 0.0)):
    """
    Build 2x3 affine matrix for cv2.warpAffine with separate sx, sy and rotation around center.
    """
    cx, cy = center_xy
    theta = np.deg2rad(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Scale then rotate:
    # A = R * S
    A = np.array([[sx * cos_t, -sy * sin_t],
                  [sx * sin_t,  sy * cos_t]], dtype=np.float32)

    tx, ty = translate_xy
    # t = c - A c + translation
    t = np.array([cx, cy], dtype=np.float32) - A @ np.array([cx, cy], dtype=np.float32) + np.array([tx, ty], dtype=np.float32)

    M = np.concatenate([A, t[:, None]], axis=1)  # 2x3
    return M, A  # A is the 2x2 linear part (to transform vectors)


def _warp_image(img, M, out_size):
    """
    img: HxWxC (float32, in [0,1] or uint8)
    M:   2x3 cv2 affine
    """
    H, W = out_size
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _warp_flow_and_mask(flow, valid, M, A, out_size):
    """
    Warp flow and mask with affine M.
    For flow vectors, apply spatial resampling AND transform vectors by A.
    """
    H, W = out_size
    # Resample flow as two scalar fields
    fu = _warp_image(flow[..., 0], M, out_size)
    fv = _warp_image(flow[..., 1], M, out_size)
    # Transform vectors by the linear component A
    # [fu', fv']^T = A * [fu, fv]^T
    fu_t = A[0, 0] * fu + A[0, 1] * fv
    fv_t = A[1, 0] * fu + A[1, 1] * fv
    flow_t = np.stack([fu_t, fv_t], axis=-1).astype(np.float32)

    # Mask: nearest interpolation; keep binary
    valid_f = _warp_image(valid.astype(np.float32), M, out_size)
    valid_t = (valid_f > 0.5).astype(np.uint8)
    return flow_t, valid_t[..., None]


def resize_flow_np(flow, new_h, new_w):
    """
    Resize flow (H,W,2) -> (new_h,new_w,2) and scale vectors appropriately.
    """
    h, w = flow.shape[:2]
    if (h, w) == (new_h, new_w):
        return flow
    sx = new_w / float(w)
    sy = new_h / float(h)
    u = cv2.resize(flow[..., 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR) * sx
    v = cv2.resize(flow[..., 1], (new_w, new_h), interpolation=cv2.INTER_LINEAR) * sy
    return np.stack([u, v], axis=-1).astype(np.float32)


# --------------------------
# Dataset
# --------------------------
class KittiFlowDataset(Dataset):
    """
    Minimal dataset for KITTI fine-tuning.

    You can either:
      1) pass a text file list with lines: "<img1> <img2> <flow_png>"
      2) auto-scan a root dir that has subfolders with left images and flow PNGs

    Returns:
      x      : float32 [6,H,W]   (RGB1||RGB2 in [0,1])
      flow   : float32 [2,H,W]   (pixels)
      valid  : uint8   [1,H,W]   (0/1 mask)
    """

    def __init__(self,
                 root: str,
                 list_txt: str = None,
                 auto_scan: bool = False,
                 crop_hw: Tuple[int, int] = (320, 896),
                 apply_aug: bool = True):
        self.root = root
        self.crop_h, self.crop_w = crop_hw
        self.apply_aug = apply_aug

        self.samples: List[Tuple[str, str, str]] = []
        if list_txt is not None:
            with open(list_txt, 'r') as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 3:
                        self.samples.append((p[0], p[1], p[2]))
        elif auto_scan:
            # a simple scan; adapt to your structure if needed
            # expects .../training/image_2/* and .../training/flow_occ/*
            img_dirs = sorted(glob(os.path.join(root, 'image_2')))
            for img_dir in img_dirs:
                flows_dir = img_dir.replace("image_2", "flow_occ")
                if not os.path.isdir(flows_dir):
                    continue
                left_imgs = sorted(glob(os.path.join(img_dir, "*.png")))
                for i in range(len(left_imgs) - 1):
                    im1 = left_imgs[i]
                    im2 = left_imgs[i + 1]
                    # flow filename often aligns with first image index
                    base = os.path.splitext(os.path.basename(im1))[0]
                    flow = os.path.join(flows_dir, f"{base}.png")
                    if os.path.isfile(flow):
                        self.samples.append((im1, im2, flow))
        else:
            raise ValueError("Provide list_txt or set auto_scan=True")

        if not self.samples:
            raise RuntimeError("No KITTI samples found")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0  # [H,W,3]
        return arr

    def _random_crop_coords(self, H, W):
        y = 0 if H == self.crop_h else random.randint(0, H - self.crop_h)
        x = 0 if W == self.crop_w else random.randint(0, W - self.crop_w)
        return y, x

    def _reduced_augmentation(self, im1, im2, flow, valid):
        """Reduced rotation/zoom/squeeze (paper) applied identically to both frames & flow."""
        H, W = im1.shape[:2]
        if not self.apply_aug:
            return im1, im2, flow, valid

        # Small chance to skip aug entirely
        if random.random() < 0.4:
            return im1, im2, flow, valid

        rot_deg, sx, sy = _affine_params_reduced()
        M, A = _cv2_affine_matrix(center_xy=(W * 0.5, H * 0.5), rot_deg=rot_deg, sx=sx, sy=sy)

        im1_t = _warp_image((im1 * 255.0).astype(np.uint8), M, (H, W)).astype(np.float32) / 255.0
        im2_t = _warp_image((im2 * 255.0).astype(np.uint8), M, (H, W)).astype(np.float32) / 255.0
        flow_t, valid_t = _warp_flow_and_mask(flow, valid, M, A, (H, W))
        return im1_t, im2_t, flow_t, valid_t

    def __getitem__(self, idx):
        im1_path, im2_path, flow_path = self.samples[idx]
        im1 = self._load_rgb(im1_path)
        im2 = self._load_rgb(im2_path)
        flow, valid = read_kitti_flow_png(flow_path)  # [H,W,2], [H,W,1]

        H, W = im1.shape[:2]
        assert im2.shape[:2] == (H, W), "Mismatched image sizes"

        # Reduced augmentation (paper)
        im1, im2, flow, valid = self._reduced_augmentation(im1, im2, flow, valid)

        # Ensure we can crop 896x320 (W x H)
        if (H, W) != (self.crop_h, self.crop_w):
            # If smaller, first upsize (and scale flow vectors)
            new_h = max(H, self.crop_h)
            new_w = max(W, self.crop_w)
            if (new_h, new_w) != (H, W):
                im1 = cv2.resize(im1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                im2 = cv2.resize(im2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                flow = resize_flow_np(flow, new_h, new_w)
                valid = cv2.resize(valid, (new_w, new_h), interpolation=cv2.INTER_NEAREST)[..., None]
                H, W = new_h, new_w

        # Random crop to exactly 320x896 (H x W)
        y, x = self._random_crop_coords(H, W)
        y = min(y, H - self.crop_h)
        x = min(x, W - self.crop_w)

        im1 = im1[y:y + self.crop_h, x:x + self.crop_w]
        im2 = im2[y:y + self.crop_h, x:x + self.crop_w]
        flow = flow[y:y + self.crop_h, x:x + self.crop_w]
        valid = valid[y:y + self.crop_h, x:x + self.crop_w]

        # Occasionally horizontal flip (optional)
        if self.apply_aug and random.random() < 0.3:
            im1 = np.ascontiguousarray(im1[:, ::-1])
            im2 = np.ascontiguousarray(im2[:, ::-1])
            flow = np.ascontiguousarray(flow[:, ::-1])
            valid = np.ascontiguousarray(valid[:, ::-1])
            flow[..., 0] *= -1.0  # invert u

        # To tensors for model
        im1_t = torch.from_numpy(im1).permute(2, 0, 1).float()  # [3,320,896]
        im2_t = torch.from_numpy(im2).permute(2, 0, 1).float()  # [3,320,896]
        x = torch.cat([im1_t, im2_t], dim=0)                    # [6,320,896]
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).float()   # [2,320,896]
        valid_t = torch.from_numpy(valid).permute(2, 0, 1).float() # [1,320,896] (0/1)
        return x, flow_t, valid_t


# --------------------------
# Utilities used by training
# --------------------------
def upsample_flow_to(flow_lr: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Upsample a [B,2,h,w] flow to [B,2,H,W] and scale u,v accordingly.
    """
    _, _, h, w = flow_lr.shape
    sx = W / float(w)
    sy = H / float(h)
    out = F.interpolate(flow_lr, size=(H, W), mode='bilinear', align_corners=False)
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out
