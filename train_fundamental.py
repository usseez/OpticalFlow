"""
PWC-Net Fine-tuning with Proxy-Label Method
Self-supervised training using consecutive frames
+ Epipolar filtering (Fundamental Matrix) to suppress outliers
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from models.PWCNet import PWCDCNet


# =========================
# Dataset load
# =========================
class ConsecutiveFrameDataset(Dataset):
    """Dataset for loading consecutive image pairs from folders"""

    def __init__(self, root_dir, transform=None, frame_gap=1):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frame_gap = frame_gap
        self.image_pairs = []

        for subfolder in sorted(self.root_dir.iterdir()):
            if subfolder.is_dir():
                images = sorted(list(subfolder.glob("*.png")))
                for i in range(len(images) - frame_gap):
                    self.image_pairs.append((images[i], images[i + frame_gap]))

        print(f"Found {len(self.image_pairs)} image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2



# =========================
# Warping helpers
# =========================
def upsample_flow_to(flow, H, W):
    """
    Resize flow [B,2,h,w] to (H,W) and scale vectors accordingly.
    """
    b, c, h, w = flow.shape
    if (h, w) == (H, W):
        return flow
    scale_y = H / h
    scale_x = W / w
    flow_up = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)
    flow_up[:, 0] *= scale_x
    flow_up[:, 1] *= scale_y
    return flow_up


def warp_image(img, flow):
    """
    img:  [B, C, H, W]
    flow: [B, 2, h, w] pixel flow (dx, dy)
    returns warped img2 -> img1
    """
    B, C, H, W = img.shape
    flow = upsample_flow_to(flow, H, W)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype),
        torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    flow_norm = torch.empty((B, H, W, 2), device=img.device, dtype=img.dtype)
    flow_norm[..., 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
    flow_norm[..., 1] = 2.0 * flow[:, 1] / max(H - 1, 1)
    grid = base + flow_norm
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


# =========================
# ProxyLabel Loss (with mask)
# =========================
class ProxyLabelLoss(nn.Module):
    """
    Photometric (SSIM+L1) + Smoothness
    Supports a boolean/float mask to IGNORE outliers (e.g., from epipolar filter).
    """

    def __init__(self, alpha_photo=1.0, alpha_smooth=0.1):
        super().__init__()
        self.alpha_photo = alpha_photo
        self.alpha_smooth = alpha_smooth

    def forward(self, flow, img1, img2, valid_mask=None):
        """
        flow: [B,2,H,W] (can be lower-res; will be upsampled inside warp)
        img1,img2: [B,3,H,W]
        valid_mask: [B,1,H,W] or [B,H,W] (True/1 = use, False/0 = ignore)
        """
        img2_warp = warp_image(img2, flow)

        photo = self.photometric_loss(img1, img2_warp, valid_mask=valid_mask)
        smooth = self.smoothness_loss(flow)  # usually unmasked

        total = self.alpha_photo * photo + self.alpha_smooth * smooth
        return total, photo, smooth

    @staticmethod
    def _masked_mean(x, mask): # 에피폴라 inlier 영역만 photometric loss를 반영할 때 사용.
        if mask is None:
            return x.mean()
        if mask.ndim == x.ndim - 1:
            mask = mask.unsqueeze(1)  # [B,1,H,W]
        m = (mask > 0.5).float()
        num = (x * m).sum()
        den = m.sum().clamp_min(1.0)
        return num / den

    def photometric_loss(self, x, y, valid_mask=None): #채널 차원 평균 L1 거리
        # L1
        l1 = (x - y).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        # SSIM
        ssim = self.ssim_map(x, y)  # [B,1,H,W] in [0,1]
        photo = 0.85 * ssim + 0.15 * l1
        return self._masked_mean(photo, valid_mask)

    @staticmethod
    def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-12
        )
        # map in [0,1]
        return torch.clamp((1 - ssim) / 2, 0, 1).mean(dim=1, keepdim=True)

    @staticmethod
    def smoothness_loss(flow):
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return dx.mean() + dy.mean()


# =========================
# *******Epipolar (F-matrix) utilities (NumPy) + Torch glue
# =========================
def _flow_to_pairs(flow_hw2, stride=4, mask_hw=None):
    """flow_hw2: (H,W,2) numpy -> x1,x2 as (N,3) homogeneous sampled on a stride grid"""
    H, W, _ = flow_hw2.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]  # (Hs, Ws)

    # base coords (strided)
    u = xs.reshape(-1).astype(np.float64)
    v = ys.reshape(-1).astype(np.float64)

    # sample flow at the SAME strided locations
    du = flow_hw2[ys, xs, 0].reshape(-1).astype(np.float64)
    dv = flow_hw2[ys, xs, 1].reshape(-1).astype(np.float64)

    # endpoints
    u2 = u + du
    v2 = v + dv

    valid = np.isfinite(u2) & np.isfinite(v2)
    if mask_hw is not None:
        valid &= mask_hw[ys, xs].reshape(-1).astype(bool)

    u, v, u2, v2 = u[valid], v[valid], u2[valid], v2[valid]
    x1 = np.stack([u, v, np.ones_like(u)], axis=1)      # (N,3)
    x2 = np.stack([u2, v2, np.ones_like(u2)], axis=1)   # (N,3)
    return x1, x2



def _normalize_points(x):
    x = x / (x[:, 2:3] + 1e-12)
    mean = np.mean(x[:, :2], axis=0)
    xc = x[:, :2] - mean
    md = np.mean(np.sqrt(np.sum(xc ** 2, axis=1)) + 1e-12)
    s = np.sqrt(2) / md
    T = np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1]], dtype=np.float64)
    x_norm = (T @ x.T).T
    return x_norm, T


def _eight_point_F(x1, x2):
    x1n, T1 = _normalize_points(x1)
    x2n, T2 = _normalize_points(x2)
    u, v = x1n[:, 0], x1n[:, 1]
    up, vp = x2n[:, 0], x2n[:, 1]
    A = np.stack([u * up, v * up, up, vp * u, vp * v, vp, u, v, np.ones_like(u)], axis=1)
    _, _, VT = np.linalg.svd(A, full_matrices=False)
    F_norm = VT[-1].reshape(3, 3)
    U, S, VT = np.linalg.svd(F_norm)
    S[-1] = 0.0
    F_norm = U @ np.diag(S) @ VT
    Fm = T2.T @ F_norm @ T1
    if np.linalg.norm(Fm) > 0:
        Fm = Fm / Fm[2, 2] if np.abs(Fm[2, 2]) > 1e-12 else Fm / np.linalg.norm(Fm)
    return Fm


def _sampson_distance(Fm, x1, x2):
    x1 = x1 / (x1[:, 2:3] + 1e-12)
    x2 = x2 / (x2[:, 2:3] + 1e-12)
    Fx1 = (Fm @ x1.T).T
    Ftx2 = (Fm.T @ x2.T).T
    x2Fx1 = np.sum(x2 * (Fm @ x1.T).T, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return (x2Fx1 ** 2) / denom  # (N,)


def _ransac_F(x1, x2, max_iters=2000, thresh=0.5, min_samples=8, seed=0):
    rng = np.random.default_rng(seed)
    N = x1.shape[0]
    if N < min_samples:
        raise RuntimeError("Not enough correspondences.")
    best_F, best_in, best_count = None, None, -1
    for _ in range(max_iters):
        idx = rng.choice(N, size=min_samples, replace=False)
        try:
            Fc = _eight_point_F(x1[idx], x2[idx])
        except np.linalg.LinAlgError:
            continue
        d = _sampson_distance(Fc, x1, x2)
        inliers = d < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_F, best_in, best_count = Fc, inliers, cnt
    if best_in is None or best_in.sum() < min_samples:
        raise RuntimeError("RANSAC failed.")
    F_ref = _eight_point_F(x1[best_in], x2[best_in])
    return F_ref


def build_epipolar_mask_from_flow(flow_b2hw, tau=1.0, stride=4, img_mask_bhw=None,
                                  keep_ratio=0.2, min_keep=0.05):
    """
    flow_b2hw: torch [1,2,H,W] (per-sample)
    tau:       Sampson base threshold (절대값 상한)
    keep_ratio: Sampson distance가 작은 픽셀 중 상위 몇 %만 살릴지 (0~1)
    min_keep:   마스크가 너무 작아지지 않도록 하는 최소 비율 (0~1)
    returns mask [1,1,H,W] torch.bool  (True=keep)
    """
    device = flow_b2hw.device
    dtype = flow_b2hw.dtype
    B, C, H, W = flow_b2hw.shape
    assert B == 1, "Call per-sample inside the batch loop."

    # to numpy (H,W,2)
    flow_np = flow_b2hw[0].permute(1, 2, 0).detach().cpu().numpy()
    mask_np = None
    if img_mask_bhw is not None:
        mask_np = img_mask_bhw[0].detach().cpu().numpy().astype(bool)

    # 1) F 추정용 대응점 샘플링
    x1, x2 = _flow_to_pairs(flow_np, stride=stride, mask_hw=mask_np)
    try:
        Fm = _ransac_F(x1, x2, thresh=0.5, max_iters=2000, seed=0)
    except Exception:
        # F 추정 실패 시 그냥 전부 사용
        return torch.ones((1, 1, H, W), dtype=torch.bool, device=device)

    # 2) 전체 픽셀에 대한 Sampson distance 계산
    ys, xs = np.mgrid[0:H, 0:W]
    u  = xs.reshape(-1).astype(np.float64)
    v  = ys.reshape(-1).astype(np.float64)
    u2 = (xs + flow_np[..., 0]).reshape(-1).astype(np.float64)
    v2 = (ys + flow_np[..., 1]).reshape(-1).astype(np.float64)

    X1 = np.stack([u,  v,  np.ones_like(u)],  axis=1)  # (HW,3)
    X2 = np.stack([u2, v2, np.ones_like(u2)], axis=1)  # (HW,3)

    d = _sampson_distance(Fm, X1, X2).reshape(H, W)    # (H,W)

    # 3) 유효한 애들만 사용
    finite = np.isfinite(d)
    if not finite.any():
        # 전부 NaN/Inf면 그냥 다 살림
        return torch.ones((1, 1, H, W), dtype=torch.bool, device=device)

    d_valid = d[finite]

    # 4) base threshold = tau 와 percentile 기반 threshold 중 더 작은 쪽 사용
    #    → Sampson distance 분포에서 가장 작은 keep_ratio 비율만 남기도록
    thr = float(tau)
    if 0.0 < keep_ratio < 1.0:
        q = float(np.quantile(d_valid, keep_ratio))
        thr = min(thr, q)

    keep = finite & (d <= thr)

    # 5) 마스크가 너무 작으면 (예: 1% 미만) 조금 풀어서 min_keep은 남겨줌
    if 0.0 < min_keep < 1.0:
        cur_ratio = keep.mean()  # bool 배열 mean → 비율
        if cur_ratio < min_keep:
            # min_keep 비율까지는 남도록 threshold 완화
            q_min = float(np.quantile(d_valid, min_keep))
            thr_relaxed = min(float(tau), q_min)
            keep = finite & (d <= thr_relaxed)

    keep_t = torch.from_numpy(keep).to(device=device)
    return keep_t.view(1, 1, H, W)



# Optional soft epipolar penalty (tiny weight recommended)
def epipolar_sampson_loss(flow_bchw, F_np, valid_mask=None, robust="huber", delta=1.0, weight=0.1):
    """
    flow_bchw: [B,2,H,W], F_np: numpy 3x3 (constant), valid_mask: [B,1,H,W] or [B,H,W]
    Returns scalar torch loss * weight
    """
    B, _, H, W = flow_bchw.shape
    device = flow_bchw.device
    dtype = flow_bchw.dtype

    # grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs, dtype=dtype)

    # per-batch residuals
    Fm = torch.from_numpy(F_np).to(device=device, dtype=dtype)
    res_all = []
    for b in range(B):
        u1 = xs.reshape(-1).float()
        v1 = ys.reshape(-1).float()
        u2 = (xs + flow_bchw[b, 0]).reshape(-1).float()
        v2 = (ys + flow_bchw[b, 1]).reshape(-1).float()
        x1 = torch.stack([u1, v1, ones.reshape(-1)], dim=1)  # (N,3)
        x2 = torch.stack([u2, v2, ones.reshape(-1)], dim=1)  # (N,3)

        Fx1 = x1 @ Fm.t()
        Ftx2 = x2 @ Fm
        x2Fx1 = torch.sum(x2 * (x1 @ Fm.t()), dim=1)
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
        d = (x2Fx1 ** 2) / denom
        res_all.append(d.view(H, W))

    dmap = torch.stack(res_all, dim=0)  # [B,H,W]

    if valid_mask is not None:
        m = (valid_mask > 0.5).float().view(B, H, W)
        dmap = dmap[m > 0.5]

    if dmap.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    if robust == "huber":
        r = torch.sqrt(dmap + 1e-12)
        loss = torch.where(r <= delta, 0.5 * (r ** 2) / delta, r - 0.5 * delta).mean()
    elif robust == "l1":
        loss = torch.sqrt(dmap + 1e-12).mean()
    else:
        loss = dmap.mean()

    return weight * loss


# =========================
# Forward–Backward consistency & OOB (validation metrics)
# =========================
@torch.no_grad()
def _select_finest_flow(outputs):
    if isinstance(outputs, (list, tuple)):
        flows = [f for f in outputs if isinstance(f, torch.Tensor)]
        flows.sort(key=lambda t: (t.shape[-2] * t.shape[-1]), reverse=True)
        return flows[0]
    return outputs


@torch.no_grad()
def forward_backward_cycle(model, img1, img2):
    model.eval()
    inp12 = torch.cat([img1, img2], dim=1)
    flow12 = _select_finest_flow(model(inp12))
    inp21 = torch.cat([img2, img1], dim=1)
    flow21 = _select_finest_flow(model(inp21))
    B, _, H, W = img1.shape
    flow12 = upsample_flow_to(flow12, H, W)
    flow21 = upsample_flow_to(flow21, H, W)
    flow21_warped = warp_image(flow21, flow12)
    cycle = flow12 + flow21_warped
    return cycle.abs().mean()


@torch.no_grad()
def oob_ratio(flow, H, W, device, dtype):
    flow = upsample_flow_to(flow, H, W)
    B = flow.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    flow_norm = torch.empty((B, H, W, 2), device=device, dtype=dtype)
    flow_norm[..., 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
    flow_norm[..., 1] = 2.0 * flow[:, 1] / max(H - 1, 1)
    grid = base + flow_norm
    x = grid[..., 0]
    y = grid[..., 1]
    oob = (x < -1) | (x > 1) | (y < -1) | (y > 1)
    return oob.float().mean()


# =========================
# Train / Validate
# =========================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                epi_stride=4, epi_thresh=0.3, epi_soft_w=0.1):
    """
    Adds epipolar hard filtering + optional soft loss.
    """
    model.train()
    running_loss = running_photo = running_smooth = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for img1, img2 in pbar:
        img1, img2 = img1.to(device), img2.to(device)
        input_pair = torch.cat([img1, img2], dim=1) #input pair = [B,6,H,W]

        optimizer.zero_grad()

        outputs = model(input_pair)
        flow_pred = _select_finest_flow(outputs)  # [B,2,h,w]
        # upsample for masking/warping consistency
        B, _, H, W = img1.shape
        flow_full = upsample_flow_to(flow_pred, H, W)

        # -------- Epipolar HARD mask (per-sample) --------
        # build a boolean mask [B,1,H,W], True=keep
        keep_masks = []
        for b in range(B):#for each sample in batch
            mask_b = build_epipolar_mask_from_flow(
                flow_full[b:b+1], tau=epi_thresh, stride=epi_stride
            )  # [1,1,H,W]
            keep_masks.append(mask_b)
        keep_mask = torch.cat(keep_masks, dim=0)  # [B,1,H,W]

        # -------- Losses --------
        total, photo, smooth = criterion(flow_pred, img1, img2, valid_mask=keep_mask)

        # Optional soft epipolar penalty (tiny weight)
        # We reuse F estimated inside 'build_epipolar_mask_from_flow' only for masking.
        # For the soft loss, estimate F once from the SAME predicted flow of the first sample (cheap).
        if epi_soft_w > 0.0:
            with torch.no_grad():
                F_np = None
                try:
                    flow_np = flow_full[0].permute(1, 2, 0).cpu().numpy()
                    x1, x2 = _flow_to_pairs(flow_np, stride=epi_stride, mask_hw=None)
                    F_np = _ransac_F(x1, x2, thresh=1.0, max_iters=1000, seed=0)
                except Exception:
                    F_np = None
            if F_np is not None:
                total = total + epipolar_sampson_loss(flow_full, F_np, valid_mask=keep_mask, weight=epi_soft_w)

        total.backward()
        optimizer.step()

        running_loss += float(total.item())
        running_photo += float(photo.item())
        running_smooth += float(smooth.item())

        pbar.set_postfix({
            "loss": f"{float(total.item()):.4f}",
            "photo": f"{float(photo.item()):.4f}",
            "smooth": f"{float(smooth.item()):.4f}",
            "keep%": f"{100.0 * keep_mask.float().mean().item():.1f}"
        })

    n = len(dataloader)
    return running_loss / n, running_photo / n, running_smooth / n


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    photo_sum = smooth_sum = fb_sum = oob_sum = 0.0
    n = 0

    for img1, img2 in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        inp = torch.cat([img1, img2], dim=1)
        outputs = model(inp)
        flow = _select_finest_flow(outputs)

        total_loss, photo_loss, smooth_loss = criterion(flow, img1, img2, valid_mask=None)
        photo_sum += float(photo_loss.item())
        smooth_sum += float(smooth_loss.item())

        fb = forward_backward_cycle(model, img1, img2)
        fb_sum += float(fb.item())

        B, _, H, W = img1.shape
        flow_full = upsample_flow_to(flow, H, W)
        oob = oob_ratio(flow_full, H, W, device=img1.device, dtype=img1.dtype)
        oob_sum += float(oob.item())
        n += 1

    n = max(n, 1)
    return {
        "val_photo": photo_sum / n,
        "val_smooth": smooth_sum / n,
        "val_fb": fb_sum / n,
        "val_oob": oob_sum / n,
    }


# =========================
# Main
# =========================
def main():
    config = {
        "train_dir": "../../../dataset/train",
        "val_dir": "../../../dataset/val",
        "checkpoint_path": "pwc_net_chairs.pth.tar",
        "output_dir": "./checkpoints_pseudo_fund_thresh0.5",
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 1e-7,
        "weight_decay": 4e-4,
        "frame_gap": 1,
        "alpha_photo": 1.0,
        "alpha_smooth": 0.1,
        "save_interval": 5,
        # Epipolar filtering params:
        "epi_stride": 6,     # sampling stride for F estimation
        "epi_thresh": 0.1,   # Sampson threshold for hard mask (lower = stricter)
        "epi_soft_w": 0.1,   # weight for optional soft loss (0.0 to disable)
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    t_dataset = ConsecutiveFrameDataset(
        root_dir=config["train_dir"], transform=transform, frame_gap=config["frame_gap"]
    )
    v_dataset = ConsecutiveFrameDataset(
        root_dir=config["val_dir"], transform=transform, frame_gap=config["frame_gap"]
    )

    train_loader = DataLoader(t_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(v_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = PWCDCNet()
    ckpt = torch.load(config["checkpoint_path"], weights_only='True')
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model = model.to(device)

    # Loss
    criterion = ProxyLabelLoss(
        alpha_photo=config["alpha_photo"],
        alpha_smooth=config["alpha_smooth"],
    )

    # Optimizer / Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss, avg_photo, avg_smooth = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            epi_stride=config["epi_stride"],
            epi_thresh=config["epi_thresh"],
            epi_soft_w=config["epi_soft_w"]
        )
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, photo={avg_photo:.4f}, smooth={avg_smooth:.4f}")

        val_metrics = validate(model, val_loader, criterion, device)
        print(
            f"val_photo={val_metrics['val_photo']:.4f} | "
            f"val_fb={val_metrics['val_fb']:.4f} | "
            f"val_oob={val_metrics['val_oob']:.4f}"
        )

        scheduler.step()

        # if epoch % config["save_interval"] == 0:
        save_path = os.path.join(config["output_dir"], f"pwcnet_proxy_epoch_{epoch}, loss_{avg_loss:.4f}.tar")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, save_path)
        print("Checkpoint saved:", save_path)


if __name__ == "__main__":
    main()
