# pwc_extract_flow.py
import math, struct, os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# =========================
# 1) Model definition / weights  (EDIT THESE TWO LINES)
# =========================
# TODO: replace with your actual import (e.g., from PWC_src.pwc_net import PWCDCNet)
import models
from models.PWCNet import PWCDCNet
import cv2

# TODO: path to weights (the state_dict keys should match the PWCDCNet module)
CKPT_PATH = "pwc_net.pth.tar"
# CKPT_PATH = './checkpoints_kitti/251020_061204/best_model251020_061204.pth'
# =========================
# 2) Utility: I/O & viz
# =========================
def load_image_as_tensor(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    # to float32 [0,1], CHW
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    t = torch.from_numpy(arr)            # [3, H, W]
    return t, (h, w)

def pad_to_multiple_of_64(img1, img2):
    # Both are [1,3,H,W]; PWC-Net likes dims divisible by 64
    _, _, h, w = img1.shape
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    img1p = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
    img2p = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')
    return img1p, img2p, pad_h, pad_w

def unpad(flow, pad_h, pad_w):
    # flow: [1,2,H,W]
    if pad_h or pad_w:
        return flow[:, :, :-pad_h if pad_h else None, :-pad_w if pad_w else None]
    return flow

def write_flo(filename, flow_uv):
    """
    Middlebury .flo writer
    flow_uv: [H, W, 2] float32
    """
    with open(filename, 'wb') as f:
        f.write(struct.pack('f', 202021.25))  # magic
        h, w, _ = flow_uv.shape
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        f.write(flow_uv.astype(np.float32).tobytes())

def flow_to_color(flow_uv, clip_flow=None):
    """
    Convert flow (H,W,2) to RGB colorwheel visualization in [0,255].
    Based on the classic Middlebury color wheel.
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    if clip_flow is not None:
        rad = np.sqrt(u**2 + v**2)
        rad_clip = np.maximum(rad, 1e-5)
        u = u * (clip_flow / np.maximum(rad_clip, clip_flow))
        v = v * (clip_flow / np.maximum(rad_clip, clip_flow))

    rad = np.sqrt(u**2 + v**2)
    ang = np.arctan2(-v, -u) / np.pi  # [-1,1]
    fk = (ang + 1) / 2 * (55 - 1) + 1  # map angle to [1,55]
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1)
    f = fk - k0
    # Build wheel
    def make_colorwheel():
        # RY, YG, GC, CB, BM, MR steps
        RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
        ncols = RY+YG+GC+CB+BM+MR
        colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY).astype(np.uint8)
        col += RY
        # YG
        colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG).astype(np.uint8)
        colorwheel[col:col+YG, 1] = 255
        col += YG
        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC).astype(np.uint8)
        col += GC
        # CB
        colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0,CB)/CB).astype(np.uint8)
        colorwheel[col:col+CB, 2] = 255
        col += CB
        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM).astype(np.uint8)
        col += BM
        # MR
        colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0,MR)/MR).astype(np.uint8)
        colorwheel[col:col+MR, 0] = 255
        return colorwheel
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    k0 = (k0 - 1) % ncols
    k1 = (k1 - 1) % ncols
    f = f[..., None]

    col0 = colorwheel[k0] / 255.0
    col1 = colorwheel[k1] / 255.0
    col = (1 - f) * col0 + f * col1

    # Attenuate saturation by magnitude
    rad_norm = np.clip(rad / (np.max(rad) + 1e-5), 0, 1)[..., None]
    col = 1 - rad_norm * (1 - col)
    rgb = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    return rgb  # H,W,3 uint8

# =========================
# 3) Inference
# =========================
def load_model(device="cuda"):
    model = PWCDCNet().to(device).eval()
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # Support either pure state_dict or wrapped
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt 
    # Strip 'module.' if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model

@torch.no_grad()
def pwc_infer(img1_path, img2_path, device="cuda"):
    # Load & batch
    t1, (h, w) = load_image_as_tensor(img1_path)
    t2, _        = load_image_as_tensor(img2_path)
    t1 = t1.unsqueeze(0).to(device)  # [1,3,H,W]
    t2 = t2.unsqueeze(0).to(device)

    # Pad
    t1p, t2p, pad_h, pad_w = pad_to_multiple_of_64(t1, t2)

    # Model
    model = load_model(device)
    # flow = model(t1p, t2p)            # [1,2,Hpad,Wpad], pixels at that scale
    # Concatenate images along channels: [1,3,H,W] + [1,3,H,W] -> [1,6,H,W]
    x = torch.cat([t1p, t2p], dim=1)

    with torch.no_grad():
        out = model(x)
        
    if isinstance(out, (tuple, list)):
        flow = out[0]
    else:
        flow = out

    # Normalize the variety of return types across repos
    if isinstance(out, (list, tuple)):
        flow = out[0]
    elif isinstance(out, dict):
        flow = out.get("flow", next(v for v in out.values() if hasattr(v, "shape") and v.dim() == 4 and v.shape[1] == 2))
    else:
        flow = out

    flow = unpad(flow, pad_h, pad_w)  # [1,2,H,W]

    # If you want to resize flow to a *different* size, remember to scale u,v
    # For original size, we’re already at (H,W) after unpad.
    flow_np = flow.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()  # [H,W,2]

    return flow_np  # u,v per pixel at original resolution

def save_outputs(flow_uv, out_prefix="flow"):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # .npy
    np.save(out_prefix + ".npy", flow_uv)  # [H,W,2], float32
    # .flo
    write_flo(out_prefix + ".flo", flow_uv)
    # colorized .png
    color = flow_to_color(flow_uv)
    Image.fromarray(color).save(out_prefix + ".png")


def save_quiver_overlay(img_path, flow_uv, out_png, step=16, scale=1, min_mag=0.5):
    """
    Draws arrows on top of the image using matplotlib.quiver.
    - step: sample stride in pixels
    - scale: quiver scale (bigger -> shorter arrows; use ~1–50)
    - min_mag: skip arrows with magnitude below this (pixels)
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]

    Hf, Wf = flow_uv.shape[:2]


    scale_x = float(W) / float(Wf)
    scale_y = float(H) / float(Hf)
    u = cv2.resize(flow_uv[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * scale_x
    v = cv2.resize(flow_uv[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * scale_y
    flow_uv = np.stack([u, v], axis=-1)
    

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]

    # sample a grid
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    uu = u[0:H:step, 0:W:step]
    vv = v[0:H:step, 0:W:step]

    # mask small motions
    mag = np.sqrt(uu**2 + vv**2)
    mask = mag >= min_mag

    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img, origin='upper')
    # angles='xy', scale_units='xy' means (uu,vv) are interpreted in pixel units.
    plt.quiver(xs[mask], ys[mask], uu[mask], vv[mask],
               angles='xy', scale_units='xy', scale=scale, width=0.0015)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--im1", default = './data/roll0_tilt0_yaw0_0065.png')
    ap.add_argument("--im2", default = './data/roll0_tilt0_yaw0_0067.png')
    ap.add_argument("--out", default="flow/flow_mymodel")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # save_path = os.path.join(args.out,CKPT_PATH)
    # os.mkdir(save_path, exist_ok=True)

    flow_uv = pwc_infer(args.im1, args.im2, device=args.device)
    save_outputs(flow_uv, args.out)
    save_quiver_overlay(args.im1, flow_uv, args.out + os.path.splitext(os.path.basename(CKPT_PATH))[0] + "_arrows_mymodel1.png", step=16, scale=1, min_mag=0.5)
    print(f"Saved: {args.out}.npy, {args.out}.flo, {args.out}.png")
    print(f"Flow shape: {flow_uv.shape} (H,W,2); dtype={flow_uv.dtype}")
