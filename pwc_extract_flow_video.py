# pwc_extract_flow_video_arrows_comparison.py
import math, struct, os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import io
from ptflops import get_model_complexity_info

# =========================
# 1) Model definition / weights
# =========================
import models
from models.PWCNet import PWCDCNet

# TODO: path to weights
CKPT_PATH = "./checkpoints_pseudo/251031/pwcnet_proxy_epoch_50.tar"



# =========================
# 2) Utility: I/O & viz
# =========================
def frame_to_tensor(frame):
    """Convert numpy frame (H,W,3) BGR to tensor (3,H,W) RGB"""
    # OpenCV loads as BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = frame_rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    t = torch.from_numpy(arr)
    return t

def pad_to_multiple_of_64(img1, img2):
    _, _, h, w = img1.shape
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    img1p = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
    img2p = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')
    return img1p, img2p, pad_h, pad_w

def unpad(flow, pad_h, pad_w):
    if pad_h or pad_w:
        return flow[:, :, :-pad_h if pad_h else None, :-pad_w if pad_w else None]
    return flow

def compute_opencv_flow(frame1, frame2, method='farneback'):
    """
    Compute optical flow using OpenCV methods
    
    Args:
        frame1, frame2: numpy arrays [H, W, 3] in BGR format
        method: 'farneback', 'dis', or 'lucaskanade_dense'
    
    Returns:
        flow: [H, W, 2] optical flow (u, v)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    elif method == 'dis':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(gray1, gray2, None)
    elif method == 'lucaskanade_dense':
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=5,
            winsize=13,
            iterations=10,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return flow

def create_quiver_frame(frame, flow_uv, step=16, scale=1, min_mag=0.5, title=None, arrow_color='red'):
    """
    OpenCV로 화살표를 바로 그리는 초고속 버전.
    frame: BGR (H,W,3), flow_uv: (H,W,2)
    """
    H, W = frame.shape[:2]
    Hf, Wf = flow_uv.shape[:2]

    # flow를 프레임 크기에 맞추고 벡터 스케일 보정
    if (Hf != H) or (Wf != W):
        sx = float(W) / float(Wf)
        sy = float(H) / float(Hf)
        u = cv2.resize(flow_uv[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * sx
        v = cv2.resize(flow_uv[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * sy
        flow = np.dstack([u, v])
    else:
        flow = flow_uv

    out = frame.copy()
    # 화살표 색
    color_map = {
        'red':  (0, 0, 255),
        'lime': (0, 255, 0),
        'blue': (255, 0, 0),
        'white':(255, 255, 255),
        'yellow':(0, 255, 255),
    }
    c = color_map.get(arrow_color, (0, 0, 255))

    # 샘플링 그리드
    for y in range(0, H, step):
        for x in range(0, W, step):
            dx = flow[y, x, 0]
            dy = flow[y, x, 1]
            mag = (dx*dx + dy*dy) ** 0.5
            if mag < min_mag:
                continue
            # scale: 값이 클수록 화살표 짧게(원래 함수 의미 유지)
            s = 1.0 / max(scale, 1e-6)
            x2 = int(round(x + dx * s))
            y2 = int(round(y + dy * s))
            cv2.arrowedLine(out, (x, y), (x2, y2), c, thickness=1, tipLength=0.3)

    if title:
        # 얇은 라벨(매트플롯보다 훨씬 가벼움)
        cv2.rectangle(out, (10, 10), (10+len(title)*12, 40), (0, 0, 0), -1)
        cv2.putText(out, title, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def create_side_by_side_comparison(frame, flow_pwc, flow_cv, step=16, scale=1, min_mag=0.5):
    """
    Create side-by-side comparison of PWC-Net and OpenCV flows
    
    Returns:
        numpy array (H, W*2, 3) in BGR format
    """
    # Create individual frames with arrows
    frame_pwc = create_quiver_frame(frame, flow_pwc, step, scale, min_mag, 
                                   title='PWC-Net', arrow_color='lime')
    frame_cv = create_quiver_frame(frame, flow_cv, step, scale, min_mag, 
                                  title='OpenCV', arrow_color='red')
    
    # Stack horizontally
    combined = np.hstack([frame_pwc, frame_cv])
    return combined

# =========================
# 3) Model Loading
# =========================
def load_model(device="cuda"):
    model = PWCDCNet().to(device).eval()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt 
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


    # get model complexity
    net = model
    macs, params = get_model_complexity_info(     
        model=net, input_res=(6, 384, 512),       # model input 크기로 수정하기  
        print_per_layer_stat=True,      
        as_strings=True, 
        verbose=True 
    )
    print(macs, params)

    return model

# =========================
# 4) Video Inference
# =========================
@torch.no_grad()
def process_frame_pair(model, frame1, frame2, device="cuda"):
    """Process a pair of frames and return optical flow"""
    # Convert frames to tensors
    t1 = frame_to_tensor(frame1).unsqueeze(0).to(device)
    t2 = frame_to_tensor(frame2).unsqueeze(0).to(device)
    
    # Pad
    t1p, t2p, pad_h, pad_w = pad_to_multiple_of_64(t1, t2)
    
    # Concatenate and infer
    x = torch.cat([t1p, t2p], dim=1)
    out = model(x)
    
    # Handle different output formats
    if isinstance(out, (list, tuple)):
        flow = out[0]
    elif isinstance(out, dict):
        flow = out.get("flow", next(v for v in out.values() 
                                    if hasattr(v, "shape") and v.dim() == 4 and v.shape[1] == 2))
    else:
        flow = out
    
    flow = unpad(flow, pad_h, pad_w)
    flow_np = flow.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    
    return flow_np

def process_video(input_video, output_video, device="cuda", step=16, scale=1, min_mag=0.5,
                 compare_opencv=False, opencv_method='farneback', output_mode='pwc'):
    """
    Process video and save with arrow overlay showing optical flow
    
    Args:
        input_video: path to input MP4 file
        output_video: path to output MP4 file
        device: 'cuda' or 'cpu'
        step: arrow sampling stride (lower = more arrows)
        scale: arrow scale (higher = shorter arrows)
        min_mag: minimum flow magnitude to show arrow
        compare_opencv: if True, also compute OpenCV flow
        opencv_method: 'farneback', 'dis', or 'lucaskanade_dense'
        output_mode: 'pwc', 'opencv', or 'comparison' (side-by-side)
    """
    # Load model once
    print("Loading model...")
    model = load_model(device)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Arrow parameters: step={step}, scale={scale}, min_mag={min_mag}")
    
    if compare_opencv:
        print(f"OpenCV method: {opencv_method}")
        print(f"Output mode: {output_mode}")
    
    # Setup output video writer
    output_width = width * 2 if output_mode == 'comparison' else width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, height))
    
    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    frame_count = 1
    pbar = tqdm(total=total_frames-1, desc="Processing video")
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        # Process frame pair to get PWC-Net optical flow
        flow_pwc = process_frame_pair(model, frame1, frame2, device)
        
        # Compute OpenCV flow if requested
        if compare_opencv:
            flow_cv = compute_opencv_flow(frame1, frame2, method=opencv_method)
        
        # Create output frame based on mode
        if output_mode == 'pwc':
            output_frame = create_quiver_frame(frame1, flow_pwc, step, scale, min_mag, 
                                              title='PWC-Net', arrow_color='lime')
        elif output_mode == 'opencv':
            output_frame = create_quiver_frame(frame1, flow_cv, step, scale, min_mag,
                                              title=f'OpenCV {opencv_method}', arrow_color='red')
        elif output_mode == 'comparison':
            output_frame = create_side_by_side_comparison(frame1, flow_pwc, flow_cv, 
                                                         step, scale, min_mag)
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")
        
        # Write frame
        out.write(output_frame)
        
        # Move to next frame pair
        frame1 = frame2
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Output saved to: {output_video}")

# =========================
# 5) Main
# =========================
if __name__ == "__main__":

    import argparse
    
    ap = argparse.ArgumentParser(description="PWC-Net Video Optical Flow with Arrow Overlay")
    ap.add_argument("--input", default="../../video/roll0_tilt0_yaw0.MP4", help="Input video file (MP4)")
    ap.add_argument("--output", default=None, help="Output video file")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    ap.add_argument("--step", type=int, default=32, help="Arrow sampling stride (default: 16)")
    ap.add_argument("--scale", type=float, default=1, help="Arrow scale factor (default: 1)")
    ap.add_argument("--min-mag", type=float, default=0.5, help="Minimum flow magnitude to show (default: 0.5)")
    ap.add_argument("--compare-opencv", action='store_true', help="Compare with OpenCV optical flow")
    ap.add_argument("--opencv-method", default='farneback', 
                   choices=['farneback', 'dis', 'lucaskanade_dense'],
                   help="OpenCV optical flow method")
    ap.add_argument("--output-mode", default='pwc',
                   choices=['pwc', 'opencv', 'comparison'],
                   help="Output mode: pwc (PWC-Net only), opencv (OpenCV only), or comparison (side-by-side)")
    args = ap.parse_args()
    



    # Generate output filename if not specified
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
        if args.compare_opencv:
            args.output = f"PWCNet_vs_OpenCV_{args.opencv_method}_{args.output_mode}.mp4"
        else:
            args.output = f"PWCNet_{model_name}_arrows.mp4"
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {CKPT_PATH}")
    print(f"Device: {args.device}")
    
    process_video(
        input_video=args.input,
        output_video=args.output,
        device=args.device,
        step=args.step,
        scale=args.scale,
        min_mag=args.min_mag,
        compare_opencv=args.compare_opencv,
        opencv_method=args.opencv_method,
        output_mode=args.output_mode
    )

