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

# =========================
# 1) Model definition / weights
# =========================
import models
from models.PWCNet import PWCDCNet
from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis, flop_count_table

# TODO: path to weights
CKPT_PATH = "./weights/checkpoints_pseudo_fund_thresh0.5/pwcnet_proxy_epoch_18, loss_0.1497.tar"

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

def estimate_vanishing_point_from_flow(
    flow_uv,
    step=16,
    min_mag=1,
    max_points=300,
    grid_size=64,
    min_pairs=50,
):
    """
    flow_uv: (H, W, 2) optical flow (u, v)
    step:    sampling stride for flow vectors
    min_mag: ignore flow vectors smaller than this
    max_points: max # of flow points used (limit complexity)
    grid_size: 2D histogram grid (grid_size x grid_size)
    min_pairs: min # of line pairs needed

    Returns:
        (vx, vy, prob)  -- estimated vanishing point + vote-based probability
        or None if estimation fails
    """
    H, W, _ = flow_uv.shape

    xs, ys = [], []
    dxs_n, dys_n = [], []
    mags = []

    # 1) Sample flow vectors
    for y in range(0, H, step):
        for x in range(0, W, step):
            dx = float(flow_uv[y, x, 0])
            dy = float(flow_uv[y, x, 1])
            mag = math.hypot(dx, dy)
            if mag < min_mag:
                continue

            # normalize direction
            dx_n = dx / mag
            dy_n = dy / mag

            xs.append(float(x))
            ys.append(float(y))
            dxs_n.append(dx_n)
            dys_n.append(dy_n)
            mags.append(mag)

    N = len(xs)
    if N < 5:
        return None  # too few vectors

    xs = np.array(xs)
    ys = np.array(ys)
    dxs_n = np.array(dxs_n)
    dys_n = np.array(dys_n)
    mags = np.array(mags)

    # 2) Limit number of points for complexity
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        dxs_n = dxs_n[idx]
        dys_n = dys_n[idx]
        mags = mags[idx]
        N = max_points

    # 3) Accumulate pairwise intersections into a 2D histogram
    inter_x = []
    inter_y = []
    inter_w = []

    # small helper for cross products
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    for i in range(N):
        p1x, p1y = xs[i], ys[i]
        d1x, d1y = dxs_n[i], dys_n[i]
        for j in range(i + 1, N):
            p2x, p2y = xs[j], ys[j]
            d2x, d2y = dxs_n[j], dys_n[j]

            # denominator of intersection formula
            denom = cross(d1x, d1y, d2x, d2y)
            if abs(denom) < 1e-6:
                # nearly parallel
                continue

            # t1 = cross(p2 - p1, d2) / cross(d1, d2)
            dp_x = p2x - p1x
            dp_y = p2y - p1y
            t1 = cross(dp_x, dp_y, d2x, d2y) / denom

            ix = p1x + t1 * d1x
            iy = p1y + t1 * d1y

            # only keep intersections reasonably close to the image region
            if -0.5 * W <= ix <= 1.5 * W and -0.5 * H <= iy <= 1.5 * H:
                # weight by magnitude product
                w = mags[i] * mags[j]
                inter_x.append(ix)
                inter_y.append(iy)
                inter_w.append(w)

    if len(inter_x) < min_pairs:
        return None  # not enough intersections

    inter_x = np.array(inter_x)
    inter_y = np.array(inter_y)
    inter_w = np.array(inter_w)

    # 4) Build 2D histogram (vote map)
    # We allow some margin outside the image to catch VPs slightly off-frame.
    x_min, x_max = -0.5 * W, 1.5 * W
    y_min, y_max = -0.5 * H, 1.5 * H

    hist, x_edges, y_edges = np.histogram2d(
        inter_x,
        inter_y,
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]],
        weights=inter_w,
    )

    # 5) Most probable bin
    max_idx = np.argmax(hist)
    if hist.flat[max_idx] <= 0:
        return None

    gx, gy = np.unravel_index(max_idx, hist.shape)
    # center of that bin
    vx = 0.5 * (x_edges[gx] + x_edges[gx + 1])
    vy = 0.5 * (y_edges[gy] + y_edges[gy + 1])

    # 6) Confidence = (votes in best bin) / (total votes)
    total_votes = np.sum(hist)
    prob = float(hist[gx, gy] / (total_votes + 1e-9))

    # 7) Optional refinement with LS only on inlier lines close to this VP
    #    (Keeps your original least-squares idea but centered on the best bin.)
    # Build normals again
    nx = -dys_n
    ny = dxs_n
    c = nx * xs + ny * ys
    A = np.stack([nx, ny], axis=1)
    b = c

    # distances from candidate VP to each line
    candidate = np.array([vx, vy], dtype=np.float32)
    dists = np.abs(A @ candidate - b)
    median_dist = np.median(dists)
    thresh = median_dist * 3.0 + 1e-6
    inliers = dists < thresh

    if inliers.sum() >= 5:
        A_in = A[inliers]
        b_in = b[inliers]
        try:
            sol_refined, _, _, _ = np.linalg.lstsq(A_in, b_in, rcond=None)
            vx, vy = float(sol_refined[0]), float(sol_refined[1])
        except np.linalg.LinAlgError:
            pass  # keep original (vx, vy)

    return (vx, vy, prob)


def create_quiver_frame(frame, flow_uv, step=16, scale=1, min_mag=1,
                        title=None, arrow_color='red',
                        arrow_thickness=1,
                        shrink_ratio=0.75,
                        draw_vanishing_point=True):
    """
    frame: BGR (H,W,3), flow_uv: (H,W,2)
    - shrink_ratio < 1.0 이면 영상이 가운데로 줄어들고 주변은 검은 배경
    - vanishing point는 줄어든 좌표계에 맞춰 같이 스케일링해서 그림
    """
    H, W = frame.shape[:2]
    Hf, Wf = flow_uv.shape[:2]

    # 1) flow를 프레임 크기에 맞추고 벡터 스케일 보정
    if (Hf != H) or (Wf != W):
        sx = float(W) / float(Wf)
        sy = float(H) / float(Hf)
        u = cv2.resize(flow_uv[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * sx
        v = cv2.resize(flow_uv[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * sy
        flow = np.dstack([u, v])
    else:
        flow = flow_uv

    # 2) 출력 캔버스: 전체는 검은색으로 시작
    out = np.zeros_like(frame)

    # shrink_ratio에 맞춰 원본 프레임 축소 후 중앙 배치
    if shrink_ratio < 1.0:
        new_W = int(W * shrink_ratio)
        new_H = int(H * shrink_ratio)
        # 최소 1픽셀 보장
        new_W = max(new_W, 1)
        new_H = max(new_H, 1)

        small_frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        offset_x = (W - new_W) // 2
        offset_y = (H - new_H) // 2
        out[offset_y:offset_y+new_H, offset_x:offset_x+new_W] = small_frame

        # 좌표 스케일링용
        s = float(new_W) / float(W)  # = float(new_H)/H (동일 비율 가정)
        scale_x = s
        scale_y = s
    else:
        # shrink 없음: 원본 그대로
        out[:] = frame
        offset_x = 0
        offset_y = 0
        scale_x = 1.0
        scale_y = 1.0

    # 색
    color_map = {
        'red':     (0, 0, 255),
        'lime':    (0, 255, 0),
        'blue':    (255, 0, 0),
        'white':   (255, 255, 255),
        'yellow':  (0, 255, 255),
        'magenta': (255, 0, 255),
        'cyan':    (255, 255, 0),
    }
    c = color_map.get(arrow_color, (0, 0, 255))

    # 3) 화살표 그리기 (원본 좌표계 -> 축소 좌표계로 스케일링)
    for y in range(0, H, step):
        for x in range(0, W, step):
            dx = float(flow[y, x, 0])
            dy = float(flow[y, x, 1])
            mag = math.hypot(dx, dy)
            if mag < min_mag:
                continue

            s_vec = 1.0 / max(scale, 1e-6)
            x_tip = x + dx * s_vec
            y_tip = y + dy * s_vec

            # 축소 + 중앙 offset 적용 (x, y 둘 다)
            x_s  = int(round(offset_x + x * scale_x))
            y_s  = int(round(offset_y + y * scale_y))
            x2_s = int(round(offset_x + x_tip * scale_x))
            y2_s = int(round(offset_y + y_tip * scale_y))

            # 화면 범위 체크
            if not (0 <= x_s < W and 0 <= y_s < H and 0 <= x2_s < W and 0 <= y2_s < H):
                continue

            cv2.arrowedLine(
                out,
                (x_s, y_s),
                (x2_s, y2_s),
                c,
                thickness=arrow_thickness,  # 🔥 두께
                tipLength=0.3
            )

    # 4) Vanishing point 추정 & 그리기
    if draw_vanishing_point:
        vp = estimate_vanishing_point_from_flow(flow, step=step, min_mag=min_mag)
        if vp is not None:
            # now vp = (vx, vy, prob)
            vx, vy, prob = vp
            if np.isfinite(vx) and np.isfinite(vy):
                # 동일한 축소/offset 변환 적용
                vx_s = int(round(offset_x + vx * scale_x))
                vy_s = int(round(offset_y + vy * scale_y))
                if 0 <= vx_s < W and 0 <= vy_s < H:
                    # 노란 동그라미 + 십자 표시
                    cv2.circle(out, (vx_s, vy_s), 8, (0, 255, 255), thickness=3)
                    cv2.line(out, (vx_s - 15, vy_s), (vx_s + 15, vy_s), (0, 255, 255), 2)
                    cv2.line(out, (vx_s, vy_s - 15), (vx_s, vy_s + 15), 2)

                    # 확률(혹은 신뢰도) 텍스트로 표기
                    text = f"p={prob:.2f}"
                    cv2.putText(out, text, (vx_s + 10, vy_s - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2, cv2.LINE_AA)

    # 5) 타이틀
    if title:
        cv2.rectangle(out, (10, 10), (10 + len(title) * 12, 40), (0, 0, 0), -1)
        cv2.putText(out, title, (14, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return out





def create_side_by_side_comparison(frame, flow_pwc, flow_cv, step=16, scale=1, min_mag=1):
    frame_pwc = create_quiver_frame(frame, flow_pwc, step, scale, min_mag, 
                                   title='PWC-Net', arrow_color='lime',
                                   arrow_thickness=3,
                                   draw_vanishing_point=True)
    frame_cv = create_quiver_frame(frame, flow_cv, step, scale, min_mag, 
                                  title='OpenCV', arrow_color='red',
                                  arrow_thickness=3,
                                  draw_vanishing_point=True)
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

def process_video(input_video, output_video, device="cuda", step=16, scale=1, min_mag=1,
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
    ap.add_argument("--input", default="../../video/roll1_tilt1_yaw-2.MP4", help="Input video file (MP4)")
    ap.add_argument("--output", default=None, help="Output video file")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    ap.add_argument("--step", type=int, default=16, help="Arrow sampling stride (default: 16)")
    ap.add_argument("--scale", type=float, default=0.2, help="Arrow scale factor (default: 1)")
    ap.add_argument("--min-mag", type=float, default=1.0, help="Minimum flow magnitude to show (default: 0.5)")
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
            args.output = f"./output/PWCNet_vs_OpenCV_{args.opencv_method}_{args.output_mode}.mp4"
        else:
            args.output = f"./output/PWCNet_{model_name}_roll1_tilt1_yaw-2.mp4"
    
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