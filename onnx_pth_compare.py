"""compare the result of onnx and pth model overlay on the same image......"""
import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

from models.PWCNet import PWCDCNet
from models.correlation_package import correlation as corr_mod  # 우리가 수정했던 그 모듈


CKPT_PATH = "./weights/checkpoints_pseudo_fund/pwcnet_proxy_epoch_85.tar"
# ONNX_PATH = "./weights/checkpoints_pseudo_fund/pwcnet_proxy_epoch_85.onnx"
ONNX_PATH = "./pwcnet_proxy_epoch_85.onnx"


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# 1) optical flow를 컬러 이미지로 변환 (간단 버전)
def flow_to_color(flow):
    """
    flow: (H, W, 2)
    """
    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)

    hsv[..., 0] = ang / (2 * np.pi)  # [0,1] 범위 hue
    hsv[..., 1] = 1.0
    if mag.max() > 1e-6:
        hsv[..., 2] = mag / (mag.max() + 1e-6)
    else:
        hsv[..., 2] = 0.0

    hsv_uint8 = (hsv * 255).astype(np.uint8)
    hsv_bgr = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)
    return rgb


# 2) 이미지 두 장을 읽어서 (H,W,3) 두 개 → (1,6,H,W) 텐서로 만들기
def load_image_pair_to_6ch_tensor(img1_path, img2_path, resize_hw=(384, 512)):
    # 1) 이미지 로드 (BGR → RGB)
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)
    assert im1 is not None and im2 is not None, "이미지 경로 다시 확인!"

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # 2) 학습 때와 동일한 Resize (H, W) = (384, 512)
    if resize_hw is not None:
        H, W = resize_hw
        im1 = cv2.resize(im1, (W, H), interpolation=cv2.INTER_LINEAR)
        im2 = cv2.resize(im2, (W, H), interpolation=cv2.INTER_LINEAR)

    # overlay용으로는 "정규화 안 된" RGB uint8 이미지 한 장 보관
    base_img = im1.copy()   # (H, W, 3), uint8, RGB

    # 3) ToTensor와 동일한 동작: [0,255] → [0,1], HWC → CHW
    im1_f = im1.astype(np.float32) / 255.0   # (H,W,3)
    im2_f = im2.astype(np.float32) / 255.0

    im1_chw = np.transpose(im1_f, (2, 0, 1))  # (3,H,W)
    im2_chw = np.transpose(im2_f, (2, 0, 1))

    # 4) Normalize(mean, std) 적용 (채널별)
    mean = IMAGENET_MEAN[:, None, None]      # (3,1,1)
    std  = IMAGENET_STD[:, None, None]       # (3,1,1)

    im1_norm = (im1_chw - mean) / std
    im2_norm = (im2_chw - mean) / std

    # 5) concat → (6,H,W), batch 차원 추가 → (1,6,H,W)
    im6 = np.concatenate([im1_norm, im2_norm], axis=0)  # (6,H,W)
    im6 = np.expand_dims(im6, axis=0)                   # (1,6,H,W)

    return im6.astype(np.float32), base_img


def run_pytorch_pwcnet(tensor_6ch):
    """
    tensor_6ch: (1,6,H,W) numpy → torch.Tensor
    """
    # 🔥 ONNX export 때랑 같은 correlation 경로를 쓰고 싶으면:
    corr_mod.USE_ONNX_CORRELATION = True   # 우리가 correlation.py에 만든 플래그
    # 만약 CUDA correlation과 비교하고 싶으면 False로 두고 비교해봐도 됨

    model = PWCDCNet()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    x = torch.from_numpy(tensor_6ch).to(DEVICE)
    with torch.no_grad():
        out = model(x)  # pwcnet 구현에 따라 튜플/리스트면 첫 번째만 사용

    if isinstance(out, (list, tuple)):
        flow = out[0]
    else:
        flow = out

    # (1,2,H,W) → (H,W,2)
    flow_np = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
    return flow_np


def run_onnx_pwcnet(tensor_6ch):
    """
    tensor_6ch: (1,6,H,W) numpy
    """
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])  ##model 파일(onnx) 열기

    # input / output 이름 확인
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    outputs = sess.run([output_name], {input_name: tensor_6ch.astype(np.float32)})
    flow = outputs[0]  # (1,2,H,W) 가정
    flow_np = np.transpose(flow[0], (1, 2, 0))  # (H,W,2)
    return flow_np


def compare_flows(flow_pth, flow_onnx):
    assert flow_pth.shape == flow_onnx.shape

    a = flow_pth.astype(np.float32)
    b = flow_onnx.astype(np.float32)

    # 1) 절대적인 차이
    diff = a - b
    l2 = np.linalg.norm(diff.ravel())                     # L2 norm of diff
    mae = np.mean(np.abs(diff))                           # MAE
    max_abs = np.max(np.abs(diff))                        # max |diff|

    # 2) 상대적인 차이 (scale 고려)
    l2_ref  = np.linalg.norm(a.ravel())                   # 기준: pth
    mae_ref = np.mean(np.abs(a)) + 1e-8

    rel_l2  = l2 / (l2_ref + 1e-8)                        # 상대 L2
    rel_mae = mae / mae_ref                               # 상대 MAE

    # 3) Pearson 상관계수 (전체 값을 1D로 펼쳐서)
    a_flat = a.ravel()
    b_flat = b.ravel()
    corr = np.corrcoef(a_flat, b_flat)[0, 1]              # -1 ~ 1, 1에 가까울수록 좋음

    # 4) 코사인 유사도 (방향이 비슷한지)
    cos_sim = np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8
    )                                                     # -1 ~ 1, 1에 가까울수록 좋음

    # 5) optical flow 전용 지표: Endpoint Error (EPE)
    #    각 픽셀마다 flow 벡터 (u,v)의 차이의 L2 norm
    vec_diff = np.linalg.norm(a - b, axis=-1)             # (H, W)
    epe_mean = np.mean(vec_diff)
    epe_max  = np.max(vec_diff)

    # 🔹 일치율 (%): EPE가 tau 이하인 픽셀 비율
    taus = [0.25, 0.5, 1.0, 2.0]

    agreements = {}
    for tau in taus:
        agree = (vec_diff <= tau).mean() * 100.0
        agreements[f"agree@{tau}"] = agree


    print(f"L2 diff           : {l2}")
    print(f"MAE               : {mae}")
    print(f"Max abs diff      : {max_abs}")
    print(f"Relative L2 diff  : {rel_l2}")
    print(f"Relative MAE      : {rel_mae}")
    print(f"Pearson corr      : {corr}")
    print(f"Cosine similarity : {cos_sim}")
    print(f"EPE mean          : {epe_mean}")
    print(f"EPE max           : {epe_max}")

    for tau in taus:
        print(f"Agreement @ EPE<={tau:4.2f} : {agreements[f'agree@{tau}']:6.2f}%")

    return {
        "L2 diff": l2,
        "MAE": mae,
        "Max abs diff": max_abs,
        "Relative L2 diff": rel_l2,
        "Relative MAE": rel_mae,
        "Pearson corr": corr,
        "Cosine similarity": cos_sim,
        "EPE mean": epe_mean,
        "EPE max": epe_max,
        "agreements": agreements,
    }


def overlay_flow_on_image(img, flow_color, alpha=0.6):
    # 혹시 grayscale이면 컬러로 변환
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(flow_color.shape) == 2:
        flow_color = cv2.cvtColor(flow_color, cv2.COLOR_GRAY2BGR)

    # 1) 크기 맞추기  (cv2.resize는 (width, height) 순서!)
    if img.shape[:2] != flow_color.shape[:2]:
        h, w = img.shape[:2]
        flow_color = cv2.resize(flow_color, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) dtype 맞추기 (보통 uint8)
    if img.dtype != flow_color.dtype:
        flow_color = flow_color.astype(img.dtype)

    # 3) 오버레이
    overlay = cv2.addWeighted(img, 1 - alpha, flow_color, alpha, 0)
    return overlay

#debug
def print_flow_stats(name, flow):
    print(f"[{name}] shape={flow.shape}")
    print(f"  min  : {flow.min():.4f}")
    print(f"  max  : {flow.max():.4f}")
    print(f"  mean : {flow.mean():.4f}")
    print(f"  std  : {flow.std():.4f}")

    

if __name__ == "__main__":
    # 예시 이미지 경로 (네 데이터셋에 맞게 수정)
    img1_path = "../Dataset/test/roll0_tilt-3_yaw-6_0043.png"
    img2_path = "../Dataset/test/roll0_tilt-3_yaw-6_0048.png"

    # 1) 입력 준비
    input_6ch, base_img = load_image_pair_to_6ch_tensor(img1_path, img2_path)
    print("input shape:", input_6ch.shape)  # (1,6,H,W)

    # 2) PyTorch 모델 추론
    flow_pth = run_pytorch_pwcnet(input_6ch)
    print("flow_pth shape:", flow_pth.shape)

    # 3) ONNX 모델 추론
    flow_onnx = run_onnx_pwcnet(input_6ch)
    print("flow_onnx shape:", flow_onnx.shape)

    # 4) 수치 비교
    metrics = compare_flows(flow_pth, flow_onnx)






    # 5) 컬러로 만들고 overlay
    color_pth = flow_to_color(flow_pth)      # RGB
    color_onnx = flow_to_color(flow_onnx)    # RGB

    overlay_pth = overlay_flow_on_image(base_img, color_pth, alpha=0.6)      # RGB
    overlay_onnx = overlay_flow_on_image(base_img, color_onnx, alpha=0.6)    # RGB

    # 6) OpenCV로 파일 저장 (BGR 변환 후)
    save_dir = "./results_overlay"
    os.makedirs(save_dir, exist_ok=True)

    # OpenCV는 BGR이므로 RGB → BGR 변환
    overlay_pth_bgr = cv2.cvtColor(overlay_pth, cv2.COLOR_RGB2BGR)
    overlay_onnx_bgr = cv2.cvtColor(overlay_onnx, cv2.COLOR_RGB2BGR)
    color_pth_bgr = cv2.cvtColor(color_pth, cv2.COLOR_RGB2BGR)
    color_onnx_bgr = cv2.cvtColor(color_onnx, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(os.path.join(save_dir, "overlay_pth.png"), overlay_pth_bgr)
    # cv2.imwrite(os.path.join(save_dir, "overlay_onnx.png"), overlay_onnx_bgr)
    # cv2.imwrite(os.path.join(save_dir, "flow_color_pth.png"), color_pth_bgr)
    # cv2.imwrite(os.path.join(save_dir, "flow_color_onnx.png"), color_onnx_bgr)

    # ==============================
    # 8) 한 장짜리 리포트 이미지 만들기
    # ==============================
    # 4장 이미지가 모두 같은 크기라고 가정 (overlay 함수에서 맞춰줬으니까)
    h, w, c = overlay_pth_bgr.shape

    # 2x2 그리드 크기
    grid_h = 2 * h
    grid_w = 2 * w

    # 오른쪽 여백 폭 (필요하면 조절)
    margin_w = 400

    # 전체 캔버스 (검정 배경)
    canvas_h = grid_h
    canvas_w = grid_w + margin_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    
    color_pth_bgr_big  = cv2.resize(color_pth_bgr,  (w, h), interpolation=cv2.INTER_LINEAR)
    color_onnx_bgr_big = cv2.resize(color_onnx_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    # 그리드에 4장 배치
    # 좌측 상단
    canvas[0:h, 0:w, :] = overlay_pth_bgr
    # 우측 상단
    canvas[0:h, w:2*w, :] = overlay_onnx_bgr
    # 좌측 하단
    canvas[h:2*h, 0:w, :] = color_onnx_bgr_big
    # 우측 하단
    canvas[h:2*h, w:2*w, :] = color_onnx_bgr_big

    # ==============================
    # 9) 오른쪽 여백에 텍스트 쓰기
    # ==============================
    text_x = grid_w + 10      # 그리드 오른쪽 + 10px
    text_y = 30               # 시작 높이
    line_height = 35          # 줄 간격

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)       # 초록색 (BGR)

    lines = [
        f"flow_pth shape : {flow_pth.shape}",
        f"flow_onnx shape: {flow_onnx.shape}",
        "",
        f"L2 diff           : {metrics['L2 diff']:.6f}",
        f"MAE               : {metrics['MAE']:.6f}",
        f"Max abs diff      : {metrics['Max abs diff']:.6f}",
        f"Relative L2 diff  : {metrics['Relative L2 diff']:.6f}",
        f"Relative MAE      : {metrics['Relative MAE']:.6f}",
        f"Pearson corr      : {metrics['Pearson corr']:.6f}",
        f"Cosine similarity : {metrics['Cosine similarity']:.6f}",
        f"EPE mean          : {metrics['EPE mean']:.6f}",
        f"EPE max           : {metrics['EPE max']:.6f}",
    ]

    for i, line in enumerate(lines):
        y = text_y + i * line_height
        cv2.putText(canvas, line, (text_x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # print(f"Saved results to: {os.path.abspath(save_dir)}")
    single_path = os.path.join(save_dir, "comparison_report.png")

    print_flow_stats("pth", flow_pth)
    print_flow_stats("onnx", flow_onnx)

    cv2.imwrite(single_path, canvas)
    print("Saved combined report image to:", os.path.abspath(single_path))