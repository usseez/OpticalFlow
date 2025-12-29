import cv2
import torch
import numpy as np
import os
import sys
from pathlib import Path

# PWC-Net 모델 import (실제 모델 파일 경로에 맞게 조정)
# from models.PWCNet import pwc_dc_net
# 또는 직접 정의
import torch.nn as nn
from torch.autograd import Variable
from models.PWCNet import PWCDCNet

from correlation_package.correlation import Correlation
# from .correlation_package.correlation import Correlation
from model.correlation_package.correlation import Correlation


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)




def load_model(model_path):
    """모델 로드"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PWCDCNet().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"모델 로드 완료: {model_path} (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"모델 로드 완료: {model_path}")
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
    
    model.eval()
    return model, device


def get_perspective_matrix(width, height):
    """Sidecam to Top-view 변환 매트릭스"""
    # 원본 영상의 4개 포인트 (sidecam) - 실제 영상에 맞게 조정 필요
    src_points = np.float32([
        [width * 0.2, height * 0.8],   # 좌하단
        [width * 0.8, height * 0.8],   # 우하단
        [width * 0.3, height * 0.4],   # 좌상단
        [width * 0.7, height * 0.4]    # 우상단
    ])
    
    # 변환될 목표 포인트 (top-view)
    dst_points = np.float32([
        [width * 0.2, height * 0.9],   # 좌하단
        [width * 0.8, height * 0.9],   # 우하단
        [width * 0.2, height * 0.1],   # 좌상단
        [width * 0.8, height * 0.1]    # 우상단
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix


def compute_flow(model, device, frame1, frame2):
    """Optical flow 계산"""
    h, w = frame1.shape[:2]
    
    # 전처리
    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
    # 두 이미지를 concat
    img_pair = torch.cat([img1, img2], dim=1).to(device)
    
    # Flow 계산
    print('img1: ', img1.shape)
    print('img1: ', img2.shape)
    print('img_pair: ', img_pair.shape)
    with torch.no_grad():
        flow = model(img_pair)
        print('flow: ', flow.shape)
    
    # [B, 2, H, W] -> [H, W, 2]
    flow = flow[0].permute(1, 2, 0).cpu().numpy()

    # 원본 크기로 리사이즈
    flow = cv2.resize(flow, (w, h))
    # 스케일 조정
    flow[:, :, 0] *= w / flow.shape[1]
    flow[:, :, 1] *= h / flow.shape[0]
    
    return flow


def calculate_dominant_direction(flow, threshold=1.0):
    """주요 흐름 방향 계산"""
    # flow의 평균 방향 계산
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    valid_mask = magnitude > threshold
    
    if not valid_mask.any():
        return np.array([0, 0])
    
    valid_flow = flow[valid_mask]
    mean_flow = np.mean(valid_flow, axis=0)
    
    return mean_flow


def draw_flow_arrows(frame, flow, step=20, scale=5.0, dominant_dir=None, angle_threshold=30):
    """Flow를 화살표로 그리기"""
    h, w = frame.shape[:2]
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            
            # 벡터의 크기 계산
            magnitude = np.sqrt(fx**2 + fy**2)
            
            # 너무 작은 벡터는 무시
            if magnitude < 0.5:
                continue
            
            # 화살표 끝점 계산 (크기를 크게)
            end_x = int(x + fx * scale)
            end_y = int(y + fy * scale)
            
            # 현재 벡터의 방향과 dominant direction의 각도 차이 계산
            if dominant_dir is not None and np.linalg.norm(dominant_dir) > 0:
                # 벡터 정규화
                current_vec = np.array([fx, fy]) / magnitude
                dominant_vec = dominant_dir / np.linalg.norm(dominant_dir)
                
                # 코사인 유사도로 각도 차이 계산
                cos_angle = np.dot(current_vec, dominant_vec)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_diff = np.arccos(cos_angle) * 180 / np.pi
                
                # 같은 방향이면 빨강, 다른 방향이면 하양
                if angle_diff < angle_threshold:
                    color = (0, 0, 255)  # 빨강 (BGR)
                else:
                    color = (255, 255, 255)  # 하양 (BGR)
            else:
                color = (0, 0, 255)  # 기본 빨강
            
            # 화살표 그리기 (두께 2)
            cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
    
    return frame


def process_video(input_path, output_path, model_path):
    """비디오 처리 메인 함수"""
    # 모델 로드
    model, device = load_model(model_path)
    
    # 비디오 캡처
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {input_path}")
        return
    
    # 비디오 속성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"비디오 정보: {width}x{height}, {fps}fps, {total_frames} 프레임")
    
    # 출력 경로 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 비디오 라이터
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Perspective 변환 매트릭스
    M = get_perspective_matrix(width, height)
    
    # 첫 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_frame_warped = cv2.warpPerspective(prev_frame, M, (width, height))
    
    frame_count = 0
    
    print(f"비디오 처리 시작...")
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Top-view 변환
        curr_frame_warped = cv2.warpPerspective(curr_frame, M, (width, height))
        
        # Optical flow 계산
        flow = compute_flow(model, device, prev_frame_warped, curr_frame_warped)
        
        # Dominant direction 계산
        dominant_dir = calculate_dominant_direction(flow)
        
        # 화살표 그리기
        result_frame = curr_frame_warped.copy()
        result_frame = draw_flow_arrows(result_frame, flow, 
                                       step=20, scale=5.0, 
                                       dominant_dir=dominant_dir,
                                       angle_threshold=30)
        
        # 프레임 저장
        out.write(result_frame)
        
        # 진행상황 출력
        if frame_count % 30 == 0:
            print(f"처리 중... {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
        
        # 다음 프레임 준비
        prev_frame_warped = curr_frame_warped
    
    # 정리
    cap.release()
    out.release()
    
    print(f"\n처리 완료! 출력 파일: {output_path}")
    print(f"총 {frame_count} 프레임 처리됨")


if __name__ == "__main__":
    model_path = "./checkpoints_pseudo/pwcnet_proxy_epoch_50.tar"
    input_path = "../../video/roll0_tilt0_yaw0.MP4"
    output_path = "./output/topview_flow_visualization.mp4"
    
    process_video(input_path, output_path, model_path)