import cv2
import os

def save_frame_as_png(video_path: str, frame_number: int, output_path: str):
    """
    mp4(또는 다른 영상)에서 특정 프레임을 PNG로 저장합니다.

    video_path   : 입력 비디오 파일 경로
    frame_number : 캡처할 프레임 번호 (0부터 시작)
    output_path  : 저장할 PNG 파일 경로 (예: 'frame_100.png')
    """
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오를 열 수 없습니다: {video_path}")

    # 전체 프레임 수 확인 (옵션)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        raise ValueError(
            f"frame_number가 범위를 벗어났습니다. "
            f"(요청: {frame_number}, 총 프레임 수: {total_frames})"
        )

    # 원하는 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"{frame_number}번 프레임을 읽는 데 실패했습니다.")

    # PNG로 저장 (OpenCV는 BGR → PNG 저장 시 자동 처리)
    # output 경로 상위 폴더 없으면 생성
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    success = cv2.imwrite(output_path, frame)
    cap.release()

    if not success:
        raise IOError(f"이미지를 저장하는 데 실패했습니다: {output_path}")

    print(f"저장 완료: {output_path} (frame {frame_number})")


if __name__ == "__main__":
    # 사용 예시
    video_file = "./output/00pwc_pseudo.mp4"     # 입력 비디오 경로
    frame_num = 10               # 캡처하고 싶은 프레임 번호 (0부터 시작)
    output_file = f"output/{video_file}frame_{frame_num}.png"  # 저장할 PNG 경로

    save_frame_as_png(video_file, frame_num, output_file)
