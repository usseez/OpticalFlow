# CUDA 12.4, cuDNN, Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL maintainer="hirakawat"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      sudo build-essential cmake g++ gfortran pkg-config \
      libhdf5-dev ninja-build \
      wget curl git htop tmux vim ffmpeg rsync openssh-server \
      python3 python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# CUDA paths
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_HOME=${CUDA_ROOT}
ENV PATH=${PATH}:${CUDA_ROOT}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_ROOT}/lib64:/usr/local/nvidia/lib64
ENV LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_ROOT}/lib64

# Python + PyTorch (cu121 wheel)
RUN python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install --no-cache-dir \
      numpy scipy matplotlib seaborn scikit-learn scikit-image pillow requests \
      jupyterlab networkx h5py pandas plotly protobuf tqdm tensorboardX colorama setproctitle && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

# 빌드 환경
ENV TORCH_CUDA_ARCH_LIST="8.6" \
    USE_NINJA=1 \
    MAX_JOBS=8

# 빌드 컨텍스트를 /home/ubuntu/Works 로 가정하고, 같은 절대경로로 복사
#   $ cd /home/ubuntu/Works
#   $ docker build -t flownet2:cuda12.4 -f Dockerfile .
RUN mkdir -p /home/ubuntu/Works/CALIBRATION/OpticalFlow/flownet2-pytorch
COPY ./ /home/ubuntu/Works/CALIBRATION/OpticalFlow/flownet2-pytorch

# 빌드 옵션 (GPU에 맞게 수정 가능)
ENV TORCH_CUDA_ARCH_LIST="8.6" USE_NINJA=1 MAX_JOBS=8

# 각 패키지 디렉터리로 이동해서 전역 설치 (cd 대신 WORKDIR)
# correlation_package
WORKDIR /home/ubuntu/Works/CALIBRATION/OpticalFlow/flownet2-pytorch/networks/correlation_package
RUN rm -rf *_cuda.egg-info build dist __pycache__ && \
    python3 -m pip install -v --no-cache-dir --no-build-isolation .

# resample2d_package
WORKDIR /home/ubuntu/Works/CALIBRATION/OpticalFlow/flownet2-pytorch/networks/resample2d_package
RUN rm -rf *_cuda.egg-info build dist __pycache__ && \
    python3 -m pip install -v --no-cache-dir --no-build-isolation .

# channelnorm_package
WORKDIR /home/ubuntu/Works/CALIBRATION/OpticalFlow/flownet2-pytorch/networks/channelnorm_package
RUN rm -rf *_cuda.egg-info build dist __pycache__ && \
    python3 -m pip install -v --no-cache-dir --no-build-isolation .

# 버전 출력
WORKDIR /home/ubuntu/Works/CALIBRATION/
RUN python3 - <<'PY'
import torch, sys
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version reported by torch:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
PY
