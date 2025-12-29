import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda
import torch.nn.functional as F
import torch.nn as nn

# ONNX export 시에만 True로 켜줄 전역 플래그
USE_ONNX_CORRELATION = False


def correlation_onnx(input1, input2,
                     pad_size, kernel_size, max_displacement,
                     stride1, stride2, corr_multiply):
    """
    ONNX export용 순수 PyTorch correlation.
    input1, input2: (B, C, H, W)
    """
    B, C, H, W = input1.shape

    # 간단히 pad_size + max_displacement 만큼 패딩
    pad = pad_size + max_displacement
    # (left, right, top, bottom)
    input2_padded = F.pad(input2, (pad, pad, pad, pad))

    outs = []
    # -d ~ +d, step=stride2 로 모든 offset에 대해 correlation 계산
    for oy in range(-max_displacement, max_displacement + 1, stride2):
        for ox in range(-max_displacement, max_displacement + 1, stride2):
            y_start = pad + oy
            x_start = pad + ox

            patch2 = input2_padded[:, :, y_start:y_start + H, x_start:x_start + W]
            # 채널 방향으로 곱하고 합 → 상관 값
            corr = (input1 * patch2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            corr = corr * corr_multiply
            outs.append(corr)

    out = torch.cat(outs, dim=1)  # (B, num_displacements, H, W)
    return out


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):

        if torch.onnx.is_in_onnx_export():
            # ONNX export 중에는 CUDA 커널 쓰지 말고
            # 순수 PyTorch 구현으로 바로 리턴
            with torch.no_grad():
                return correlation_onnx(input1, input2,
                                        pad_size, kernel_size, max_displacement,
                                        stride1, stride2, corr_multiply)

        ctx.save_for_backward(input1, input2)

        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(nn.Module):
    def __init__(self, pad_size, kernel_size, max_displacement,
                 stride1, stride2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # 🔥 ONNX export 모드일 땐 무조건 PyTorch 구현 사용
        if USE_ONNX_CORRELATION:
            return correlation_onnx(
                input1, input2,
                self.pad_size, self.kernel_size, self.max_displacement,
                self.stride1, self.stride2, self.corr_multiply
            )

        # 평상시에는 기존 CUDA 커널 그대로 사용
        return CorrelationFunction.apply(
            input1, input2,
            self.pad_size, self.kernel_size, self.max_displacement,
            self.stride1, self.stride2, self.corr_multiply
        )


