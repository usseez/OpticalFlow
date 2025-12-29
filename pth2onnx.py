from loguru import logger
import sys
import torch
from torch import nn
import onnx
from onnxsim import simplify
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))
from models.PWCNet import PWCDCNet # model import
from models.correlation_package import correlation as corr_mod

import torch.onnx

def make_parser():
    parser = argparse.ArgumentParser("Resnet onnx deploy")
    parser.add_argument(
        "-t", "--model_type", default="pwcdnet", type=str, help="input model type of resnet model"
    )
    parser.add_argument(
        "--output_name", type=str, default="pwcdnet.onnx", help="output name of models"
    )
    parser.add_argument(
        "-i", "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "-o", "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "--opset", default=13, type=int, help="onnx opset version"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    
    
    return parser

@logger.catch
def main(): 
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = 'cpu'
    corr_mod.USE_ONNX_CORRELATION = True

    if args.model_type == 'pwcdnet':
        model = PWCDCNet()
    
    else:
        print('Unknown model type!')
        sys.exit(1)

    # 체크포인트 로드
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    
    logger.info("loading checkpoint done.")
    
    # 입력 이미지 크기에 맞춰 더미 이미지 생성
    dummy_image = torch.randn([1, 6, 256, 256])
    # dummy_image = torch.randn([1, 6, 256, 256], device=device)


    dummy_image = dummy_image.to(device)
    model.eval()
    model = model.to(device)
    print(dummy_image.shape)

    # onnx_program = torch.onnx.dynamo_export(model, dummy_image)
    # onnx_program.save(args.output_name)
    # logger.info(f"onnx saved to {args.output_name}")

    with torch.no_grad():  # gradient 비활성화
        torch.onnx.export(
            model,
            dummy_image,
            args.output_name,
            opset_version=args.opset,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={
                args.input: {0: 'batch', 2: 'height', 3: 'width'},
                args.output: {0: 'batch', 2: 'height', 3: 'width'}
            }
        )

    logger.info("generated onnx model named {}".format(args.output_name))
    if not args.no_onnxsim:
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "generated onnx model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    # ONNX 모델 단순화
    onnx_model = onnx.load(args.output_name)
    model_simp, check = simplify(onnx_model)
    assert check, "generated onnx model could not be validated"
    onnx.save(model_simp, args.output_name)
    logger.info("generated simplified onnx model named {}".format(args.output_name))

if __name__ == "__main__":
    main()
    

