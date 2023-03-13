import sys

import torch
import detect

sys.path.insert(0, './yolov5')
from yolov5.utils.general import check_img_size
from yolov5.utils.dataloaders import LoadImages
from yolov5.models.common import DetectMultiBackend

if __name__ == '__main__':
    parser = detect.decter_init()
    args = parser.parse_args()
    # 验证输入图像尺寸是否为32的倍数
    args.img_size = check_img_size(args.img_size)
    print(args)
    model = DetectMultiBackend(weights=args.weights,
                               device=args.device,
                               dnn=False,
                               fp16=False)
    model.to(args.device).eval()

    stride, names, pt = model.stride, model.names, model.pt
    dataset = LoadImages(args.source,
                         img_size=args.img_size,
                         stride=stride,
                         auto=pt,
                         vid_stride=1)
    detect.detect(args, model, dataset)
