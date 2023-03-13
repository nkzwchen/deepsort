import argparse
import sys
import time
import platform
import os
import math

import cv2
from pathlib import Path
import torch

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

sys.path.insert(0, './yolov5')
from yolov5.utils.general import (non_max_suppression, scale_boxes, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box

def decter_init():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path')  # 加载的v5权重，可以是公开数据集的预训练权重，也可以是自己数据集训练的权重
    parser.add_argument('--weights',
                        type=str,
                        default='model/best.pt',
                        help='model.pt path')
    parser.add_argument('--source',
                        type=str,
                        default='data/test01.mp4',
                        help='source')  # 待检测的视频路径
    parser.add_argument('--output',
                        type=str,
                        default='output/output01.mp4',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.5,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.5,
                        help='IOU threshold for NMS')
    parser.add_argument('--fourcc',
                        type=str,
                        default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device',
                        default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='filter by class')  # car
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort",
                        type=str,
                        default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()

    return parser

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
def box_centers_f(outputs):
    box_centers = []
    for i, each_box in enumerate(outputs):
        # 求得每个框的中心点
        box_centers.append([(each_box[0] + each_box[2]) / 2,
                            (each_box[1] + each_box[3]) / 2, each_box[4],
                            each_box[5], each_box[2] - each_box[0]])
    return (box_centers)

def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])  # 获得当前帧中跟踪到车辆的ID
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])  # 获得前一帧中跟踪到车辆的ID
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:  # 将当前帧有效检测车辆的信息存入work_locations中
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:  # 将前一帧有效检测车辆的ID索引存入work_IDs_prev_index中
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:  # 将前一帧有效检测车辆的信息存入work_prev_locations中
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]
                       )**2 +  # 计算有效检测车辆的速度，采用线性的从像素距离到真实空间距离的映射
                      (work_locations[i][1] - work_prev_locations[i][1])**2)
            *  # 当视频拍摄视角并不垂直于车辆移动轨迹时，测算出来的速度将比实际速度低
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 *
            3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]
                    ]  # 将保留一位小数的单位为km/h的车辆速度及其ID存入speed二维列表中
    return speed

def draw_speed(img, speed, bbox_xywh, identities):
    for i, j in enumerate(speed):
        for m, n in enumerate(identities):
            if j[1] == n:
                xy = [int(i) for i in bbox_xywh[m]]
                cv2.putText(img,
                            str(j[0]) + 'km/h', (xy[0], xy[1] - 7),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255], 2)
                break


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, classes2, identities=None):
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i] * 100))
        label = '%d %s' % (id, cls_names[i])
        # label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4),
                      color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def detect(opt, model, dataset):
    
    # 获取输出文件夹，输入源，权重，参数等参数    
    out, source, _, _, _, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    ### 保存路径 视频源 模型权重 是否观察图片 是否存储txt 图片大小
 
    ### 异常判断代码 无用
    # 获得视频的帧宽高
    capture = cv2.VideoCapture(source)
    ### 读取视频
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ### 获取视频的长和宽   
    # deepsort模块初始化
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=False)
    
    # 从加载好的模型里面读取names模块，为类别信息
    names = model.module.names if hasattr(model, 'module') else model.names
    # class_name = dict(zip(list(range(len(names))), names))
    
    # 设置每种车型的真实车宽
    # width = [1.85, 2.3, 2.5]  # car、bus、truck，单位m
    width = [1.85]  # car、bus、truck，单位m
    locations = []
    speed = []

    t0 = time.time()  # 系统时钟的时间戳，ti-t0即为t0到ti之间这段程序运行的系统时间（在程序并发执行时,该时间并非该程序的精确运行时间）
    img = torch.zeros((1, 3, imgsz, imgsz), device=opt.device)  # init img
    save_path = str(Path(out))
    
    vid_path, vid_writer = None, None
    for frame_idx, (path, img, im0s, vid_cap, _) in enumerate(dataset):
        print(f"for {frame_idx}th image") 
        img = torch.from_numpy(img).to(opt.device)
        img = img.float()  
        img /= 255.0 
         
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 进行推理
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS(非极大值抑制)
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_sync()
        for i, det in enumerate(pred):  # detections per image
            # 判断张量是否为空，即没有检测到目标的情况，为空直接跳过下面的语句，进入下一次循环
            if det is None:
                continue

            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
                
            if det is not None and len(det):
                # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                annotator = Annotator(im0s, line_width=3, example=str(names))
                # Rescale boxes from img_size to im0 size(用scale_coords函数来将图像缩放)

                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0
                                                  ]]  # normalization gain whwh
                # 写入结果，绘制目标框
                num = 0
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) # label format
                    # with open(f'{txt_path}.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c = int(cls)  # integer class
                    label = ""
                    # label = f'{names[c]}'
                    # label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    num += 1
                
                bbox_xywh = []
                confs = []
                classes = []
                img_h, img_w, _ = im0.shape

                # 把im0的检测结果调整至deepsort的输入数据类型
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                xywhs = torch.Tensor(
                    bbox_xywh)  # 调用Tensor类的构造函数__init__，生成单精度浮点类型的张量
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                # 把调整好的检测结果输入deepsort
                outputs = deepsort.update(xywhs, confss, im0, classes)
                location = box_centers_f(outputs)
                locations.append(location)
                # 每五帧写入一次测速的数据
                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                    
                    locations = []
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]
                    draw_speed(im0, speed, bbox_xyxy, identities)
                    # draw_boxes(im0, bbox_xyxy, [names[i] for i in classes2],
                    #            classes2, identities)
            # Print time (inference + NMS)
            print('%s Done. (%.3fs)' % (s, t2 - t1))

            # print('sample %d / %d :, %s Done. (%.3fs)' % (i, len(pred), s, t2 - t1))
            # Stream results
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release(
                    )  # release previous video writer
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*opt.fourcc),
                    fps, (w, h))
            print(save_path)
            vid_writer.write(im0) 

            # Save results (image with detections)
             
                
    print('Results saved to %s' % out)
    # if platform == 'darwin':  # MacOS
    #     os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0)) 
