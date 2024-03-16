import argparse
import json
import os
import os.path as osp
import sys

import torch
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import AverageMeter, Corrector, ResultFormat

import math
import random
import string

import cv2
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
    


def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img



from PIL import Image, ImageDraw, ImageFont,ImageOps
fontText = ImageFont.truetype("simsun.ttc", 30, encoding="utf-8")
def test(img_path, model, cfg):
    model.eval()

    raw_img = Image.open(img_path)
    (w,h) = raw_img.size  

    img = np.array(raw_img)

    img_meta = dict(org_img_size=np.array(img.shape[:2]))

    img = scale_aligned_short(img, cfg.data.test.short_size)
    img_meta.update(dict(img_size=np.array(img.shape[:2])))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(img)
    data = dict(imgs=img, img_metas=img_meta)

    data['imgs'] = data['imgs'].cuda().unsqueeze(0)
    data.update(dict(cfg=cfg))

    with torch.no_grad():
        outputs = model(**data)
    bboxes = outputs['bboxes']
    lines = []
    
    for i, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = '%d,%d,%d,%d,%d,%d,%d,%d' % tuple(
            values) + '\n' 
        x1,y1,x2,y2,x3,y3,x4,y4 = values
        draw = ImageDraw.Draw(raw_img)
        draw.point(((x1,y1),(x2,y2),(x3,y3),(x4,y4)),(0, 255, 0))
        draw.line((x1, y1, x2, y2), 'yellow',width=5)
        draw.line((x2, y2, x3, y3), 'yellow',width=5)
        draw.line((x3, y3, x4, y4), 'yellow',width=5)
        draw.line((x4, y4, x1, y1), 'yellow',width=5)
        coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        x_min, y_min, x_max, y_max = w+1, h+1, -1, -1
        for i in coordinates:
            x_min = 0 if i[0] < 0 else (i[0] if i[0] < x_min else x_min)
            y_min = 0 if i[1] < 0 else (i[1] if i[1] < y_min else y_min)
            x_max = w if i[0] > w else (i[0] if i[0] > x_max else x_max)
            y_max = h if i[1] > h else (i[1] if i[1] > y_max else y_max)
        draw.text((x_max, y_max), ".", (0, 255, 0), font=fontText)
        lines.append(line)
    raw_img.save("test.png")
    return data
   

def main(args):
    args.config = "./config/psenet/psenet_r50_ic15_1024.py"
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    data_loader = build_data_loader(cfg.data.test)
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=data_loader.voc,
                char2id=data_loader.char2id,
                id2char=data_loader.id2char,
            ))
    model = build_model(cfg.model)
    model = model.cuda()
    args.checkpoint = "checkpoints/psenet_r50_ic15_1024/checkpoint_133ep.pth.tar"
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(
                args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model_structure(model)
    # test
    img_path = "/media/data1/yrq/pan_pp.pytorch/data/ship_data_panpp_version/test_images/O_20190504_00_22_08_044002.jpg"
    test(img_path,model, cfg)


def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('config', help='config file path')
    # parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    parser.add_argument('--vis', nargs='?', type=str2bool, default=False)
    args = parser.parse_args()

    main(args)

