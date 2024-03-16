import json



#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   oov_result_submit.py
@Time    :   2023/02/21 22:24:59
@Author  :   HeiTong 
@Version :   1.0
@Contact :   1193700079@qq.com
@WebSite :   https://www.heitong.site/
todo: 
1.变成单模型推理 保存json 带置信度文件
 
'''

import argparse

import torch

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import gradio as gr

def recognize_text(image):
    raw_image = image.convert('RGB')
    image_mae = img_transform_cmf(raw_image).unsqueeze(0).to(args.device)
    image_clip = img_transform_clip(raw_image).unsqueeze(0).to(args.device)

    p = model_mae3(image_mae,image_clip).softmax(-1)
    new_pred, p = model_mae3.tokenizer.decode(p)
    new_pred = new_pred[0]
    p_score = torch.prod(p[0])
    return new_pred


import natsort,glob            
if __name__ == '__main__':
    
    examples = natsort.natsorted(glob.glob('./example/*.png'))
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    args.device = "cuda"
    args.checkpoint = "checkpoints/CMFSTR_ocr/sota_union14m.ckpt"
    model_mae3 = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device) # 新模型

    img_transform_cmf = SceneTextDataModule.get_transform(model_mae3.hparams.img_size)
    img_transform_clip = SceneTextDataModule.get_transform([224,224])

    iface = gr.Interface(fn=recognize_text,
                     inputs=gr.Image(type='pil', label="upload image"),
                     outputs=gr.Textbox(label="recognized text"),
                     examples=examples,
                     title="CMFSTR-Scene Text Recognition Example",
                     description="Upload images and view recognized text! Note that this example only supports recognizing English characters.")

    # 启动应用程序
    iface.launch()


