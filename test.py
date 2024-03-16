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
import shutil,os      
true_output_path = 'error/our_true'
os.makedirs(true_output_path,exist_ok=True)
error_output_path = 'error/our_error'
os.makedirs(error_output_path,exist_ok=True)


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

  
    abinet_error_img_list = []
    ABINet_pred_dict = {}
    path = "/media/data1/yrq/ocr/parseq/det/error/ABInet_error"
    files = natsort.natsorted(glob.glob(path + "/*.jpg"))
    for i in files:
        name = i.replace(".jpg","")
        jpg_name = i.split('/')[-1]
        gt = name.split('$')[-1]
        ABINet_pred = name.split('$')[-2]
        abinet_error_img = jpg_name.split('$')[0]
        abinet_error_img_list.append(abinet_error_img)
        ABINet_pred_dict[abinet_error_img] = ABINet_pred
    
    parseq_error_img_list = []
    Parseq_pred_dict = {}
    path = "/media/data1/yrq/ocr/parseq/det/error/Parseq_error"
    files = natsort.natsorted(glob.glob(path + "/*.jpg"))
    for i in files:
        name = i.replace(".jpg","")
        jpg_name = i.split('/')[-1]
        gt = name.split('$')[-1]
        Parseq_pred = name.split('$')[-2]    
        parseq_error_img = jpg_name.split('$')[0]
        parseq_error_img_list.append(parseq_error_img)
        Parseq_pred_dict[parseq_error_img] = Parseq_pred


    path = "/media/data1/yrq/ocr/parseq/det/error/maerec_error"
    files = natsort.natsorted(glob.glob(path + "/*.jpg"))
    for i in files:
        name = i.replace(".jpg","")
        jpg_name = i.split('/')[-1]
        gt = name.split('$')[-1]
        maerec_pred = name.split('$')[-2]
        maerec_error_img = jpg_name.split('$')[0]

        our_pred = recognize_text(Image.open(i))
        if our_pred == gt:
            if maerec_error_img in Parseq_pred_dict and maerec_error_img in ABINet_pred_dict: 
                shutil.copy(i, f"{true_output_path}/{maerec_error_img}${our_pred}${gt}.jpg")   
                # print(f"img:{maerec_error_img} gt:{gt} our_pred:{our_pred} maerec_pred:{maerec_pred}  Parseq_pred:{Parseq_pred_dict[maerec_error_img]} ABINet_pred:{ABINet_pred_dict[maerec_error_img]}")
        else:
            if maerec_error_img in Parseq_pred_dict and maerec_error_img in ABINet_pred_dict: 
                shutil.copy(i, f"{error_output_path}/{maerec_error_img}${our_pred}${gt}.jpg")   
                print(f"img:{maerec_error_img} gt:{gt} our_pred:{our_pred} maerec_pred:{maerec_pred}  Parseq_pred:{Parseq_pred_dict[maerec_error_img]} ABINet_pred:{ABINet_pred_dict[maerec_error_img]}")
                
