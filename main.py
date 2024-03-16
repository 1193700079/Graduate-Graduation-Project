import json
import random
import argparse
import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import gradio as gr
from mmengine.config import Config
from PIL import Image, ImageDraw, ImageFont,ImageOps
from mmocr.apis import MMOCRInferencer

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

from tools import scale_aligned_short,calculate_angle_with_x_axis,model_structure,find_max_enclosing_rectangle

fontText = ImageFont.truetype("simsun.ttc", 30, encoding="utf-8")


def recognize_text(image,img_trf_mae,img_trf_clip,model):
    raw_image = image.convert('RGB')
    image_mae = img_trf_mae(raw_image).unsqueeze(0).to(args.device)
    image_clip = img_trf_clip(raw_image).unsqueeze(0).to(args.device)

    p = model(image_mae,image_clip).softmax(-1)
    new_pred, p = model.tokenizer.decode(p)
    new_pred = new_pred[0]
    p_score = float(torch.prod(p[0]))
    return (new_pred,p_score)


def ship_detect(image):
    raw_img = image.copy()
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
        outputs = det_model(**data)
    bboxes = outputs['bboxes']
    lines = []
    crop_imgs=[]
    rotate_imgs=[]
    for i, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = '%d,%d,%d,%d,%d,%d,%d,%d' % tuple(
            values) + '\n' 
        x1,y1,x2,y2,x3,y3,x4,y4 = values
        draw = ImageDraw.Draw(image)
        draw.point(((x1,y1),(x2,y2),(x3,y3),(x4,y4)),(0, 255, 0))
        draw.line((x1, y1, x2, y2), 'yellow',width=5)
        draw.line((x2, y2, x3, y3), 'yellow',width=5)
        draw.line((x3, y3, x4, y4), 'yellow',width=5)
        draw.line((x4, y4, x1, y1), 'yellow',width=5)
        coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        angle = calculate_angle_with_x_axis(coordinates)
        x_min, y_min, x_max, y_max = w+1, h+1, -1, -1
        for i in coordinates:
            x_min = 0 if i[0] < 0 else (i[0] if i[0] < x_min else x_min)
            y_min = 0 if i[1] < 0 else (i[1] if i[1] < y_min else y_min)
            x_max = w if i[0] > w else (i[0] if i[0] > x_max else x_max)
            y_max = h if i[1] > h else (i[1] if i[1] > y_max else y_max)
        draw.text((x_max, y_max), ".", (0, 255, 0), font=fontText)
        lines.append(line)
        crop_img = image.crop((x_min, y_min, x_max, y_max))
        crop_imgs.append(crop_img)
        tmp_img = raw_img.crop((x_min, y_min, x_max, y_max)).rotate(angle)
        rotate_imgs.append(tmp_img)
    reg_pred_list = []
    for i in rotate_imgs:
        reg_pred_list.append(recognize_text(i,img_transform_cmf,img_transform_clip,model_ship_rec))

    return lines,image,crop_imgs,rotate_imgs,reg_pred_list


def ocr_detect(image):
    image.save('tmp.jpg')
    raw_img = image.copy()
    rgb_image = np.array(image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    draw = ImageDraw.Draw(image)
    ocr_results = ocr(bgr_image)

    lines = []
    crop_imgs=[]
    rotate_imgs=[]
    reg_pred_list = []
    for prediction in ocr_results['predictions']:
        for i in range(len(prediction['det_polygons'])):
            polygon = prediction['det_polygons'][i]
            try:
                lines.append(polygon)
                polygon = [int(i) for i in polygon]
                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                draw.polygon(polygon, outline='green', fill=None)
                bottom_left, top_right, _  = find_max_enclosing_rectangle(polygon)
                x_min, y_min = bottom_left
                x_max, y_max = top_right
                crop_img = raw_img.crop((x_min, y_min, x_max, y_max))
                crop_imgs.append(crop_img)
                rotate_imgs.append(crop_img)
            except:
                pass    

    reg_pred_list = []
    print(len(rotate_imgs))
    try:
        if len(rotate_imgs) != 0:
            for i in rotate_imgs:
                pred1,score1 = recognize_text(i,img_transform_cmf_rec,img_transform_clip_rec,model_rec)
                pred2,score2 = recognize_text(i,img_transform_cmf_rec,img_transform_clip_rec,model_rec_u14m)
                if score1 >= score2:
                    reg_pred_list.append((pred1,score1))
                else:
                    reg_pred_list.append((pred2,score2))  
            return lines,image,crop_imgs,rotate_imgs,reg_pred_list

        else:
            pred1,score1 = recognize_text(raw_img,img_transform_cmf_rec,img_transform_clip_rec,model_rec)
            pred2,score2 = recognize_text(raw_img,img_transform_cmf_rec,img_transform_clip_rec,model_rec_u14m)
            if score1 >= score2:
                reg_pred_list.append((pred1,score1))
            else:
                reg_pred_list.append((pred2,score2))    
            print(reg_pred_list)    

            return lines,image,crop_imgs,rotate_imgs,reg_pred_list
    except:
        pred1,score1 = recognize_text(raw_img,img_transform_cmf_rec,img_transform_clip_rec,model_rec)
        pred2,score2 = recognize_text(raw_img,img_transform_cmf_rec,img_transform_clip_rec,model_rec_u14m)
        if score1 >= score2:
            reg_pred_list.append((pred1,score1))
        else:
            reg_pred_list.append((pred2,score2))    
        print(reg_pred_list)    

        return lines,image,crop_imgs,rotate_imgs,reg_pred_list

            



mirror_mode = False

def ocr_main(dropdown,image):

    if mirror_mode:
        image = ImageOps.mirror(image)

    if dropdown == "船牌检测和识别":
        return ship_detect(image)
    elif dropdown == "通用OCR检测和识别":
        return ocr_detect(image)
    else:
        return ship_detect(image)
        


def clear_input():
    input_image.clear()
    det_out.clear()
    det_out_full_image.clear()
    det_out_image.clear()
    det_rotate.clear()

import natsort,glob     
import random
random.seed(2024)       
if __name__ == '__main__':

    ocr = MMOCRInferencer(det='DBNetPP')
    examples = natsort.natsorted(glob.glob('./test_images/*.jpg'))
    examples = examples[:20]
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

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
    det_model = build_model(cfg.model)
    det_model = det_model.cuda()
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
            det_model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    det_model = fuse_module(det_model)
    model_structure(det_model)
    det_model.eval()


    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    args.device = "cuda"
    args.checkpoint = "checkpoints/CMFSTR_ship/best.ckpt"
    model_ship_rec = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device) # 新模型

    img_transform_cmf = SceneTextDataModule.get_transform(model_ship_rec.hparams.img_size)
    img_transform_clip = SceneTextDataModule.get_transform([224,224])


    args.checkpoint = "checkpoints/CMFSTR_ocr/sota_real.ckpt"  # SOTA_real
    model_rec = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device) # 新模型

    img_transform_cmf_rec = SceneTextDataModule.get_transform(model_rec.hparams.img_size)
    img_transform_clip_rec = SceneTextDataModule.get_transform([224,224])


    args.checkpoint = "checkpoints/CMFSTR_ocr/sota_union14m.ckpt"   # SOTA_union14m
    model_rec_u14m = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device) # 新模型

    
    input_Imgabox = gr.Image()
    algorithm_choices = ["船牌检测和识别", "通用OCR检测和识别"]
    with gr.Blocks(title="智能船牌识别") as demo:
        # gr.Markdown('<h1 style="text-align: center;">浙江工商大学-智能船牌检测识别系统</h1>')
        gr.Markdown('<div style="text-align: center; font-size: 20px; font-weight: bold;">浙江工商大学-智能船牌检测识别系统</div>')
        gr.Markdown('<div style="text-align: center;">请先上传图片进行检测然后进行识别. 点击submit按钮，赶快试试吧！</div>')
        with gr.Column():    
            dropdown = gr.Dropdown(choices=algorithm_choices, label="Select an Algorithm")

            input_image = gr.Image(type='pil', label="上传图片")
        with gr.Row():      
            clear = gr.ClearButton()
            submit = gr.Button("submit")
        with gr.Column():   
            det_out = gr.Textbox(label="检测框坐标")
            det_out_full_image = gr.Image(label="检测结果")
        with gr.Row():    
            det_out_image = gr.Gallery(label="检测结果")
            det_rotate = gr.Gallery( label="矫正结果")
        with gr.Column():    
            reg_out = gr.Textbox(label="识别结果")    
        submit.click(fn=ocr_main, inputs=[dropdown,input_image], outputs=[det_out,det_out_full_image,det_out_image,det_rotate,reg_out])
        clear.add(components=[input_image,det_out,det_out_full_image, det_out_image,det_rotate,reg_out])
        with gr.Row():  
            gr.Examples(examples,inputs=input_image)
    demo.launch(share=False, favicon_path="system/static/images/login-logo.png")
    
  
    


