import os
import clip
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy as np
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model, config):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_3 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (torch.mm(100.0 * image_features,text_features.T))
            similarity = similarity.view(b, -1).softmax(dim=-1)
            # similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_3, indices_3 = similarity.topk(3, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_3[i]:
                    corr_3 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top3 = float(corr_3) / num * 100
    top5 = float(corr_5) / num * 100
    print('Epoch: [{}/{}]: Top1: {:.4f}%, Top3: {:.4f}%, Top5: {:.4f}%'.format(epoch, config.solver.epochs, top1, top3, top5))
    return top1, top3, top5

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)



    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32


    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    val_data = VideoDataset(config.data.val_list, config.data.label_list, config.data.num_segments, config.data.input_size)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=False)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    best_prec1 = 0.0
    prec1, prec3, prec5 = validate(start_epoch,val_loader, val_data.classes, device, model,fusion_model, config)

if __name__ == '__main__':
    main()
