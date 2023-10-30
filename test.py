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
import torch
from utils.tools import *
from modules import FMCLIP


def validate(epoch, val_loader, item_des, score_des, device, model, fm_model, config):
    model.eval()
    fm_model.eval()
    num = 0
    item_corr_1, score_corr_1 = 0, 0
    item_corr_3, score_corr_3 = 0, 0

    with torch.no_grad():
        score_des_emb = model.encode_text(score_des)
        item_des_emb = model.encode_text(item_des)
        for iii, (frames, texts, item_label, score_label) in enumerate(tqdm(val_loader)):

            frames = frames.view((-1, config.data.num_segments, 3) + frames.size()[-2:])
            b,t,c,h,w = frames.size()
            frames= frames.to(device).view(-1,c,h,w)
            frame_emb = model.encode_image(frames)
            frame_emb = frame_emb.view(b,t,-1)
            texts = texts.view(-1, 77).to(device)
            text_emb = model.encode_text(texts)  #.view(b, 5, -1)

            logit_scale = 1.0
            score_truth = torch.tensor(np.zeros(shape=(len(score_label), score_des_emb.shape[0])), dtype=frame_emb.dtype, device=device) # (bs, bs)
            item_truth = torch.tensor(np.zeros(shape=(len(item_label), item_des_emb.shape[0])), dtype=frame_emb.dtype, device=device) # (bs, bs)

            v2i_sim, v2s_sim, loss = fm_model(frame_emb, text_emb, item_des_emb, score_des_emb, logit_scale, item_truth, score_truth)
            # print(v2i_sim, v2s_sim)
            v2i_sim = v2i_sim.view(b, -1).softmax(dim=-1)
            v2s_sim = v2s_sim.view(b, -1).softmax(dim=-1)

            item_1, idx_item_1 = v2i_sim.topk(1, dim=-1)
            item_3, idx_item_3 = v2i_sim.topk(3, dim=-1)

            score_1, idx_score_1 = v2s_sim.topk(1, dim=-1)
            score_3, idx_score_3 = v2s_sim.topk(3, dim=-1)

            num += b
            for i in range(b):
                if idx_item_1[i] == item_label[i]:
                    item_corr_1 += 1
                if item_label[i] in idx_item_3[i]:
                    item_corr_3 += 1
                if idx_score_1[i] == score_label[i]:
                    score_corr_1 += 1
                if score_label[i] in idx_score_3[i]:
                    score_corr_3 += 1
    item_top1 = float(item_corr_1) / num * 100
    item_top3 = float(item_corr_3) / num * 100
    score_top1 = float(score_corr_1) / num * 100
    score_top3 = float(score_corr_3) / num * 100
    print('Epoch: [{}/{}]:'.format(epoch, config.solver.epochs))
    print('Test Item: Top1: {:.4f}%, Top3: {:.4f}%'.format(item_top1, item_top3))
    print('Test Score: Top1: {:.4f}%, Top3: {:.4f}%'.format(score_top1, score_top3))
    return item_top1, item_top3, score_top1, score_top3

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

    fm_model = FMCLIP(config.network.sim_header, clip_state_dict,config.data.num_segments).to(device)

    val_data = VideoDataset(config.data.val_list, config.data.label_list, config.data.num_segments, config.data.input_size)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=False)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fm_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    score_des = val_data.score_des.to(device)
    item_des = val_data.item_des.to(device)
    item_top1, item_top3, score_top1, score_top3 = validate(start_epoch, val_loader, item_des, score_des, device, model, fm_model, config)


if __name__ == '__main__':
    main()
