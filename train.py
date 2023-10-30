import os
import torch.nn as nn
from dataloader import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
import pprint
from modules import FMCLIP
from test import validate
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.saving import  *
from dotmap import DotMap
import warnings
import random
import numpy as np
import clip
from fvcore.nn import FlopCountAnalysis, parameter_count_table
warnings.filterwarnings('ignore')

# class TextCLIP(nn.Module):
#     def __init__(self, model) :
#         super(TextCLIP, self).__init__()
#         self.model = model
#
#     def forward(self,text):
#         return self.model.encode_text(text)
#
# class ImageCLIP(nn.Module):
#     def __init__(self, model) :
#         super(ImageCLIP, self).__init__()
#         self.model = model
#
#     def forward(self,image):
#         return self.model.encode_image(image)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp',config['network']['arch'], str(config['data']['num_segments']))
    print('-' * 80)
    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32

    fm_model = FMCLIP(config.network.sim_header, clip_state_dict,config.data.num_segments).to(device)
    # fm_model = torch.nn.DataParallel(fm_model).to(device)
    # wandb.watch(fm_model)
    train_data = VideoDataset(config.data.train_list, config.data.label_list, config.data.num_segments, config.data.input_size, isTraining=True)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True, pin_memory=False, drop_last=True)
    val_data = VideoDataset(config.data.val_list, config.data.label_list, config.data.num_segments, config.data.input_size)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=False, drop_last=False)

    start_epoch = config.solver.start_epoch

    optimizer = _optimizer(config, model, fm_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    # i1 = torch.randint(100, (1, 77)).to(device)
    # i11 = torch.randint(100, (4, 77)).to(device)
    # i12 = torch.randint(100, (12, 77)).to(device)
    #
    # i2 = torch.randn(8, 3, 224, 224).to(device)
    # i31 = torch.randn(1, 8, clip_state_dict["text_projection"].shape[1]).to(device)
    # i32 = torch.randn(1, clip_state_dict["text_projection"].shape[1]).to(device)
    #
    # i33 = torch.randn(4, clip_state_dict["text_projection"].shape[1]).to(device)
    # i34 = torch.randn(12, clip_state_dict["text_projection"].shape[1]).to(device)
    #
    # i35 = torch.tensor(0., device=device)
    # i36 = torch.randn(1, 4).to(device)
    # i37 = torch.randn(1, 12).to(device)
    #
    # text_enc = TextCLIP(model).to(device)
    # img_enc = ImageCLIP(model).to(device)
    # flops = FlopCountAnalysis(text_enc, (i1)).total() + FlopCountAnalysis(text_enc, (i11)).total()+FlopCountAnalysis(text_enc, (i12)).total()
    # flops2 = FlopCountAnalysis(img_enc, (i2)).total()
    # flops3 = FlopCountAnalysis(fm_model, (i31, i32, i33, i34, i35, i36, i37)).total()
    #
    # print("FLOPs: ", (flops+flops2+flops3)/1e9)
    # print(parameter_count_table(model))
    # print('-' * 80)
    # n_parameters = sum(p.numel() for p in fm_model.parameters() if p.requires_grad) + sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('# params:', n_parameters)
    # print('-' * 80)

    bi1, bi3, bs1, bs3 = 0.0, 0.0, 0.0, 0.0

    score_des = train_data.score_des.to(device)
    item_des = train_data.item_des.to(device)

    for epoch in range(start_epoch, config.solver.epochs):
        model.train()
        fm_model.train()
        total_loss = 0.0
        for kkk, (frames, texts, item_label, score_label) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            frames = frames.view((-1, config.data.num_segments, 3) + frames.size()[-2:])
            b,t,c,h,w = frames.size()
            frames= frames.to(device).view(-1,c,h,w)
            frame_emb = model.encode_image(frames)
            frame_emb = frame_emb.view(b,t,-1)
            texts = texts.view(-1, 77).to(device)
            text_emb = model.encode_text(texts)  #.view(b, -1)

            score_des_emb = model.encode_text(score_des)
            item_des_emb = model.encode_text(item_des)

            logit_scale = model.logit_scale.exp()

            score_truth = torch.tensor(gen_label(score_label, score_des_emb.shape[0]), dtype=frame_emb.dtype, device=device) # (bs, bs)
            item_truth = torch.tensor(gen_label(item_label, item_des_emb.shape[0]), dtype=frame_emb.dtype, device=device) # (bs, bs)
            # score_truth = score_label.to(device)
            # item_truth = item_label.to(device)
            v2i_sim, v2s_sim, loss = fm_model(frame_emb, text_emb, item_des_emb, score_des_emb, logit_scale, item_truth, score_truth)

            total_loss += loss
            loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        total_loss /= (kkk+1)
        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            item_top1, item_top3, score_top1, score_top3 = validate(epoch, val_loader, item_des, score_des, device, model, fm_model, config)

        is_best = score_top1 > bs1
        bs1 = max(score_top1, bs1)

        if is_best:
            print('Saving Best:')
            bi1, bi3, bs1, bs3 = item_top1, item_top3, score_top1, score_top3
            best_saving(working_dir, epoch, model, fm_model, optimizer)
            print('Best Testing: item: {:.2f}%, score: {:.2f}%'.format(bi1, bs1))

    print('Best Test Item: Top1: {:.2f}%, Top3: {:.2f}%'.format(bi1, bi3))
    print('Best Test Score: Top1: {:.2f}%, Top3: {:.2f}%'.format(bs1, bs3))

if __name__ == '__main__':
    main()
