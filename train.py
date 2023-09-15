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
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from test import validate
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.saving import  *
from dotmap import DotMap
import warnings
import random
import numpy as np
import clip
warnings.filterwarnings('ignore')


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

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
    model, clip_state_dict = clip.load(config.network.arch, device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).to(device)
    model_image = torch.nn.DataParallel(model_image).to(device)
    fusion_model = torch.nn.DataParallel(fusion_model).to(device)
    # wandb.watch(model)
    # wandb.watch(fusion_model)
    train_data = VideoDataset(config.data.train_list, config.data.label_list, config.data.num_segments, config.data.input_size, isTraining=True)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True, pin_memory=False, drop_last=True)
    val_data = VideoDataset(config.data.val_list, config.data.label_list, config.data.num_segments, config.data.input_size)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=False, drop_last=False)
    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch

    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1, best_prec5 = 0.0, 0.0

    step_count = 0
    loss_max = 100

    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        total_loss = 0.0
        for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
            b,t,c,h,w = images.size()
            # text_id = np.random.randint(train_data.num_text_aug, size=len(list_id))
            # texts = torch.stack([train_data.text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            texts = train_data.classes[list_id]

            images= images.to(device).view(-1,c,h,w)
            texts = texts.to(device)
            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            loss = (loss_imgs + loss_texts)/2
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
            prec1, prec3, prec5 = validate(epoch,val_loader, val_data.classes, device, model,fusion_model, config)

        if total_loss < loss_max:
            step_count = 0
            loss_max = total_loss
            filename = "{}/best_loss.pt".format(working_dir)
            epoch_saving(epoch, model, fusion_model, optimizer, filename)
        else:
            step_count += 1

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {:.3f}%/{:.3f}%, loss: {:.3f}'.format(prec1, best_prec1, total_loss))
        print('Saving:')

        if is_best:
            best_prec5 = prec5
            best_saving(working_dir, epoch, model, fusion_model, optimizer)
    print('Best Testing: {}/{}'.format(best_prec1, best_prec5))

if __name__ == '__main__':
    main()
