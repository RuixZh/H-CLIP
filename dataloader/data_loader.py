import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pdb
from PIL import Image, ImageOps
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collections import defaultdict
from .transforms_ss import *
from RandAugment import RandAugment
import clip

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def video_path(self):
        return self._data[0]

    @property
    def label(self):
        return self._data[1]


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class VideoDataset(Dataset):
    def __init__(self, list_file, category_file, max_frames, n_px=224, isTraining=False):
        self.list_file = list_file
        self.category_file = category_file
        self.max_frames = max_frames
        self.n_px = n_px
        self.isTraining = isTraining

        self.transform = self._transform()
        if self.isTraining:
            self.transform.transforms.insert(0, GroupTransform(RandAugment(2, 9)))

        self._parse_list()
        self._gen_label_dict()
        # self.text_prompt()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split('\t')) for x in open(self.list_file)]

    def _gen_label_dict(self):
        item = {}  # (12,)
        score = {}   # (4,)
        prefix = ''
        with open(self.category_file) as f:
            b = json.load(f)
        for i in range(1, 5):
            item[str(i)] = b[str(i)]
            prefix = b[str(i)]
            for j in range(0, 3):
                score[str(i)+str(j)] = prefix + b[str(i)+str(j)]
        self.score_des = clip.tokenize(list(score.values()))
        self.item_des = clip.tokenize(list(item.values()))
        self.score_map = {k: i for i,k in enumerate(score.keys())}
        self.item_map = {k: i for i,k in enumerate(item.keys())}

    def _transform(self):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = self.n_px * 256 // 224
        if self.isTraining:

            unique = torchvision.transforms.Compose([GroupMultiScaleCrop(self.n_px, [1, .875, .75, .66]),
                                                     GroupRandomHorizontalFlip(is_sth=False),
                                                     GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                            saturation=0.2, hue=0.1),
                                                     GroupRandomGrayscale(p=0.2),
                                                     GroupGaussianBlur(p=0.0),
                                                     GroupSolarization(p=0.0)]
                                                    )
        else:
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(self.n_px)])

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean,
                                                                input_std)])
        return torchvision.transforms.Compose([unique, common])

    def _sample_indices(self, num_frames):
        if num_frames <= self.max_frames:
            offsets = np.concatenate((
                np.arange(num_frames),
                np.random.randint(num_frames,
                        size=self.max_frames - num_frames)))
            return np.sort(offsets)
        offsets = list()
        ticks = [i * num_frames // self.max_frames for i in range(self.max_frames + 1)]

        for i in range(self.max_frames):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= 1:
                tick += np.random.randint(tick_len)
            offsets.extend([j for j in range(tick, tick + 1)])
        return np.array(offsets)

    def _get_val_indices(self, num_frames):
        if self.max_frames == 1:
            return np.array([num_frames //2], dtype=np.int_)

        if num_frames <= self.max_frames:
            return np.array([i * num_frames // self.max_frames
                             for i in range(self.max_frames)], dtype=np.int_)
        offset = (num_frames / self.max_frames - 1) / 2.0
        return np.array([i * num_frames / self.max_frames + offset
                         for i in range(self.max_frames)], dtype=np.int_)

    def _load_image(self, filepath):
        return [Image.open(filepath).convert('RGB')]

    def _load_knowledge(self, record):
        path2imgs = os.path.join('dataset', 'frames', record.video_path)
        anns_file = os.path.join('dataset', 'annotations', record.video_path + '.txt')
        item_label, score_label = self.item_map[record.label[0]], self.score_map[record.label]

        img_filenames = [f for f in os.listdir(path2imgs)]
        img_filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))

        nb_frame = len(img_filenames)
        try:
            segment_indices = self._sample_indices(nb_frame) if self.isTraining else self._get_val_indices(nb_frame)
        except ValueError:
            print(record.video_path)
        img_filenames = [img_filenames[i] for i in segment_indices]

        images = []
        for i, filename in enumerate(img_filenames):
            try:
                frame_file = os.path.join(path2imgs, filename)
                image = self._load_image(frame_file)
                images.extend(image)

            except OSError:
                print('ERROR: Could not read image "%s"' % frame_file)
                raise
        frame_emb = self.transform(images)

        # text_emb = torch.zeros((5, 77), dtype=np.compat.long)
        lines = ''
        with open(anns_file, 'r') as f:
            for l in f.readlines():
                lines += l
        try:
            text_emb = clip.tokenize(lines)[0]
        except Exception as e:
            print(anns_file)
            raise


        return frame_emb, text_emb, item_label, score_label

    def __getitem__(self, idx):
        record = self.video_list[idx]
        return self._load_knowledge(record)

    def __len__(self):
        return len(self.video_list)
