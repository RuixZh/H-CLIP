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
        return ' '.join(self._data[1].split('_'))


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
        self.text_prompt()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split('\t')) for x in open(self.list_file)]

    @property
    def categories(self):
        return [[i, l] for i, l in self.label_mapping.items()]

    def text_prompt(self):
        text_aug = f"The video shows {{}}"
        text_dict = {}
        with open(self.category_file, 'r') as f:
            for line in f.readlines():
                l, d = line.strip().split('\t')
                if l in [str(i) for i in range(1, 6)]:
                text_dict[self.label_mapping_reverse[l]] = clip.tokenize(text_aug.format(d))

        classes = torch.cat([text_dict[v] for v in range(len(text_dict))])
        self.classes = classes

        # text_aug = [f"The video shows {{}}",
                    # f"This is an assessment of {{}}",
                    # f"{{}} is an evaluation item of Fugl-Meyer Assessment",
                    # f"The person is doing {{}}, to assess how the motor recovery is",
                    # f"Can you recognize the assessment of {{}}?", f"Video classification of {{}}",
                    # f"Human action of {{}}", f"{{}}, a video of Fugl-Meyer Assessment",
                    # f"{{}}"]
        # self.text_dict = {}
        # classes = {}
        # self.num_text_aug = len(text_aug)
        # with open(self.category_file, 'r') as f:
        #     for line in f.readlines():
        #         l, d = line.strip().split('\t')
        #         classes[self.label_mapping_reverse[l]] = d
        #
        # for ii, txt in enumerate(text_aug):
        #     self.text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in classes.items()])
        #
        # self.classes  = torch.cat([v for k, v in self.text_dict.items()])

    def _gen_label_dict(self):
        self.label_mapping = {}
        self.label_mapping_reverse = {}
        i = 0
        for action in [str(i) for i in range(1, 6)]:
            self.label_mapping[i] = action
            self.label_mapping_reverse[action] = i
            i += 1
            for score in [str(j) for j in range(3)]:
                self.label_mapping[i] = action+score
                self.label_mapping_reverse[action+score] = i
                i += 1
        self.nb_label = len(self.label_mapping)

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
        # f, c, v = record.video_path.split('/')
        # path2imgs = os.path.join('dataset', 'cleaned_frames', c, v)
        path2imgs = os.path.join('dataset',record.video_path)
        label = record.label

        filenames = [f for f in os.listdir(path2imgs)]
        filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))

        nb_frame = len(filenames)
        try:
            segment_indices = self._sample_indices(nb_frame) if self.isTraining else self._get_val_indices(nb_frame)
        except ValueError:
            print(record.video_path)
        filenames = [filenames[i] for i in segment_indices]

        images = []
        for i, filename in enumerate(filenames):
            try:
                frame_file = os.path.join(path2imgs, filename)
                image = self._load_image(frame_file)
                images.extend(image)

            except OSError:
                print('ERROR: Could not read image "%s"' % frame_file)
                raise
        video_emb = self.transform(images)
        label_id = self.label_mapping_reverse[label]

        return video_emb, label_id

    def __getitem__(self, idx):
        record = self.video_list[idx]
        return self._load_knowledge(record)

    def __len__(self):
        return len(self.video_list)
