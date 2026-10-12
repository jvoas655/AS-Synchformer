import csv
import json
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

import torch


sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_video_and_audio, subsample_dataset)


class SportsAndNews(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='VideoReader',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None,
                 to_filter_bad_examples=True,
                 data_cfg=None
                 ):
        super().__init__()
        self.fast_skip = False
        self.data_cfg = data_cfg
        self.max_clip_len_sec = data_cfg.crop_len_sec
        logging.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        split_clip_ids_path = os.path.join(splits_path, f'sports_and_news_uniform.{split}.json')

        # read the ids from a split
        self.clip_paths = []
        self.offset_secs = []
        self.v_start_i_secs = []
        with open(split_clip_ids_path) as split_clip_ids:
            split_clip_ids_data = json.load(split_clip_ids)
            for video_id, specs in split_clip_ids_data.items():
                video_folder = "_".join(video_id.split("_")[:-4])
                file_stem = "_".join(video_id.split("_")[:-2])
                self.clip_paths.append(data_cfg.vids_path + video_folder + '/' + file_stem + '.mkv')
                self.offset_secs.append(float(specs["offset_sec"]))
                self.v_start_i_secs.append(float(specs["v_start_i_sec"]))

        self.dataset = list(range(len(self.clip_paths)))
        self.dataset = subsample_dataset(self.dataset, getattr(size_ratio, self.split), shuffle=split == 'train')

        logging.info(f'{split} has {len(self.dataset)} items')

    def __getitem__(self, index):
        if (self.fast_skip):
            return 0
        dataset_index = self.dataset[index]
        clip_path, offset_sec, v_start_i_sec = self.clip_paths[dataset_index],  self.offset_secs[dataset_index], self.v_start_i_secs[dataset_index]

        if (hasattr(self.data_cfg, "audio_only")):
            _, audio, meta = get_video_and_audio(clip_path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=v_start_i_sec, loading_shift = self.data_cfg.loading_shift, loading_buffer=self.data_cfg.loading_buffer, audio_only=True)
            if audio is None or meta is None:
                return self[index - 1]
            item = {'audio': audio, 'meta': meta, 'path': clip_path, 'targets': {}, 'split': self.split}
        else:
            rgb, audio, meta = get_video_and_audio(clip_path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=v_start_i_sec, loading_shift = self.data_cfg.loading_shift, loading_buffer=self.data_cfg.loading_buffer)
            if rgb is None or audio is None or meta is None:
                return self[index - 1]
            item = {'video': rgb, 'audio': audio, 'meta': meta, 'path': clip_path, 'targets': {}, 'split': self.split, "index": dataset_index}

        # loading fixed offsets so we could evaluate on the same data each time (valid and test)
        if self.split in self.load_fixed_offsets_on:
            item['targets']['offset_sec'] = offset_sec
            item['targets']['v_start_i_sec'] = 2
         

        if self.transforms is not None:
            item = self.transforms(item)

        return item


    def __len__(self):
        return len(self.dataset)
    def enable_fast_skip(self):
        print("Fast Skip Enabled")
        self.fast_skip = True
    def disable_fast_skip(self):
        if (self.fast_skip):
            print("Fast Skip Disabled")
            self.fast_skip = False


