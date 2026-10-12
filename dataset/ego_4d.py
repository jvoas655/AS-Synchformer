import csv
import json
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

# Import a file locking library for thread-safe, multi-process file writes.
from filelock import FileLock

sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_video_and_audio, subsample_dataset)


class Ego4D(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='VideoReader',
                 size_ratio=None,
                 data_cfg=None,
                 sounds_only=False
                 ):
        super().__init__()
        self.sounds_only = sounds_only
        self.fast_skip = False
        self.data_cfg = data_cfg
        self.max_clip_len_sec = data_cfg.crop_len_sec
        logging.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = os.path.expanduser(splits_path)
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        split_clip_ids_path = os.path.join(self.splits_path, f'ego4d_uniform.{split}.json')
        source_data_path = os.path.join(self.splits_path, f'ego4d_uniform.{split}.csv')
        scenarios_path = os.path.join(self.splits_path, f'processed_scenarios.json')
        with open(scenarios_path) as fileref:
            self.scenario_data = json.load(fileref)  # 581 noun and 117 verb
        
        self.num_noun_ids = self.scenario_data["max_noun_id"]
        self.num_verb_ids = self.scenario_data["max_verb_id"]
        source_data_map = {}
        with open(source_data_path) as fileref:
            for line in fileref:
                split_line = line.split(",")
                vid_id, source = "_".join(split_line[0].split("_")[:-2]), split_line[-1]
                source_data_map[vid_id] = source

        # read the ids from a split
        self.clip_paths = []
        self.offset_secs = []
        self.v_start_i_secs = []
        self.uids = []
        with open(split_clip_ids_path) as split_clip_ids:
            split_clip_ids_data = json.load(split_clip_ids)
            for video_id, specs in split_clip_ids_data.items():
                file_stem = "_".join(video_id.split("_")[:-2])
                if self.sounds_only:
                    if "Ego4DSounds" not in source_data_map[file_stem]:
                        continue
                self.uids.append(file_stem)
                self.clip_paths.append(data_cfg.vids_path + file_stem + '.mp4')
                self.offset_secs.append(float(specs["offset_sec"]))
                self.v_start_i_secs.append(float(specs["v_start_i_sec"]))

        self.dataset = list(range(len(self.clip_paths)))
        # Subsample the dataset after initial generation to keep sampling repeatable.
        self.dataset = subsample_dataset(self.dataset, getattr(size_ratio, self.split), shuffle=split == 'train')
        logging.info(f'{split} has {len(self.dataset)} items')

        # Remove paths that previously failed.
        if hasattr(self.data_cfg, "skip_ids_path") and self.data_cfg.skip_ids_path:
            fail_paths_file = self.data_cfg.skip_ids_path
            failed_paths = set()
            if os.path.exists(fail_paths_file):
                try:
                    with open(fail_paths_file, 'r') as f:
                        data = json.load(f)
                        # Expecting a list of paths.
                        failed_paths = set(data) if isinstance(data, list) else set()
                except Exception as e:
                    logging.error(f"Error loading fail paths file: {e}")
            original_len = len(self.dataset)
            self.dataset = [i for i in self.dataset if Path(self.clip_paths[i]).stem not in failed_paths]
            logging.info(f"Removed {original_len - len(self.dataset)} failed paths from the dataset.")

    def __getitem__(self, index):
        if self.fast_skip:
            return 0

        dataset_index = self.dataset[index]
        clip_path = self.clip_paths[dataset_index]
        offset_sec = self.offset_secs[dataset_index]
        v_start_i_sec = self.v_start_i_secs[dataset_index]
        try:
            if hasattr(self.data_cfg, "audio_only") and self.data_cfg.audio_only:
                _, audio, meta = get_video_and_audio(
                    clip_path,
                    get_meta=True,
                    max_clip_len_sec=self.max_clip_len_sec,
                    start_sec=v_start_i_sec,
                    loading_shift=self.data_cfg.loading_shift,
                    loading_buffer=self.data_cfg.loading_buffer,
                    audio_only=True
                )
                if audio is None or meta is None:
                    raise ValueError("Audio or meta is None")
                item = {'audio': audio, 'meta': meta, 'path': clip_path, 'targets': {}, 'split': self.split}
            else:
                rgb, audio, meta = get_video_and_audio(
                    clip_path,
                    get_meta=True,
                    max_clip_len_sec=self.max_clip_len_sec,
                    start_sec=v_start_i_sec,
                    loading_shift=self.data_cfg.loading_shift,
                    loading_buffer=self.data_cfg.loading_buffer
                )
                if rgb is None or audio is None or meta is None:
                    raise ValueError("One of the video/audio/meta components is None")
                item = {'video': rgb, 'audio': audio, 'meta': meta, 'path': clip_path, 'targets': {}, 'split': self.split, "index": dataset_index}

            if self.split in self.load_fixed_offsets_on:
                item['targets']['v_start_i_sec'] = 2.1
            
            assert item["video"].shape[0] > 0 and item["audio"].shape[0] > 0, f"Shape vid {item['video'].shape[0]} aud {item['audio'].shape[0]} > 0 for path: {clip_path}"
            if self.transforms is not None:
                item = self.transforms(item)

            if item is None:
                raise ValueError("Transforms returned None")
        
        except Exception as e:
            # Log the error and update the fail_paths_file with the failed clip path.
            #raise e
            logging.error(f"Error processing index {index} (clip path: {clip_path}, max_clip_len: {self.max_clip_len_sec}, start_sec: {v_start_i_sec}): {e}")
            return self[index - 1]

        return item

    def __len__(self):
        return len(self.dataset)

    def enable_fast_skip(self):
        print("Fast Skip Enabled")
        self.fast_skip = True

    def disable_fast_skip(self):
        print("Fast Skip Disabled")
        self.fast_skip = False
