import csv
import os
from pathlib import Path
import random
from time import sleep
import torchaudio
import torchvision
from glob import glob
import shutil
import sys
sys.path.append('..')
from utils.utils import get_fixed_off_fname
import json
import s3fs
import torch
import itertools
import logging



def maybe_cache_file(path: os.PathLike):
    """
    If LOCAL_SCRATCH is set, copies the file from a shared disk (or S3) to a faster local cache.
    For local files, if the original file is under a directory named 's3_cache',
    that prefix is removed before appending to LOCAL_SCRATCH.
    """
    path = os.path.expanduser(path)
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        
        if "s3://" in path:
            # For S3 files, use the path relative to "s3://"
            relative = Path(path).relative_to("s3://")
            cache_path = os.path.join(cache_dir, relative.as_posix())
        else:
            p = Path(path)
            # Attempt to remove everything up to and including "s3_cache"
            try:
                idx = p.parts.index("s3_cache")
                # Use only the parts after "s3_cache"
                relative_parts = p.parts[idx+1:]
            except ValueError:
                # If "s3_cache" isn't found, fallback to using the entire path minus root
                filter_part = os.environ.get('FILTER_PATH_SCRATCH', '')
                relative_parts = Path(str(p).replace(filter_part, "")).parts
            cache_path = os.path.join(cache_dir, *relative_parts)
        
        # Ensure the destination directory exists
        if not os.path.exists(cache_path):
            print(f"Copying {path} -> {cache_path}")
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            if "s3://" in path:
                fs = s3fs.S3FileSystem()
                fs.get(path, cache_path)
            else:
                shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path

def stream_read_video(video_object, start=0, end=None, read_video=True, read_audio=True):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = torch.empty(0)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)
            video_frames = torch.permute(video_frames, (0, 2, 3, 1))
    audio_frames = torch.empty(0)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            audio_pts.append(frame['pts'])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, video_object.get_metadata()


def get_video_and_audio(path, get_meta=False, max_clip_len_sec=None, start_sec=None, loading_shift=None, loading_buffer=0, audio_only=True):
    torchvision.set_video_backend('video_reader')
    path = maybe_cache_file(path)
    num_threads = 4
    try:
        if start_sec != None:
            if (loading_shift is not None):
                end_sec = start_sec + max_clip_len_sec + loading_shift[1]
                start_sec = start_sec - loading_shift[0]
            video_object = torchvision.io.VideoReader(path, num_threads=num_threads)
            rgb, audio, meta = stream_read_video(video_object, start=start_sec, end=end_sec+loading_buffer)
        else:
            if (loading_shift is not None):
                end_sec = max_clip_len_sec + sum(loading_shift)
                start_sec = 0
            video_object = torchvision.io.VideoReader(path, num_threads=num_threads)
            rgb, audio, meta = stream_read_video(video_object, start=start_sec, end=end_sec+loading_buffer)
    except KeyError:
        #print(f'Problem at {path}. Trying to wait and load again...')
        #sleep(5)
        video_object = torchvision.io.VideoReader(path, num_threads=num_threads)
        rgb, audio, meta = stream_read_video(video_object, start=start_sec, end=end_sec+loading_buffer)
    try:
        assert(audio.shape[0] >= 16000 * (max_clip_len_sec + sum(loading_shift))) 
    except AssertionError:
        #print(path, file=open('failed_paths.txt', 'a'))
        return None, None, None
    rgb = rgb.permute(0, 3, 1, 2)
    audio = audio.squeeze()
    if meta == {}:
        #print('video path of failure', path)
        #print('extracted rgb and audio shapes', rgb.shape, audio.shape)
        exit(0)
    if (len(meta['video']['fps']) == 0 or len(meta['audio']['framerate']) == 0):
        return None, None, None
    meta = {
        'video': {'fps': meta['video']['fps']},
        'audio': {'framerate': meta['audio']['framerate']},
    }
    return rgb, audio, meta



def get_audio_stream(path, get_meta=False):
    '''Used only in feature extractor training'''
    path = str(Path(path).with_suffix('.wav'))
    path = maybe_cache_file(path)
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if get_meta:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        meta = {'audio': {'duration': [duration], 'framerate': [info.sample_rate]}}
        return waveform, meta
    else:
        return waveform

def subsample_dataset(dataset: list, size_ratio: float, shuffle: bool = False):
    if size_ratio is not None and 0.0 < size_ratio < 1.0:
        logging.info(f'Subsampling dataset to {size_ratio}')
        # shuffling is important only during subsampling (sometimes paths are sorted by class)
        if shuffle:
            random.shuffle(dataset)
        cut_off = int(len(dataset) * size_ratio)
        # making sure that we have at least one example
        dataset = dataset[:max(1, cut_off)]
        logging.info(f'Subsampled dataset to {size_ratio} (size: {len(dataset)})')
    return dataset
