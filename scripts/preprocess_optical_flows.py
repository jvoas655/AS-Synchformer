#!/usr/bin/env python
import json
import os
import argparse
import pathlib
import time
import pickle
import itertools

import pandas as pd

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

# Use the faster VideoReader backend.
import torchvision
from torchvision.io import VideoReader

# Import RAFT models using updated weights.
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import h5py
from concurrent.futures import ThreadPoolExecutor
import psutil
import time

# Create a global thread pool executor (you can adjust max_workers as needed).
executor = ThreadPoolExecutor(max_workers=64)



def submit_with_memory_limit(fn, *args, max_usage_percent=90, sleep_time=0.5, **kwargs):
    """Wait until system memory usage is below max_usage_percent before submitting a task."""
    while psutil.virtual_memory().percent >= max_usage_percent:
        time.sleep(sleep_time)
    return executor.submit(fn, *args, **kwargs)

# Define a helper function for asynchronous saving.
def async_save_flow(out_path, flows, original_size, timing_info, video_path, start_frame, end_frame):
    # Optionally convert flows to lower precision (e.g. float16) to reduce file size.
    flows_to_save = flows.astype(np.float16)
    os.makedirs(out_path.parent, exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("flows", data=flows_to_save, compression="gzip")
        h5f.create_dataset("original_size", data=np.array(original_size))
        # Save timing info as a group of attributes.
        timing_grp = h5f.create_group("timing_info")
        for key, value in timing_info.items():
            timing_grp.attrs[key] = value
        # Save additional metadata as attributes.
        h5f.attrs["video_path"] = video_path
        h5f.attrs["start_frame"] = start_frame
        h5f.attrs["end_frame"] = end_frame


#############################
# Efficient Video Streaming
#############################

def get_video_segment_frames(video_path, start_frame, end_frame):
    """
    Opens the video file using OpenCV and extracts frames based on frame indices.
    The segment is defined by frame indices [start_frame, end_frame).
    Returns a tensor of shape (T, H, W, C) in RGB order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path} for frame-based reading.")
    
    # Seek to the starting frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    num_frames_to_read = end_frame - start_frame
    for _ in range(num_frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break  # End of video reached before expected.
        # Convert BGR (OpenCV default) to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor.
        frame_tensor = torch.from_numpy(frame_rgb)
        frames.append(frame_tensor)
    
    cap.release()
    
    if frames:
        video_frames = torch.stack(frames, 0)
    else:
        video_frames = torch.empty(0)
        
    return video_frames

#############################
# Utility Functions
#############################

def pad_to_multiple_of_8(frames):
    """
    Pads frames so that their height and width are multiples of 8.
    Returns the padded tensor and the original (H, W).
    """
    _, _, H, W = frames.shape
    pad_H = (8 - (H % 8)) % 8
    pad_W = (8 - (W % 8)) % 8
    if pad_H > 0 or pad_W > 0:
        frames = torch.nn.functional.pad(frames, (0, pad_W, 0, pad_H), mode="replicate")
    return frames, (H, W)

def preprocess(frames):
    """
    Preprocess frames:
      - Convert to float32 in [0,1] and normalize to [-1,1]
      - Pad spatial dimensions so that H and W are multiples of 8.
    Expects frames in shape (T, H, W, C).
    Returns the padded frames (with shape (T, C, H, W)) and the original (H, W).
    """
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),  # scales to [0,1]
        T.Normalize(mean=0.5, std=0.5),        # maps to [-1,1]
    ])
    # Apply transforms on each frame; our transforms work on (H, W, C) tensors.
    frames = transforms(frames)
    # Permute to (T, C, H, W) for model processing.
    frames = frames.permute(0, 3, 1, 2)
    frames_padded, original_size = pad_to_multiple_of_8(frames)
    return frames_padded, original_size

def compute_acceleration(flows):
    a = flows[1:] - flows[:-1]
    N = flows.shape[0]
    a_full = torch.empty_like(flows)
    a_full[0] = a[0]
    a_full[-1] = a[-1]
    if N > 2:
        a_full[1:-1] = (a[:-1] + a[1:]) / 2
    return a_full

def compute_jerk(acceleration):
    j = acceleration[1:] - acceleration[:-1]
    N = acceleration.shape[0]
    j_full = torch.empty_like(acceleration)
    j_full[0] = j[0]
    j_full[-1] = j[-1]
    if N > 2:
        j_full[1:-1] = (j[:-1] + j[1:]) / 2
    return j_full

def render_color_map(scalar_field, colormap=cv2.COLORMAP_JET):
    norm = cv2.normalize(scalar_field, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm_uint8, colormap)
    return colored

def save_video(frames, out_path, fps=25):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

def get_video_frame_count_and_fps(video_path):
    """
    Returns the total frame count and FPS of the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path} for reading frame count and FPS.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

#############################
# Dataset for Video Segments
#############################

class VideoSegmentDataset(Dataset):
    def __init__(self, segments):
        """
        segments: list of tuples (video_path, start_frame, end_frame)
        """
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        video_path, start_frame, end_frame = self.segments[idx]
        frames = get_video_segment_frames(video_path, start_frame, end_frame)
        return {
            "video_path": str(video_path),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": frames
        }

#############################
# Optical Flow Processing for a Segment
#############################

def process_segment(segment, device, model, output_dir=pathlib.Path(".")):
    """
    Processes a video segment:
      - Preprocesses frames.
      - Computes optical flows for all adjacent frame pairs.
      - Saves results asynchronously as an HDF5 file.
    Returns the computed flows, original frame size, timing info, and output path.
    """
    video_path = segment["video_path"]
    start_frame = segment["start_frame"]
    end_frame = segment["end_frame"]
    frames = segment["frames"]

    # Construct output path and asynchronously save using HDF5.
    video_stem = pathlib.Path(video_path).stem
    out_fname = f"{video_stem}_{start_frame}_{end_frame}.h5"
    out_path = pathlib.Path(output_dir) / out_fname

    timing_info = {}
    start_total = time.perf_counter()

    if frames.size(0) < 2:
        raise ValueError(f"Segment {video_path} [{start_frame}-{end_frame}] has less than 2 frames; skipping.")

    # Preprocessing.
    t0 = time.perf_counter()
    frames_padded, original_size = preprocess(frames)
    timing_info["prep_time"] = time.perf_counter() - t0

    num_frames = frames_padded.size(0)
    # Compute optical flow in one batch.
    t0 = time.perf_counter()
    model.eval()
    tried_half_batch = False
    try:
        with torch.no_grad():
            img1_batch = frames_padded[:-1].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
            img2_batch = frames_padded[1:].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
            flows_iter = model(img1_batch, img2_batch)
            flows_all = flows_iter[-1].to(torch.float32).cpu()  # Shape: (T-1, 2, H, W)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and not tried_half_batch:
            print(f"[INFO] Trying Half Batch {e}...")
            tried_half_batch = True
            torch.cuda.empty_cache()
            if not frames_padded.device.type == 'cuda':
                raise e  # should not happen, but just in case.
            # Split the frame pairs into two halves.
            N = num_frames
            total_pairs = N - 1
            half = total_pairs // 2
            flows_list = []
            with torch.no_grad():
                # Process first half if it exists.
                if half > 0:
                    img1_batch1 = frames_padded[:half+1].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
                    img2_batch1 = frames_padded[1:half+1].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
                    flows_iter1 = model(img1_batch1, img2_batch1)
                    flows1 = flows_iter1[-1].to(torch.float32).cpu()
                    flows_list.append(flows1)
                # Process second half.
                img1_batch2 = frames_padded[half:N-1].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
                img2_batch2 = frames_padded[half+1:N].to(device, dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
                flows_iter2 = model(img1_batch2, img2_batch2)
                flows2 = flows_iter2[-1].to(torch.float32).cpu()
                flows_list.append(flows2)
            # Concatenate the flows from the two halves.
            flows_all = torch.cat(flows_list, dim=0)
        else:
            raise e
    timing_info["flow_time"] = time.perf_counter() - t0

    # Create the full flow tensor and interpolate the last missing flow.
    flows_full = torch.empty((num_frames, 2, flows_all.shape[-2], flows_all.shape[-1]), dtype=flows_all.dtype)
    flows_full[:-1] = flows_all
    flows_full[-1] = flows_all[-1]
    orig_H, orig_W = original_size
    flows_full = flows_full[:, :, :orig_H, :orig_W]
    assert flows_full.shape[2] == orig_H and flows_full.shape[3] == orig_W, \
        "Flow output shape does not match original resolution"
    timing_info["total_time"] = time.perf_counter() - start_total

    # Submit the save operation to the thread pool.
    submit_with_memory_limit(
        async_save_flow,
        out_path,
        flows_full.numpy(),   # Convert torch.Tensor to NumPy array.
        original_size,
        timing_info,
        video_path,
        start_frame,
        end_frame,
    )

    return flows_full, original_size, timing_info, out_path


#############################
# Main Function
#############################

def main():
    parser = argparse.ArgumentParser(
        description="Precompute optical flows for fixed-frame video segments with distributed DataLoader. "
                    "Each video is split into non-overlapping segments defined by a fixed number of frames. "
                    "For each segment, optical flow is computed and saved as an HDF5 file; one segment gets visualization videos."
    )
    parser.add_argument("--root_path", type=str, required=True,
                        help="Path to the root directory containing video files.")
    parser.add_argument("--skip_file", type=str, required=True,
                        help="Json list file of videos to skip.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where HDF5 outputs will be stored.")
    parser.add_argument("--segment_frame_count", type=int, default=16,
                        help="Number of frames per segment.")
    parser.add_argument("--model", type=str, choices=["raft_small", "raft_large"], default="raft_small",
                        help="Which RAFT model to use for optical flow.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers for video segment loading.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-essential logging output.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed processing (or use LOCAL_RANK env var).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to a fixed number of videos for debugging.")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Setup device and distributed processing.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1

    # Load the optical flow model.
    if args.model == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    model = model.to(device, dtype=torch.float32)
    model.eval()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load the list of video files to skip.
    root_path = pathlib.Path(os.path.expanduser(args.root_path))
    if not root_path.exists():
        raise ValueError(f"Root path {root_path} does not exist.")
    with open(pathlib.Path(os.path.expanduser(args.skip_file)), "r") as fileref:
        skip_videos = json.load(fileref)

    # Load the CSV and keep only the key and channel columns.
    video_sources = pd.read_csv(
        pathlib.Path(os.path.expanduser(args.root_path)).parent / "script" / "all_uniform.csv"
    )[["key", "channel"]]

    # Extract the video id from the key (assuming key format: "{id}_{start}_{stop}").
    video_sources["id"] = video_sources["key"].apply(lambda x: x.split("_")[0])

    # Check for duplicate ids with conflicting channels.
    # This groups the data by 'id' and ensures that each group has only one unique channel.
    id_to_channel = {}
    for vid, group in video_sources.groupby("id"):
        unique_channels = group["channel"].unique()
        if len(unique_channels) > 1:
            raise ValueError(f"Duplicate id '{vid}' has different channels: {unique_channels}")
        id_to_channel[vid] = unique_channels[0]

    # Identify all ids that do NOT have the channel "Ego4DSounds-test".
    # (Change the channel name as needed if there are extra spaces or variations.)
    non_sounds_videos = [vid for vid, ch in id_to_channel.items() if "Ego4DSounds" not in ch]

    # Update the list of videos to skip by adding these ids.
    # Using a set ensures there are no duplicates.
    print(len(set(non_sounds_videos)), len(set(skip_videos)))
    skip_videos = list(set(skip_videos) | set(non_sounds_videos))

    # (Optional) Prepare a list of video file paths, skipping any whose id is in skip_videos.
    # Assuming that the id is part of the video file's name.
    video_paths = sorted(
        [
            p for p in root_path.rglob("*")
            if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]
            and (p.stem not in skip_videos)
        ]
    )

    if not video_paths:
        raise ValueError("No video files found in the specified root_path after applying skip_videos.")
    
    if (args.limit > 0 and args.limit < len(video_paths)):
        print(f"[INFO] Limiting videos to {args.limit} max")
        video_paths = video_paths[:args.limit]

    # Preprocessing: For each video, split it into non-overlapping segments based on frame count.
    segments = []
    seg_frame_count = args.segment_frame_count
    if not args.quiet and rank == 0:
        print(f"[INFO] Preprocessing videos to create segments (segment_frame_count={seg_frame_count} frames)...")
    for video_path in tqdm(video_paths, total=len(video_paths)):
        try:
            total_frames, fps = get_video_frame_count_and_fps(video_path)
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] Skipping {video_path} due to error reading frame count: {e}")
            continue
        start_frame = 0
        while start_frame < total_frames:
            end_frame = start_frame + seg_frame_count
            if end_frame > total_frames:
                # If the last segment would have only one frame, extend it by including one extra frame
                if (total_frames - start_frame) == 1 and start_frame > 0:
                    start_frame = start_frame - 1  # Extend the segment backwards by one frame
                    end_frame = total_frames
                else:
                    end_frame = total_frames
            video_stem = pathlib.Path(video_path).stem
            out_fname = f"{video_stem}_{start_frame}_{end_frame}.h5"
            out_path = pathlib.Path(pathlib.Path(args.output_dir)) / out_fname
            if (not os.path.exists(out_path)):
                segments.append((video_path, start_frame, end_frame))
            start_frame = end_frame  # non-overlapping segments

    if not segments:
        raise ValueError("No segments to process after preprocessing.")

    # Create dataset and distributed DataLoader for segments.
    dataset = VideoSegmentDataset(segments)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True)

    # Set output directories relative to the parent of root_path.
    output_root = pathlib.Path(args.output_dir)
    video_output_root = pathlib.Path(f"{args.output_dir}_videos")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(video_output_root, exist_ok=True)

    # Process segments.
    cumulative_times = {"prep_time": 0, "flow_time": 0, "total_time": 0}
    processed = 0
    vis_done = False  # Only one segment will get visualization.
    pbar = tqdm(dataloader, desc=f"Rank {rank}/{world_size} segments", unit="segment")
    for batch in pbar:
        segment = {
            "video_path": batch["video_path"][0],
            "start_frame": int(batch["start_frame"][0]),
            "end_frame": int(batch["end_frame"][0]),
            "frames": batch["frames"][0]
        }
        try:
            flows, original_size, timing_info, out_path = process_segment(segment, device, model,
                                                                           output_dir=output_root)
        except Exception as e:
            if not args.quiet:
                pbar.write(f"[ERROR] Processing segment {segment['video_path']} [{segment['start_frame']} - {segment['end_frame']}] failed: {e}")
            continue

        for k in cumulative_times:
            cumulative_times[k] += timing_info.get(k, 0)
        processed += 1
        avg_times = {k: cumulative_times[k] / processed for k in cumulative_times}
        pbar.set_postfix(avg_times)

        # Save visualization for the very first processed segment.
        if not vis_done and processed == 1:
            try:
                acceleration = compute_acceleration(flows)
                jerk = compute_jerk(acceleration)
                flows_np = flows.numpy()  # shape (T, 2, H, W)
                acc_np = acceleration.numpy()
                jerk_np = jerk.numpy()
                def compute_magnitude(x):
                    return np.sqrt(x[0]**2 + x[1]**2)
                flow_frames = []
                acc_frames = []
                jerk_frames = []
                N = flows_np.shape[0]
                for i in range(N):
                    flow_mag = compute_magnitude(flows_np[i])
                    acc_mag = compute_magnitude(acc_np[i])
                    jerk_mag = compute_magnitude(jerk_np[i])
                    flow_frames.append(render_color_map(flow_mag))
                    acc_frames.append(render_color_map(acc_mag))
                    jerk_frames.append(render_color_map(jerk_mag))
                base_name = pathlib.Path(segment["video_path"]).stem
                flow_vid_path = video_output_root / f"{base_name}_{segment['start_frame']}_{segment['end_frame']}_flow.mp4"
                acc_vid_path = video_output_root / f"{base_name}_{segment['start_frame']}_{segment['end_frame']}_acceleration.mp4"
                jerk_vid_path = video_output_root / f"{base_name}_{segment['start_frame']}_{segment['end_frame']}_jerk.mp4"
                save_video(flow_frames, str(flow_vid_path), fps=25)
                save_video(acc_frames, str(acc_vid_path), fps=25)
                save_video(jerk_frames, str(jerk_vid_path), fps=25)
                if not args.quiet:
                    pbar.write(f"[INFO] Saved visualization videos for segment {base_name} [{segment['start_frame']}-{segment['end_frame']}].")
                vis_done = True
            except Exception as e:
                pbar.write(f"[ERROR] Failed to create visualization videos: {e}")
        del flows
    pbar.close()

    if processed > 0 and not args.quiet:
        avg_times = {k: cumulative_times[k] / processed for k in cumulative_times}
        print(f"[INFO] Processed {processed} segments. Average timings (s): {avg_times}")
    else:
        print("[INFO] No segments processed.")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
