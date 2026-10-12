#!/usr/bin/env python
import os
import argparse
import pathlib
import json
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch.distributed as dist

def get_video_frame_count(video_path):
    """Return the total number of frames in the video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path} to read frame count.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def load_flow_h5(flow_file):
    """Load a single HDF5 file containing flows."""
    with h5py.File(flow_file, "r") as h5f:
        flows = h5f["flows"][:]  # shape: (T, 2, H, W)
        original_size = tuple(h5f["original_size"][:])
    return flows, original_size

def save_combined_flow_h5(out_file, flows, original_size, metadata=None):
    """
    Save the combined flows (NumPy array) in an HDF5 file.
    metadata is an optional dict that can be stored in attributes.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as h5f:
        h5f.create_dataset("flows", data=flows, compression="gzip")
        h5f.create_dataset("original_size", data=np.array(original_size))
        if metadata:
            meta_grp = h5f.create_group("metadata")
            for k, v in metadata.items():
                meta_grp.attrs[k] = v

def combine_video_flows(video_path, flows_dir, output_dir):
    """
    For a given base video, find all per-segment HDF5 flow files (expected naming format):
      {video_stem}_{start_frame}_{end_frame}.h5
    in the flows_dir. Load, sort them by start_frame, remove overlapping frames if present,
    and then concatenate the flows.
    Finally, check that the number of unique flow frames matches the video frame count,
    and save the combined flow and JSON stats.
    """
    video_path = pathlib.Path(video_path)
    video_stem = video_path.stem

    output_dir = pathlib.Path(output_dir)
    flag_file = output_dir / f"{video_stem}_combined.done"
    # Skip if flag file exists (i.e., video was processed successfully already)
    if flag_file.exists():
        print(f"[INFO] Skipping {video_stem} because flag file exists: {flag_file}")
        return

    # Find all flow files in flows_dir with names starting with video_stem + "_"
    flows_dir = pathlib.Path(flows_dir)
    flow_files = sorted(list(flows_dir.glob(f"{video_stem}_*_*.h5")))
    if not flow_files:
        #print(f"[WARN] No flow files found for video {video_path}")
        return

    # Parse filenames using rsplit to extract start_frame and end_frame
    segments = []
    for f in flow_files:
        try:
            # Split from the right into three parts: [video_stem, start_frame, end_frame]
            parts = f.stem.rsplit("_", 2)
            if len(parts) != 3:
                raise ValueError("Filename does not have three parts")
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            segments.append((start_frame, end_frame, f))
        except Exception as e:
            print(f"[WARN] Could not parse flow file name {f.name}: {e}")
    if not segments:
        print(f"[WARN] No valid flow segments found for {video_stem}")
        return
    segments.sort(key=lambda x: x[0])
    
    combined_flows = []
    original_size = None
    prev_end_frame = None
    for start_frame, end_frame, flow_file in segments:
        flows, orig_size = load_flow_h5(flow_file)
        if original_size is None:
            original_size = orig_size
        else:
            if original_size != orig_size:
                print(f"[WARN] Mismatch in original_size between segments in {video_stem}")
        # If this segment overlaps with the previous segment, remove duplicate frames.
        if prev_end_frame is not None:
            overlap = prev_end_frame - start_frame
            if overlap > 0:
                # Remove the first 'overlap' frames from this segment.
                flows = flows[overlap:]
        combined_flows.append(flows)
        prev_end_frame = end_frame

    combined_flows = np.concatenate(combined_flows, axis=0)  # (T_total, 2, H, W)

    try:
        total_frames = get_video_frame_count(video_path)
    except Exception as e:
        print(f"[ERROR] Could not get frame count for {video_path}: {e}")
        return

    combined_frames = combined_flows.shape[0]
    if combined_frames != total_frames:
        print(f"[ERROR] Mismatch in frame count for {video_stem}: combined flows have {combined_frames} frames but video has {total_frames} frames.")
    else:
        print(f"[INFO] Frame count verified for {video_stem}: {combined_frames} frames.")

    out_flow_file = output_dir / f"{video_stem}_combined.h5"
    save_combined_flow_h5(out_flow_file, combined_flows, original_size,
                          metadata={"source_video": str(video_path),
                                    "frame_count": combined_frames})

    # Create flag file to mark the video as completely processed.
    flag_file.touch()
    print(f"[INFO] Saved combined flow to {out_flow_file} and created flag file {flag_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine per-segment flow files for each video and verify frame counts. "
                    "Also compute per-video statistics (mean, std, var, min, max)."
    )
    parser.add_argument("--videos", type=str, required=True,
                        help="Path to the directory containing the original video files.")
    parser.add_argument("--flows", type=str, required=True,
                        help="Path to the directory containing per-segment flow HDF5 files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for combined flows and JSON stats.")
    args = parser.parse_args()

    # Setup distributed processing if available.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    videos_path = pathlib.Path(args.videos)
    video_files = sorted([p for p in videos_path.rglob("*") if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]])
    if not video_files:
        print("[ERROR] No video files found in the specified videos directory.")
        return

    # Distribute work: Each process handles videos where (index mod world_size == rank)
    video_files = [v for i, v in enumerate(video_files) if i % world_size == rank]
    if not video_files:
        print(f"[INFO] Rank {rank} has no videos to process.")
        return

    for video_file in tqdm(video_files, desc=f"Rank {rank} processing videos"):
        try:
            combine_video_flows(video_file, args.flows, args.output)
        except Exception as e:
            print(f"[ERROR] Failed to process {video_file}: {e}")

if __name__ == "__main__":
    main()
