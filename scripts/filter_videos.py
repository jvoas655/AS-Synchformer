import os
import json
import argparse
import pathlib
from tqdm import tqdm
from torchvision.io import VideoReader
import torchvision
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def process_video(video_path, skip_full_check=False):
    torchvision.set_video_backend('video_reader')
    result = {
        "sub_id": pathlib.Path(video_path).stem,
        "video_duration": None,
        "audio_duration": None,
        "missing_audio": 0.0,
        "missing_audio_count": 0.0,
        "accepted": False,
        "error": None
    }
    try:
        vr_video = VideoReader(video_path, "video")
        video_meta = vr_video.get_metadata()
        video_duration = video_meta.get('video', {}).get('duration', None)
        fps = video_meta.get('video', {}).get('fps', None)
        if video_duration is None or video_duration[0] <= 0:
            raise Exception("Invalid or missing video duration")
        if fps is None or fps[0] <= 0:
            raise Exception("Invalid or missing video FPS")
        result["video_duration"] = video_duration[0]
        frame_tolerance = 0.8
    except Exception as e:
        result["error"] = "Video stream error: " + str(e)
        return result

    try:
        vr_audio = VideoReader(video_path, "audio")
        audio_meta = vr_audio.get_metadata()
        audio_duration = audio_meta.get('audio', {}).get('duration', None)
        sample_rate = audio_meta.get('audio', {}).get('framerate', None)
        if audio_duration is None or audio_duration[0] <= 0:
            raise Exception("Invalid or missing audio duration")
        if sample_rate is None or sample_rate[0] <= 0:
            raise Exception("Invalid or missing audio sample rate")
        result["audio_duration"] = audio_duration[0]
    except Exception as e:
        result["error"] = "Audio stream error: " + str(e)
        return result

    if (result["video_duration"] - result["audio_duration"]) > frame_tolerance:
        missing = (result["video_duration"] - result["audio_duration"])
        result["missing_audio"] = missing
        result["missing_audio_count"] = 1
        result["error"] = f"Audio missing for {missing:.3f} seconds"
        return result

    if not skip_full_check:
        try:
            vr_video = VideoReader(video_path, "video")
            for _ in vr_video:
                pass
            vr_audio = VideoReader(video_path, "audio")
            for _ in vr_audio:
                pass
        except Exception as e:
            result["error"] = "Error loading frames or audio samples: " + str(e)
            return result

    result["accepted"] = True
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate video files for full audio/video integrity and collect statistics."
    )
    parser.add_argument("--root_path", type=str, required=True,
                        help="Path to the root directory containing video files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the JSON output file for failed sub_ids.")
    parser.add_argument("--subset_limit", type=int, default=0,
                        help="Optional: Limit processing to the first N video files (default: process all).")
    parser.add_argument("--skip_full_check", action="store_true",
                        help="Skip the full check (iterating through all frames and audio samples).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers to use (default: os.cpu_count()).")
    args = parser.parse_args()

    root_path = os.path.expanduser(args.root_path)
    output_file = os.path.expanduser(args.output)
    workers = args.workers or os.cpu_count()

    video_entries = [entry for entry in os.scandir(root_path) if entry.is_file()]
    if args.subset_limit > 0:
        video_entries = video_entries[:args.subset_limit]
    video_paths = [os.path.join(root_path, e.name) for e in video_entries]

    failed_ids = []
    total_rejected_duration = 0.0
    total_accepted_duration = 0.0
    total_missing_audio = 0.0
    num_failures = 0
    missing_audio_counts = 0

    results = []
    print(f"Starting processing with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        worker_fn = partial(process_video, skip_full_check=args.skip_full_check)
        futures = {executor.submit(worker_fn, path): path for path in video_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing videos", unit="video"):
            res = f.result()
            results.append(res)

    for res in results:
        duration = res.get("video_duration") or 0.0
        if res["accepted"]:
            total_accepted_duration += duration
        else:
            failed_ids.append(res["sub_id"])
            total_rejected_duration += duration
            total_missing_audio += res.get("missing_audio", 0.0)
            num_failures += 1
            missing_audio_counts += res.get("missing_audio_count", 0.0)
            print(f"[ERROR] {res['sub_id']}: {res['error']}")

    with open(output_file, "w") as f:
        json.dump(sorted(list(set(failed_ids))), f, indent=2)

    mean_missing_audio = total_missing_audio / missing_audio_counts if missing_audio_counts > 0 else 0.0

    print(f"\nFinished processing {len(video_paths)} videos.")
    print(f"Accepted video duration: {total_accepted_duration:.2f} seconds")
    print(f"Rejected video duration: {total_rejected_duration:.2f} seconds")
    print(f"Number of perceptual failures: {num_failures}")
    print(f"Mean missing audio duration in partial videos ({missing_audio_counts}): {mean_missing_audio*100:.3f}%")
    print(f"{len(failed_ids)} failed IDs written to {output_file}.")

if __name__ == "__main__":
    main()
