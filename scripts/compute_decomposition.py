# compute_decomposition.py

import os

import tqdm
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '.')  # nopep8
from utils.utils import cfg_sanity_check_and_patch
from main import get_config, set_env_variables

# Import necessary functions from your training utilities.
from torch.utils.data import DataLoader
from scripts.train_utils import (
    broadcast_obj,
    get_loaders,
    init_ddp,
    get_device,
    get_transforms,
    get_datasets,
    get_model,
    prepare_inputs
)

def main():
    phase = "train"
    # Load configuration and patch it
    cfg = get_config()
    cfg_sanity_check_and_patch(cfg)

    # Set environment variables and initialize DDP if running in a distributed setting.
    set_env_variables()
    
    init_ddp(cfg)
    global_rank = cfg.training.global_rank
    world_size = cfg.training.world_size
    torch.backends.cudnn.benchmark = True
    device, num_gpus = get_device(cfg)
    cfg.ckpt_path = broadcast_obj(cfg.ckpt_path, global_rank, device)
    # Instantiate the model (which creates the feature extractors) and move to device.
    model, model_without_ddp = get_model(cfg, device)
    model.eval()
    
    # Unconditionally load the checkpoint (assume we are always finetuning).
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model_without_ddp.load_state_dict(ckpt['model'], strict=False)
    print(f"Loaded checkpoint from {cfg.ckpt_path}")
    
    # Get transforms and datasets (using the training set for decomposition).
    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms)
    for dphase in list(datasets.keys()):
        if dphase != phase:
            del datasets[dphase]
    
    # Create a DataLoader (using the batch size specified in the decomposition config).
    batch_size = cfg.decomposition.get("batch_size", 32) if "decomposition" in cfg else 32
    loaders = get_loaders(cfg, datasets, {phase: batch_size // num_gpus})
    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(0)
    # Total number of tokens to use (each token is one D-dimensional embedding).
    num_samples_total = cfg.decomposition.get("num_samples", 10000) if "decomposition" in cfg else 10000
    num_samples_per_instance = cfg.decomposition.get("num_samples_per_instance", 50)
    audio_embeddings_list = []
    vision_embeddings_list = []
    total_tokens = 0
    num_layers = 12  # You can modify this if needed
    
    for batch in tqdm.tqdm(loaders[phase], total=len(loaders[phase])):
        with torch.no_grad():
            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                # Prepare the inputs; the function returns audio, video, original audio, targets, and others.
                aud, vid, _, _, _ = prepare_inputs(batch, device, phase)
                
                # Extract features from the model's feature extractors.
                vis_feats, aud_feats = model(vid, aud)
                
                # Suppose aud_feats has shape (B, S, ta, D) and vis_feats (B, S, tv, D).
                # Flatten the segments so that each row is one D-dimensional token.
                B, S, ta, D = aud_feats.shape
                aud_feats = aud_feats.view(B, S * ta, D)
                B, S, tv, D = vis_feats.shape
                vis_feats = vis_feats.view(B, S * tv, D)
                
                # For each instance in the batch, randomly sample num_samples_per_instance tokens.
                selected_aud = []
                selected_vis = []
                for i in range(B):
                    n_tokens = aud_feats[i].shape[0]
                    n_to_sample = min(num_samples_per_instance, n_tokens)
                    # Randomly sample indices without replacement.
                    indices = torch.randperm(n_tokens)[:n_to_sample]
                    selected_aud.append(aud_feats[i][indices])
                    selected_vis.append(vis_feats[i][indices])
                
                # Concatenate the selected tokens across the batch.
                selected_aud = torch.cat(selected_aud, dim=0)
                selected_vis = torch.cat(selected_vis, dim=0)
                
                # Append these tokens (move to CPU for later SVD/PCA computation).
                audio_embeddings_list.append(selected_aud.cpu())
                vision_embeddings_list.append(selected_vis.cpu())
                
                total_tokens += selected_aud.shape[0]
                if total_tokens >= num_samples_total:
                    break

    # Concatenate tokens from all batches.
    audio_embeddings = torch.cat(audio_embeddings_list, dim=0).view(-1, D)
    vision_embeddings = torch.cat(vision_embeddings_list, dim=0).view(-1, D)
    
    # Optionally, limit to the requested number of samples.
    if audio_embeddings.shape[0] > num_samples_total:
        audio_embeddings = audio_embeddings[:num_samples_total]
    if vision_embeddings.shape[0] > num_samples_total:
        vision_embeddings = vision_embeddings[:num_samples_total]
    
    # --- Distributed Gathering ---
    # If running in DDP, each process has a subset of embeddings. We gather all embeddings
    # so that rank 0 can compute the SVD.
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_audio = [None for _ in range(world_size)]
        gathered_vision = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_audio, audio_embeddings)
        dist.all_gather_object(gathered_vision, vision_embeddings)
        if dist.get_rank() == 0:
            audio_embeddings = torch.cat(gathered_audio, dim=0)
            vision_embeddings = torch.cat(gathered_vision, dim=0)
        else:
            audio_embeddings = None
            vision_embeddings = None
    
    # Only rank 0 computes the SVD and saves the results.
    if (not dist.is_initialized()) or (dist.get_rank() == 0):
        print("Computing SVD for audio embeddings...")
        U_aud, S_aud, Vh_aud = torch.linalg.svd(audio_embeddings, full_matrices=False)
    
        print("Computing SVD for vision embeddings...")
        U_vis, S_vis, Vh_vis = torch.linalg.svd(vision_embeddings, full_matrices=False)
    
        output_audio = cfg.decomposition.get("output_audio", "audio_decomposition.pt") if "decomposition" in cfg else "audio_decomposition.pt"
        output_vision = cfg.decomposition.get("output_vision", "vision_decomposition.pt") if "decomposition" in cfg else "vision_decomposition.pt"
    
        torch.save({"U": U_aud, "S": S_aud, "Vh": Vh_aud}, output_audio)
        torch.save({"U": U_vis, "S": S_vis, "Vh": Vh_vis}, output_vision)
    
        print(f"Saved audio decomposition to {output_audio}")
        print(f"Saved vision decomposition to {output_vision}")

if __name__ == '__main__':
    main()
