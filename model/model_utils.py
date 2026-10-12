from collections import defaultdict
import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from contextlib import contextmanager
from torchvision.io import write_video
from torchaudio import transforms
import re

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config, instantiate_cfg_from_config


@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def contrastive_cosine_self_loss(embeddings, temp=0.2):
    # Normalize embeddings along the feature dimension
    embeddings = F.normalize(embeddings, dim=-1)
    
    B, S, D = embeddings.shape
    
    # Compute pairwise cosine similarities within each sequence
    similarity_matrix = torch.matmul(embeddings, embeddings.transpose(1, 2)) / temp
    
    # Create labels: each embedding in the sequence is its own positive
    labels = torch.arange(S).unsqueeze(0).expand(B, S).to(embeddings.device)
    
    # Reshape similarity_matrix and labels for batch-wise computation
    similarity_matrix = similarity_matrix.reshape(B * S, S)
    labels = labels.reshape(B * S)
    
    # Compute cross-entropy loss for contrastive pairs
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def contrastive_cosine_other_loss(embeddings1, embeddings2, temp=0.2):
    # Normalize embeddings along the feature dimension
    embeddings1 = F.normalize(embeddings1, dim=-1)
    embeddings2 = F.normalize(embeddings2, dim=-1)
    
    B, S, D = embeddings1.shape
    
    # Compute pairwise cosine similarities within each sequence
    similarity_matrix = torch.matmul(embeddings1, embeddings2.transpose(1, 2)) / temp
    
    # Create labels: each embedding in the sequence is its own positive
    labels = torch.arange(S).unsqueeze(0).expand(B, S).to(embeddings2.device)
    
    # Reshape similarity_matrix and labels for batch-wise computation
    similarity_matrix = similarity_matrix.reshape(B * S, S)
    labels = labels.reshape(B * S)
    
    # Compute cross-entropy loss for contrastive pairs
    loss = F.cross_entropy(similarity_matrix, labels)
    loss /= math.log(S-1)
    return loss

def create_k_hot_tensor_per_batch_seed(B, S, K, seeds=None):
    """
    Create a tensor of shape (B, S) where K values in each batch are 1.0, and the rest are 0.0.
    Each batch entry can use a different seed.
    
    Args:
        B (int): Batch size.
        S (int): Sequence length.
        K (int): Number of 1.0s in each batch.
        seeds (list or Tensor of floats, optional): List of seeds, one for each batch entry.

    Returns:
        torch.Tensor: A tensor of shape (B, S) with K ones per batch.
    """
    assert seeds is None or len(seeds) == B, "Number of seeds must match batch size if provided."
    
    # Initialize the tensor with zeros
    tensor = torch.zeros(B, S)
    
    for b in range(B):
        # Generate a per-batch random generator with the given seed (if provided)
        if seeds is not None:
            generator = torch.Generator()
            # Ensure seeds[b] is a scalar
            seed_value = seeds[b].item() if isinstance(seeds[b], torch.Tensor) else seeds[b]
            generator.manual_seed(int(seed_value * 1e6))  # Scale to create a valid seed
        else:
            generator = None  # Use default randomness
        
        # Randomly select K indices for the batch
        indices = torch.randperm(S, generator=generator)[:K]
        tensor[b, indices] = 1.0
    
    return tensor

def compute_loss(
    logits: torch.Tensor,
    targets: dict = None,
    prefix: str = "",
    losses=None,
    scale: float = 1.0,
    loss_fns=None
):
    if losses is None:
        losses = defaultdict(list)

    perf_targets = None
    if targets is not None:
        if "cross_entropy" in loss_fns:
            # 1) Cross‑entropy loss
            ce_loss = F.cross_entropy(
                logits,
                targets["offset_target"].long()
            )
            losses[f"loss_{prefix}CE"].append(ce_loss * scale)
        
        # 2) Earth Mover’s Distance (Wasserstein‑1) loss
        #    – convert logits → probabilities
        probs = F.softmax(logits, dim=1)
        #    – build one‑hot ground truth distribution
        target_idx = targets["offset_target"].long()
        one_hot = F.one_hot(target_idx, num_classes=probs.size(1)).float()
        #    – compute CDFs
        cdf_pred = torch.cumsum(probs, dim=1)
        cdf_true = torch.cumsum(one_hot, dim=1)
        #    – EMD = mean over batch of L1 distance between CDFs
        emd_loss = torch.sum(torch.abs(cdf_pred - cdf_true), dim=1)
        if "EMD" in loss_fns:
            losses[f"loss_{prefix}EMD"].append(torch.mean(emd_loss) * scale)
        perf_targets = emd_loss.detach()
    return losses, perf_targets


def save_melspectrogram_as_png(audio_tensor: torch.Tensor, 
                               sample_rate: int, 
                               out_path: Path):
    """
    Computes a mel spectrogram from an audio waveform and saves it as a PNG image.

    Args:
        audio_tensor (torch.Tensor): 1D tensor of audio samples.
        sample_rate (int): Sample rate of the audio (e.g., 16000 Hz).
        out_path (Path): Path to save the PNG file.
    """

    # Ensure audio is on CPU and convert to NumPy
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    if audio_tensor.dim() > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)  # Convert to mono if needed

    # Compute the mel spectrogram
    mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=0,
        f_max=sample_rate // 2
    )
    mel_spectrogram = mel_transform(audio_tensor)  # Shape: (n_mels, time)

    # Convert to log scale for visualization (avoid log(0))
    mel_spectrogram = mel_spectrogram.numpy()
    mel_spectrogram_db = np.log(mel_spectrogram + 1e-7)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()

    # Save as PNG
    plt.savefig(out_path)
    plt.close()

def find_next_index(top_level_dir: Path) -> int:
    """
    Scan existing subdirectories named 'sample_<int>' and return the next index.
    """
    pattern = re.compile(r"sample_(\d+)")
    indices = []
    for subdir in top_level_dir.iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                indices.append(int(match.group(1)))
    return 0 if not indices else (max(indices) + 1)


def save_full_av_video(full_video_tensor: torch.Tensor,
                       full_audio_tensor: torch.Tensor,
                       out_path: Path,
                       video_fps: int = 25,
                       audio_sample_rate: int = 16000,
                       audio_codec: str = "aac"):
    """
    Writes out a combined audio+video file using `torchvision`.
    full_video_tensor shape: (num_frames, 3, H, W)
    full_audio_tensor shape: (num_samples,)
    """
    # Ensure CPU tensors
    if full_video_tensor.is_cuda:
        full_video_tensor = full_video_tensor.cpu()
    if full_audio_tensor.is_cuda:
        full_audio_tensor = full_audio_tensor.cpu()

    # Normalize video to [0, 255] and convert to uint8
    if full_video_tensor.dtype != torch.uint8:
        full_video_tensor = (full_video_tensor * 255).clamp(0, 255).byte()

    # Ensure audio is float32 and normalized between [-1, 1]
    if full_audio_tensor.dtype != torch.float32:
        # Avoid dividing by zero if the audio is all zeros
        denom = full_audio_tensor.abs().max() if full_audio_tensor.abs().max() > 0 else 1.0
        full_audio_tensor = (full_audio_tensor.float() / denom).clamp(-1.0, 1.0)

    # Add a channel dimension to the audio array for write_video
    full_audio_tensor = full_audio_tensor.unsqueeze(0)  # Shape: (1, num_samples)

    write_video(
        filename=str(out_path),
        video_array=full_video_tensor.permute(0, 2, 3, 1),  # (T, H, W, C)
        fps=video_fps,
        audio_array=full_audio_tensor,  # (channels, samples)
        audio_fps=audio_sample_rate,
        audio_codec=audio_codec
    )

def visualize_tensors(
    tensors_list,
    labels=None,
    out_dir: str = "~/temp",
    prefix: str = "fft_visualization"
):
    # Ensure execution on GPU 0 only
    if torch.cuda.current_device() != 0:
        return  # Exit early if not on GPU 0

    # Expand user path and create directory if necessary
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Find the next available iteration number based on existing files
    existing_files = glob.glob(os.path.join(out_dir, f"{prefix}_batch_*.png"))
    existing_iterations = sorted(
        [int(os.path.basename(f).split('_batch_')[1].split('.')[0]) for f in existing_files]
    )
    next_iteration = existing_iterations[-1] + 1 if existing_iterations else 0

    # Assuming all tensors have the same batch size B
    B = tensors_list[0].shape[0]
    num_tensors = len(tensors_list)

    # Default labels if none provided
    if labels is None or len(labels) != num_tensors:
        labels = [f"Tensor {i}" for i in range(num_tensors)]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_tensors,
        figsize=(6 * num_tensors, 6),
        squeeze=False
    )

    # Flatten axes for easier indexing
    axes = axes[0]

    for idx, tensor in enumerate(tensors_list):
        ax = axes[idx]
        data = tensor.detach().cpu().numpy()
        B, S, D = data.shape  # Ensure correct shape

        # Handle cases where data is in shape (B, S, 2) or (B, S, 3)
        if D in [2, 3]:
            batch_indices = np.repeat(np.arange(B), S)
            points = data.reshape(-1, D)
            scatter = ax.scatter(points[:, 0], points[:, 1], c=batch_indices, cmap='viridis', alpha=0.6)
            ax.set_title(f"{labels[idx]}\nShape={data.shape}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(scatter, ax=ax, label="Batch Index")
        
        # Handle other visualizations
        elif data.ndim == 2 and data.shape[-1] % 2 == 0:
            im = ax.imshow(
                data.T,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            ax.set_title(f"{labels[idx]}\nShape={data.shape}\nMagnitude")
            fig.colorbar(im, ax=ax)
        elif data.ndim == 2:
            im = ax.imshow(
                data.T,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            ax.set_title(f"{labels[idx]}\nShape={data.shape}")
            fig.colorbar(im, ax=ax)
        elif data.ndim == 1:
            ax.plot(data)
            ax.set_title(f"{labels[idx]}\nShape={data.shape}")
        else:
            im = ax.imshow(
                np.squeeze(data),
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            ax.set_title(f"{labels[idx]}\nShape={data.shape}")
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{prefix}_batch_{next_iteration}.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved visualization to {save_path}")


def vae_loss(recon_x, x, mu, logvar):
    BCE = contrastive_cosine_other_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def multi_scale_fft_processing(input):
    """
    Processes the input tensor with sliding windows of decreasing scales,
    applies upscaling, FFT, and concatenates the results across scales.

    Args:
        orig_vis_feats (torch.Tensor): Input tensor of shape (B, S, T, D)

    Returns:
        torch.Tensor: Concatenated tensor containing FFT features from all scales
    """
    B, S, T, D = input.shape
    half_T = T // 2

    # Step 1: Aggregate data across segments
    aggregated_tensor = torch.zeros((B, (S + 1) * half_T, D), 
                                    device=input.device, 
                                    dtype=input.dtype)

    # First segment
    aggregated_tensor[:, :half_T, :] = input[:, 0, :half_T, :]

    # Intermediate segments (averaging overlapping halves)
    for s in range(S - 1):
        aggregated_tensor[:, (s + 1) * half_T:(s + 2) * half_T, :] = (
            input[:, s, half_T:, :] + input[:, s + 1, :half_T, :]
        ) / 2

    # Last segment
    aggregated_tensor[:, S * half_T:, :] = input[:, -1, half_T:, :]

    # Change shape to (B, D, T) for interpolation
    aggregated_tensor = aggregated_tensor.permute(0, 2, 1)
    #print(1, aggregated_tensor.shape)

    # Step 2: Upscale the time dimension (T -> T * 2)
    T_new = aggregated_tensor.shape[-1]
    #x_upscaled = torch.nn.functional.interpolate(aggregated_tensor, size=(T_new,), mode='linear', align_corners=False)
    #print(2, x_upscaled.shape)
    # Permute back to original (B, T, D) shape
    x_upscaled = aggregated_tensor.permute(0, 2, 1)

    # Step 3: Process with decreasing sliding window scales
    scales = [1, 2, 4, 8]  # T, T/2, T/4, T/8
    fft_results = []

    for scale in scales:
        window_size = T_new // scale
        scale_results = []

        # Sliding window over the upscaled tensor
        for start in range(0, T_new, window_size // 2):
            end = min(start + window_size, T_new)
            window_slice = x_upscaled[:, start:end, :]

            # Apply FFT on each window slice
            fft_result = torch.fft.rfft(window_slice, dim=1)

            def rms_norm(tensor, dim=-1, epsilon=1e-8):
                rms = torch.sqrt(torch.mean(tensor ** 2, dim=dim, keepdim=True) + epsilon)
                return tensor / rms

            # Compute the real and imaginary RMS normalized tensors
            real_fft = torch.log(fft_result.real - torch.min(fft_result.real, dim=-1, keepdim=True)[0] + 1e-6)
            imag_fft = torch.log(fft_result.imag - torch.min(fft_result.imag, dim=-1, keepdim=True)[0] + 1e-6)

            # Apply RMS normalization
            real_fft_norm = rms_norm(real_fft)
            imag_fft_norm = rms_norm(imag_fft)
            fft_concat = torch.cat((real_fft_norm, imag_fft_norm), dim=-1)  # (B, window_size//2 + 1, 2*D)
            scale_results.append(fft_concat)

        # Concatenate results across windows for this scale
        scale_concat = torch.cat(scale_results, dim=1)
        fft_results.append(scale_concat)

    # Step 4: Concatenate results from all scales
    final_output = torch.cat(fft_results, dim=1)

    return final_output  # Final shape: (B, sum_of_all_scales, 2*D)

def contrastive_separation_loss(stream_embd, stream_embd_u, temperature=0.07, margin=0.1):
    """
    Computes a loss that pushes apart the embeddings in stream_embd and stream_embd_u,
    and even more so, the embeddings across the batch dimension (for each sequence position).
    
    Parameters:
      stream_embd: Tensor of shape (B, S, D)
      stream_embd_u: Tensor of shape (B, S, D)
      temperature: scaling factor for the cross-stream loss (InfoNCE style)
      margin: margin used to enforce extra separation among batch items
      
    Returns:
      A scalar loss value.
    """
    B, S, D = stream_embd.shape
    
    # (1) Cross–stream separation
    # Flatten out the B and S dimensions so that each (b,s) becomes a separate example.
    A = stream_embd.view(B * S, D)
    B_u = stream_embd_u.view(B * S, D)
    
    # Normalize the embeddings so that cosine similarity is used.
    A = F.normalize(A, dim=-1)
    B_u = F.normalize(B_u, dim=-1)
    
    # Compute the cosine similarity matrix between A and B_u.
    # (Diagonal entries correspond to the "matching" pair from the same (b,s))
    sim_matrix = torch.matmul(A, B_u.t())
    
    # Divide by temperature (as in InfoNCE).
    logits = sim_matrix / temperature
    
    # Since we want to “separate” (i.e. push apart) even the matching pairs,
    # we will treat the diagonal as the “positive” class and use cross entropy loss
    # so that even these are forced to be lower than all the other (negative) entries.
    targets = torch.arange(B * S, device=A.device)
    loss_cross = F.cross_entropy(logits, targets)
    
    # (2) Within–batch separation
    # For each sequence index, we want embeddings coming from different batch items
    # to be even more dissimilar. We do this separately for stream_embd and stream_embd_u.
    loss_within = 0.0
    for s in range(S):
        # For stream_embd at sequence position s:
        a_s = stream_embd[:, s, :]  # shape (B, D)
        a_s = F.normalize(a_s, dim=-1)
        # Compute pairwise cosine similarities; we want these to be low.
        sim = torch.matmul(a_s, a_s.t())  # shape (B, B)
        # Create a mask to ignore the diagonal (self–similarity)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        # We use a ReLU so that any similarity above -margin is penalized.
        # (Since cosine similarity is in [-1,1], making the loss be ReLU(similarity + margin)
        # will push the similarity below -margin.)
        loss_within += F.relu(sim[mask] + margin).mean()
        
        # Do the same for stream_embd_u:
        a_u_s = stream_embd_u[:, s, :]
        a_u_s = F.normalize(a_u_s, dim=-1)
        sim_u = torch.matmul(a_u_s, a_u_s.t())
        loss_within += F.relu(sim_u[mask] + margin).mean()
    
    # There are 2*S terms in the above loop.
    loss_within /= (2 * S)
    
    # Total loss: you can weight these two parts differently if desired.
    total_loss = loss_cross + loss_within
    return total_loss