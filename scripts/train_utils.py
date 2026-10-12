import itertools
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import re

import scipy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
try:
    import wandb
except ImportError:
    wandb = None

from matplotlib import pyplot as plt
from sklearn.metrics import (top_k_accuracy_score, average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler

from utils.utils import (fix_prefix, get_obj_from_str, get_transform_instance_from_compose,
                         disable_print_if_not_master, instantiate_from_config, show_cfg_diffs)

from collections import defaultdict
import typing as tp
from torch import autograd

import torch
import torch.autograd as autograd
from collections import defaultdict
from typing import Dict, Any, List

def plot_focal_maps(tensor_dict, spectrogram, base_path, b=None):
    """
    Plots time series or heatmaps for each tensor in the dictionary with optional background spectrograms.
    Tensors of shape (B, L) are overlaid on spectrograms, while (B, S, S) tensors are plotted as standalone heatmaps.
    
    Args:
        tensor_dict (dict): Dictionary where keys are labels and values are tensors.
                            Expected shapes are (B, L) for time series and (B, S, S) for heatmaps.
        spectrogram (torch.Tensor): Spectrogram tensor of shape (B, W, H).
        base_path (str): Base path to save the plot, excluding file extension.
        b (int, optional): Maximum number of batch items to plot. If None, plots all batch items.

    Returns:
        None
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(base_path)
    os.makedirs(dir_name, exist_ok=True)

    # Process the spectrogram
    if spectrogram is not None:
        spectrogram = spectrogram.detach().clone().cpu()

    # Determine the maximum number of items to plot from each tensor and spectrogram
    if b is None:
        tensor_sizes = [tensor.size(0) for tensor in tensor_dict.values()]
        if spectrogram is not None:
            tensor_sizes.append(spectrogram.size(0))
        b = min(tensor_sizes)

    # Process each batch item up to `b`
    for i in range(b):
        # Count the number of subplots needed
        num_subplots = len(tensor_dict)
        fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots), dpi=60)

        # Ensure axs is iterable
        if num_subplots == 1:
            axs = [axs]

        # Plot the spectrogram or heatmaps in each subplot
        for ax, (label, tensor) in zip(axs, tensor_dict.items()):
            tensor = tensor.detach().clone().cpu()  # Detach and clone

            if tensor.dim() == 2:  # (B, L) tensors
                # No normalization of tensor
                # Calculate the min and max values with a 5% extension for the spectrogram's y-axis
                y_min, y_max = tensor[i].min(), tensor[i].max()
                y_range = y_max - y_min if y_max != y_min else 1
                y_min -= y_range * 0.05  # Extend 5% below min
                y_max += y_range * 0.05  # Extend 5% above max

                # Normalize the spectrogram based on spectrogram[i]
                if spectrogram is not None and i < spectrogram.size(0):
                    spec_i = spectrogram[i]
                    # Normalize spectrogram per sample
                    spec_i = (spec_i - spec_i.min()) / (spec_i.max() - spec_i.min() + 1e-8)
                    ax.imshow(spec_i.numpy(), aspect='auto', cmap='gray',
                              extent=[0, len(tensor[i]), y_min, y_max])

                # Overlay the 'step' plot
                if i < tensor.size(0):
                    ax.step(range(len(tensor[i]) + 1), 
                            np.concatenate((tensor[i], tensor[i, -1, np.newaxis]), axis=0),
                            where='post', color='red')
                    ax.set_title(label)
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Values")

            elif tensor.dim() == 3:  # (B, S, S) tensors
                # No normalization of tensor
                # Plot the heatmap without spectrogram
                if i < tensor.size(0):
                    heatmap = tensor[i].numpy()
                    im = ax.imshow(heatmap, aspect='auto', cmap='coolwarm')
                    ax.set_title(label)
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Frequency")
                    # Add colorbar next to the heatmap
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")

            else:
                raise ValueError(f"Tensor for {label} does not have the correct shape (B, L) or (B, S, S).")

        # Layout adjustment and save the figure
        plt.tight_layout()
        final_path = f"{base_path}_{i}.png"
        plt.savefig(final_path)
        plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_ddp(cfg):
    local_rank = os.environ.get('LOCAL_RANK')

    if local_rank is not None:
        print('NUM_GPUS', os.environ.get("WORLD_SIZE"),
              '- LOCAL RANK', os.environ.get('LOCAL_RANK'),
              '- GLOBAL RANK', os.environ.get('RANK'))
        print('HOST:', os.environ.get('HOSTNAME'),
              'MASTER:', os.environ.get('MASTER_ADDR'), ':', os.environ.get('MASTER_PORT'))
        dist.init_process_group(cfg.training.dist_backend, timeout=timedelta(0, 600))
        cfg.training.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.training.global_rank = int(os.environ['RANK'])
        cfg.training.world_size = dist.get_world_size()
        disable_print_if_not_master(is_master(dist.get_rank()))
    else:
        cfg.training.local_rank = 0
        cfg.training.global_rank = 0
        cfg.training.world_size = 1


def is_master(global_rank):
    return global_rank == 0


def get_curr_time_w_random_shift():
    # shifting for a random number of seconds so that exp folder names coincide less often
    now = datetime.now() - timedelta(seconds=np.random.randint(60))
    return now.strftime('%y-%m-%dT%H-%M-%S')


def broadcast_obj(object, global_rank, device):
    if dist.is_initialized():
        objects = [object if is_master(global_rank) else None for _ in range(dist.get_world_size())]
        dist.broadcast_object_list(objects, src=0, device=device)
        object = objects[0]
    return object


def get_device(cfg):
    device = torch.device(cfg.training.local_rank)
    torch.cuda.set_device(device)
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    return device, num_gpus


def get_transforms(cfg, which_transforms=['train', 'test']):
    transforms = {}
    for mode in which_transforms:
        ts_cfg = cfg.get(f'transform_sequence_{mode}', None)
        ts = [lambda x: x] if ts_cfg is None else [instantiate_from_config(c) for c in ts_cfg]
        transforms[mode] = torchvision.transforms.Compose(ts)
    return transforms


def get_datasets(cfg, transforms, which_datasets=['train', 'valid', 'test']):
    if not isinstance(which_datasets, (list, tuple)):
        which_datasets = [which_datasets]
    DatasetClass = get_obj_from_str(cfg.data.dataset.target)
    load_fixed_offsets_on = cfg.data.dataset.params.load_fixed_offsets_on
    vis_load_backend = cfg.data.dataset.params.vis_load_backend
    splits_path = cfg.data.dataset.params.splits_path
    if 'size_ratios' in cfg.data.dataset.params:
        size_ratios = cfg.data.dataset.params.size_ratios
    else:
        size_ratios = {d: cfg.data.dataset.params.size_ratio for d in which_datasets}
    vids_path = cfg.data.vids_path
    attr_annot_path = cfg.data.dataset.params.get('attr_annot_path', None)
    max_attr_per_vid = cfg.data.dataset.params.get('max_attr_per_vid', None)

    datasets = dict()
    if 'train' in which_datasets:
        datasets['train'] = DatasetClass(
            split='train', vids_dir=vids_path, transforms=transforms['train'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['train'], splits_path=splits_path, sounds_only=cfg.data.dataset.params.get("sounds_only", False),
            data_cfg=cfg.data)
        logging.info(f'Loaded {len(datasets["train"])} train samples')
    if 'valid' in which_datasets:
        datasets['valid'] = DatasetClass(
            split='valid', vids_dir=vids_path, transforms=transforms['test'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['valid'], splits_path=splits_path, sounds_only=cfg.data.dataset.params.get("sounds_only", False), 
            data_cfg=cfg.data)
        logging.info(f'Loaded {len(datasets["valid"])} valid samples')
    if 'test' in which_datasets:
        datasets['test'] = DatasetClass(
            split='test', vids_dir=vids_path, transforms=transforms['test'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['test'], splits_path=splits_path, sounds_only=cfg.data.dataset.params.get("sounds_only", False), 
            data_cfg=cfg.data)
        logging.info(f'Loaded {len(datasets["test"])} test samples')
    return datasets


def get_batch_sizes(cfg, num_gpus=None):
    return {'train': cfg.training.base_batch_size, 'test': max(1, cfg.training.base_batch_size), }


def get_loaders(cfg, datasets, batch_sizes):
    loaders = dict()
    for phase, dataset in datasets.items():
        if dist.is_initialized():
            sampler = DistributedSampler(datasets[phase], shuffle=True)
        else:
            sampler = None

        if phase == 'train':
            # NOTE: don't change this to True, as it is used in `sampler`
            loaders[phase] = DataLoader(dataset, batch_sizes['train'], shuffle=sampler is None,
                                        sampler=sampler, num_workers=cfg.training.num_workers, pin_memory=True)
        else:
            loaders[phase] = DataLoader(dataset, batch_sizes['test'], shuffle=False,
                                        sampler=sampler, num_workers=cfg.training.num_workers, pin_memory=True)
    return loaders


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    '''If the `model` object is wrapped in `torch.nn.parallel.DistributedDataParallel` we have
    to use `model.modules` to get access to methods of the model. This wrapper allows
    to avoid using `if ddp: model.module.* else: model.*`. Used during `evaluate_on_sync_w_shifts`.'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def get_model(cfg, device):
    model = instantiate_from_config(cfg.model)
    for params in model.parameters():
            params.requires_grad = True
    # TODO: maybe in the module
    if (hasattr(model, "vfeat_extractor") and hasattr(model, "afeat_extractor")):
        if cfg.model.params.vfeat_extractor.is_trainable is False:
            for params in model.vfeat_extractor.parameters():
                params.requires_grad = False
        if cfg.model.params.afeat_extractor.is_trainable is False:
            for params in model.afeat_extractor.parameters():
                params.requires_grad = False
    if hasattr(model, "copy_modules"):
        for target, source in model.copy_modules.items():
            if (isinstance(getattr(model, target), torch.nn.Parameter)):
                getattr(model, target).requires_grad = False
            else:
                for params in getattr(model, target).parameters():
                    params.requires_grad = False
    

    model = model.to(device)
    if cfg.training.use_half_precision:
        for name, param in model.named_parameters():
            if ("raft" not in name):
                param = param.to(torch.bfloat16)
    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[cfg.training.local_rank], find_unused_parameters=True)
        # any mistaken calls on `model_without_ddp` (=None) will likely raise an error
        model_without_ddp = model.module
    frozen_submodules = cfg.training.get("stage_1_frozen_submodules", None)
    if frozen_submodules is not None:
        for name, param in model_without_ddp.named_parameters():
            for pattern in frozen_submodules:  # Assuming frozen_submodules contains regex patterns
                if re.search(pattern, name):  # Use regex search to match submodule name
                    param.requires_grad = False
                    #print("Disabled:", name)
                    break
    unfrozen_submodules = cfg.training.get("stage_2_unfrozen_submodules", None)
    if unfrozen_submodules is not None:
        for name, param in model_without_ddp.named_parameters():
            for pattern in unfrozen_submodules:  # Assuming frozen_submodules contains regex patterns
                if re.search(pattern, name):  # Use regex search to match submodule name
                    param.requires_grad = True
                    #print("Enabled:", name)
                    break
    frozen_submodules = cfg.training.get("stage_3_frozen_submodules", None)
    if frozen_submodules is not None:
        for name, param in model_without_ddp.named_parameters():
            for pattern in frozen_submodules:  # Assuming frozen_submodules contains regex patterns
                if re.search(pattern, name):  # Use regex search to match submodule name
                    param.requires_grad = False
                    #print("Disabled:", name)
                    break
    unfrozen_submodules = cfg.training.get("stage_4_unfrozen_submodules", None)
    if unfrozen_submodules is not None:
        for name, param in model_without_ddp.named_parameters():
            for pattern in unfrozen_submodules:  # Assuming frozen_submodules contains regex patterns
                if re.search(pattern, name):  # Use regex search to match submodule name
                    param.requires_grad = True
                    #print("Enabled:", name)
                    break
    for name, param in model_without_ddp.named_parameters():
        if (param.requires_grad):
            print("Enabled:", name)
    return model, model_without_ddp


def get_optimizer(cfg, model, num_gpus):
    learning_rate = cfg.training.base_learning_rate * num_gpus
    # TODO: instantiate (but we need to pass params as well - fix the intantiate fn)
    # TODO: consider uncommenting this line and optimize only these tensors (from torchvision reference)
    # params = [p for p in model.parameters() if p.requires_grad]
    # avoiding NaN during half precision training
    eps = 1e-7 if cfg.training.use_half_precision else 1e-8
    #if cfg.training.lora.enabled:
    #    params = [
    #            {
    #                "params": [p for n, p in model.named_parameters() if 'lora' in n],
    #                "weight_decay": cfg.training.optimizer.weight_decay
    #            },
    #            # Include other parameters if needed
    #        ]
    #else:
    params = model.parameters()
    if cfg.training.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(params, learning_rate, cfg.training.optimizer.betas,
                                     eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(params, learning_rate, cfg.training.optimizer.betas,
                                      eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(params, learning_rate, cfg.training.optimizer.momentum,
                                    weight_decay=cfg.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer: "{cfg.training.optimizer.name}" is not implemented')
    return optimizer


import torch.optim.lr_scheduler as lr_scheduler

class CosineAnnealingWarmRestartsWithDecay(lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, restart_decay_factor=0.9, last_epoch=-1, verbose=False):
        """
        CosineAnnealingWarmRestarts scheduler that decays the max learning rate after each restart.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations until the first restart.
            T_mult (int): Factor by which the period increases after each restart.
            eta_min (float): Minimum learning rate.
            restart_decay_factor (float): Multiplicative factor to reduce max LR at each restart.
            last_epoch (int): The index of last epoch.
            verbose (bool): If True, prints a message to stdout for each update.
        """
        super().__init__(optimizer, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
        self.restart_decay_factor = restart_decay_factor

    def step(self, epoch=None):
        # Capture the current T_cur before the update
        prev_T_cur = self.T_cur
        super().step(epoch)
        # Check for restart: T_cur resets to 0 (except at initialization)
        if self.last_epoch > 0 and self.T_cur == 0 and prev_T_cur > 0:
            # At restart, decay the base learning rates.
            new_base_lrs = []
            for base_lr in self.base_lrs:
                new_lr = base_lr * self.restart_decay_factor
                new_base_lrs.append(new_lr)
            self.base_lrs = new_base_lrs

def get_lr_scheduler(cfg, optimizer):
    sched_name = cfg.training.lr_scheduler.name

    if sched_name == 'constant_with_warmup':
        warmup = cfg.training.lr_scheduler.warmup
        lr_sched = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=warmup),
                lr_scheduler.ConstantLR(optimizer, factor=1),
            ],
            milestones=[warmup]
        )
    elif sched_name == 'constant':
        lr_sched = lr_scheduler.ConstantLR(optimizer, factor=1)
    elif sched_name == 'cosine_with_warmup':
        warmup = cfg.training.lr_scheduler.warmup
        T_max = cfg.training.lr_scheduler.T_max
        lr_sched = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=warmup),
                lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            ],
            milestones=[warmup]
        )
    elif sched_name == 'cosine_with_warm_restarts':
        warmup = cfg.training.lr_scheduler.warmup
        T_0 = cfg.training.lr_scheduler.T_0  # iterations until first restart
        T_mult = cfg.training.lr_scheduler.T_mult  # multiplicative factor for subsequent cycles
        restart_decay_factor = cfg.training.lr_scheduler.restart_decay_factor
        eta_min = cfg.training.lr_scheduler.eta_min if 'eta_min' in cfg.training.lr_scheduler else 0.0

        base_scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            restart_decay_factor=restart_decay_factor,
            verbose=False
        )

        if warmup > 0:
            lr_sched = lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    lr_scheduler.LinearLR(optimizer, start_factor=1/1e3, total_iters=warmup),
                    base_scheduler
                ],
                milestones=[warmup]
            )
        else:
            lr_sched = base_scheduler
    else:
        raise ValueError(f"Unknown lr_scheduler name: {sched_name}")
    
    return lr_sched



class EarlyStopper(object):

    def __init__(self, patience: int, to_max: bool, metric_name: str) -> None:
        '''E.g. If loss is the trackable metric, to_max=False; if accuracy, to_max=True'''
        self.no_change_epochs = 0
        self.to_max = to_max
        self.best_metric = 0.0 if to_max else float('inf')
        self.best_metrics = {metric_name: self.best_metric}
        self.metric_name = metric_name
        self.triggered = False
        self.patience = patience

    def reset_patience(self, global_rank, metrics):
        self.no_change_epochs = 0
        prior_best = self.best_metric
        self.set_best_metrics(metrics)
        if is_master(global_rank):
            logging.info(f'New best {self.metric_name}: {prior_best:.5f} -> {self.best_metric:.5f}')

    def increment_patience(self, global_rank):
        self.no_change_epochs += 1
        if self.no_change_epochs >= self.patience:
            self.triggered = True
        if is_master(global_rank):
            logging.info(
                f'{self.metric_name} ({self.best_metric:.5f}) hasnt changed for {self.no_change_epochs} '
                f'patience: {self.patience}'
            )

    def is_new_model_better_than_curr(self, metrics: dict):
        new_metric_val = metrics[self.metric_name]
        return self.best_metric < new_metric_val if self.to_max else self.best_metric > new_metric_val

    def set_best_metrics(self, metrics):
        self.best_metric = metrics[self.metric_name]
        self.best_metrics = metrics


def toggle_mode(cfg, model, phase):
    if phase == 'train':
        model.train()
        if (hasattr(model, "vfeat_extractor") and hasattr(model, "afeat_extractor")):
            if cfg.model.params.afeat_extractor.is_trainable is False:
                if dist.is_initialized():
                    model.module.afeat_extractor.eval()
                else:
                    model.afeat_extractor.eval()
            if cfg.model.params.vfeat_extractor.is_trainable is False:
                if dist.is_initialized():
                    model.module.vfeat_extractor.eval()
                else:
                    model.vfeat_extractor.eval()
    else:
        model.eval()


def apply_fn_recursive(obj, fn: callable):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, dict):
        return {k: apply_fn_recursive(v, fn) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [apply_fn_recursive(v, fn) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([apply_fn_recursive(v, fn) for v in obj])
    else:
        raise NotImplementedError(f'obj type: {type(obj)}')

def prepare_inputs(batch, device, phase=None, get_targets=True, get_orig_audio=False):
    targets = None
    if get_targets:
        targets = batch['targets']
        # targets = {k: targets[k].to(device, non_blocking=True) for k in targets if 'target' in k}
        for k, v in targets.items():
            if 'target' in k:
                targets[k] = apply_fn_recursive(v, lambda x: x.to(device))

    aud = batch['audio'].to(device)
    vid = batch['video'].to(device)
    if (get_orig_audio):
        orig_aud = batch['orig_audio'].to(device)
    else:
        #del batch['orig_audio']
        orig_aud = None
    others = {"jitter": batch['meta']['a_jitter_sec']} #{"full_video": batch['full_video'].to(device), "full_audio": batch['full_audio'].to(device)}

    return aud, vid, orig_aud, targets, others

def make_optim_step(cfg, model, optimizer, lr_scheduler):
    max_clip_norm = cfg.training.get('max_clip_norm', None)
    if max_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


def make_backward_and_optim_step(cfg, loss, model, optimizer, lr_scheduler):
    loss.backward()
    max_clip_norm = cfg.training.get('max_clip_norm', None)
    if max_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    optimizer.step()
    lr_scheduler.step()

def log_window_metrics(window_metrics: dict, iter_step: int, targets: dict, logger, jitter):
    """
    Logs the window metrics as a single multi-line plot per batch.
    
    Args:
        window_metrics (dict): A dict containing keys like 
            "window_cosine_similarity", "window_magnitude_similarity", and "window_mse".
            Each value is a tensor of shape [B, num_windows].
        iter_step (int): Current iteration step (used for naming directories/files).
        targets (dict): A dictionary that includes keys "offset_target" (a torch.long tensor)
            and "jitter" (a float tensor).
        logger: Logger object which contains ckpt_path attribute.
        jitter: Not used directly since jitter is included in targets.
    """
    if not hasattr(logger, "ckpt_path"):
        return
    # Create the save directory:
    save_dir = os.path.join(Path(logger.ckpt_path).parent, "window_metrics", str(iter_step))
    os.makedirs(save_dir, exist_ok=True)
    
    # Assuming all metrics have the same shape: [B, num_windows]
    sample_metric = next(iter(window_metrics.values()))
    B, num_windows = sample_metric.shape

    # Create symmetric x-axis biases.
    half = num_windows // 2
    if num_windows % 2 == 1:
        x_axis = np.arange(-half, half + 1)
    else:
        x_axis = np.linspace(-half, half, num_windows)
    
    # Determine the minimum and maximum x positions for scaling.
    x_min, x_max = x_axis[0], x_axis[-1]

    # For each batch index, create a plot.
    for b in range(B):
        plt.figure(figsize=(8, 6))
        for metric_name, metric_tensor in window_metrics.items():
            # Get the bth sample and convert to numpy
            y = metric_tensor[b].detach().cpu().numpy()
            # Normalize so that sum equals 1.0.
            s = y.sum()
            if s != 0:
                y = y / s
            # Plot the normalized metric.
            plt.plot(x_axis, y, label=metric_name)
        
        # Calculate the x position for the vertical line.
        # Mapping: label 0 -> x_min, label 20 -> 0, label 40 -> x_max.
        class_label = targets['offset_target'][b].item()
        x_line = x_min + (class_label / 40.0) * (x_max - x_min)
        plt.axvline(x=x_line, color='black', linestyle='--')
        if x_line != 0.0:
            plt.axvline(x=-x_line, color='red', linestyle='--')
        
        plt.xlabel("Bias")
        plt.ylabel("Normalized Metric Value")
        # Include jitter from the targets in the title.
        jitter_val = round(jitter[b].item(), 3)
        plt.title(f"Batch {b}, Iter {iter_step}, Class {class_label}, Jitter {jitter_val}")
        plt.legend()
        plt.tight_layout()
        
        # Save the figure to a file that includes the batch index, iter_step, and class label.
        filename = f"batch_{b}_iter_{iter_step}_class_{class_label}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

def verbose_epoch_progress(global_rank, logger, running_results: dict, phase, epoch, train_dataset,
                           save_csv_with_preds=False):
    running_results = gather_dict(running_results)
    # logging loss values
    if is_master(global_rank):
        for key, val in running_results.items():
            if "loss_" in key:
                logger.log_epoch_loss(running_results[key], epoch, phase, prefix=key.split("loss")[1])
        if 'loss_off' in running_results:
            logger.log_epoch_loss(running_results['loss_off'], epoch, phase, prefix='off')
        if 'loss_doubt' in running_results:
            logger.log_epoch_loss(running_results['loss_doubt'], epoch, phase, prefix='doubt')
    # logging metrics
    logits = torch.cat(running_results['logits']).float()
    targets = torch.cat(running_results['targets']).long()
    #regression = torch.cat(running_results['regression']).float()
    #regression_targets = torch.cat(running_results['regression_targets']).float()
    metrics = calc_cls_metrics(targets, logits, only_accuracy=True)
    #reg_metrics = calc_reg_metrics(regression_targets, regression)
    #metrics.update(reg_metrics)
    epoch_loss = running_results['loss_total']
    # a shortcut for binary tasks
    is_task_binary = logits.shape[-1] == 2  # during fine-tuning for syncability the task is binary
    if is_task_binary:
        if is_master(global_rank):
            logger.log_epoch_metrics(metrics, epoch, phase)
            # log to wandb
            if wandb.run is not None:
                wandb.log({'epoch': epoch})
    else:
        off_transf = get_transform_instance_from_compose(train_dataset.transforms, 'TemporalCropAndOffset')
        metrics = calc_cls_metrics(targets, logits)
        #reg_metrics = calc_reg_metrics(regression_targets, regression)
        #metrics.update(reg_metrics)
        # make a map from targets (int) to labels (human-readable class names)
        label_grid = [str(round(c, 3)) for c in off_transf.class_grid.tolist()]
        target2label = {target: label for target, label in enumerate(label_grid)}
        label2metrics = calc_performance_per_class(target2label, 'off', logits, targets,
                                                   train_dataset=train_dataset)
        preds = logits.argmax(dim=1)  # id of max logit
        # compute median accuracy for k=1 and k=k
        k = min(logits.shape[-1], 5)  # top-5 is not always available (e.g. for 3 classes)
        if not all(['accuracy_1' in v for v in label2metrics.values()]):
            logging.warning('Not all offset classes have predictions. Median accs of those are replaced with 0.0')
        metrics['accuracy_1_median'] = np.median([v.get('accuracy_1', 0.0) for v in label2metrics.values()])
        metrics[f'accuracy_{k}_median'] = np.median([v.get(f'accuracy_{k}', 0.0) for v in label2metrics.values()])
        if is_master(global_rank):
            #plots_and_log_perf_per_cls(logger, phase, epoch, train_dataset, label2metrics, targets, preds)
            logger.log_epoch_metrics(metrics, epoch, phase)
    return epoch_loss, metrics

# NOTE: wrap the call in `is_master` to avoid calling `plt.show()` on all processes
def plots_and_log_perf_per_cls(logger, phase, epoch, train_dataset, label2metrics, targets, preds,
                               metric='accuracy_1'):
    # adaptive hight
    fig, ax = plt.subplots(1, 3, figsize=(13, 7*len(label2metrics)/41))
    fig.suptitle(f'{logger.start_time} | {phase} (e{epoch}) | {train_dataset.__class__.__name__}')
    fontsize = None
    width = 0.60
    # NOTE: if metric is not in label2metrics, set it to 0.0
    num_cls = len(label2metrics)

    # all sorted by label
    label_list = list(label2metrics.keys())
    target_list = list(range(len(label2metrics)))
    pred_count_list = [preds.tolist().count(target) for target in target_list]
    target_count_list = [targets.tolist().count(target) for target in target_list]
    perf_list = [label2metrics[label].get(metric, 0.0) for label in label_list]
    if all([v == 0.0 for v in perf_list]):
        logging.warning(f'All {metric} values are 0.0')

    # distribution of predictions
    _dict = {'perf_per_cls': perf_list, 'preds': pred_count_list, 'targets': target_count_list}
    for i, (name, vals) in enumerate(_dict.items()):
        ax[i].set_title(name, fontsize=fontsize)
        ax[i].set_axisbelow(True)
        ax[i].xaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
        # mean line
        ax[i].vlines(sum(vals) / len(vals), 0, num_cls, color='black', linestyles='dashed', alpha=0.7)
        y = list(range(num_cls))
        # Add some text for labels, title and custom x-axis tick labels, etc.
        x_axis_name = metric if name == 'perf_per_cls' else 'Count'
        ax[i].set_xlabel(x_axis_name, fontsize=fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].set_yticks(target_list)
        ax[i].set_yticklabels(label_list, fontsize=fontsize)
        if name == 'perf_per_cls':
            ax[i].set_xlim(left=0.0, right=1.0)

        for rect in ax[i].barh(y, vals, width, alpha=0.5, color='C0', lw=1, align='center'):
            ax[i].annotate(f'{rect.get_width():.2f}',
                           xy=(rect.get_width(), rect.get_y()),
                           xytext=(10, 1),
                           textcoords="offset points",
                           ha='center',
                           va='center',
                           fontsize=fontsize,)

        ax[i].autoscale(enable=True, axis='y', tight=True)  # tight layout inside of the figure

    fig.tight_layout()
    # save to disk
    save_dir = Path(logger.logdir) / 'viz' / 'perf_per_class'
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f'{phase}_e{epoch:04d}.png')

    # log to wandb
    if wandb.run is not None:
        wandb.log({f"{phase}/perf_per_cls_{metric}": wandb.Image(fig), 'epoch': epoch})


def calc_performance_per_class(target2label, class_category, logits_off, targets_off, targets_cls=None,
                               train_dataset=None, add_doubt_cls=False):
    logits_and_targets = [logits_off, targets_off]
    if class_category == 'cls':
        assert targets_cls is not None, 'targets_cls is None'
        logits_and_targets += [targets_cls]

    target_cls2preds = {target_cls: {'logits_off': [], 'targets_off': []} for target_cls in target2label.keys()}

    for logits_targets_i in zip(*logits_and_targets):
        if class_category == 'off':
            logits_off, target_off = logits_targets_i
            target_off = target_off.item()
            key = target_off
        elif class_category == 'cls':
            raise NotImplementedError('performance per data class is not implemented (see running_results)')
            logits_off, target_off, target_cls = logits_targets_i
            target_cls = target_cls.item()
            target_off = target_off.item()
            key = target_cls
        target_cls2preds[key]['logits_off'] += [logits_off]
        target_cls2preds[key]['targets_off'] += [target_off]

    label2metrics = {}
    for key, preds in target_cls2preds.items():
        if len(preds['logits_off']) > 0 and len(preds['targets_off']) > 0:
            # if key target class was never predicted, calc_cls_metrics will throw some warnings
            metric_name2val = calc_cls_metrics(torch.tensor(preds['targets_off']),
                                               torch.stack(preds['logits_off']),
                                               only_accuracy=True, verbose=False, add_doubt_cls=add_doubt_cls)
        else:
            metric_name2val = dict()
        label2metrics[target2label[key]] = metric_name2val
    return label2metrics

def gather_dict(dct):
    if dist.is_initialized():
        dist.barrier()
        for k, v in dct.items():
            gather_buffer = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_buffer, v)
            if isinstance(v, list):
                # flattens a list of lists into one list
                dct[k] = list(itertools.chain(*gather_buffer))
            elif isinstance(v, float):
                # average
                dct[k] = sum(gather_buffer) / len(gather_buffer)
            else:
                raise NotImplementedError(f'{type(v)}, \n{v}')
    return dct


def calc_reg_metrics(regression_targets, outputs: torch.FloatTensor, prefix='', verbose=True):
    prefix = fix_prefix(prefix)
    metrics_dict = dict()
    
    # Check for finite values in outputs
    if verbose and not torch.isfinite(outputs).all():
        outputs = torch.rand_like(outputs)
        logging.warning('Infinity or NaN found in outputs. Replacing with random values.')
    
    # Ensure targets and outputs have the same shape
    assert regression_targets.shape == outputs.shape, "Targets and outputs must have the same shape"
    
    # Convert tensors to NumPy arrays
    regression_targets_np = regression_targets.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    
    # Compute regression metrics
    metrics_dict[f'{prefix}mse'] = mean_squared_error(regression_targets_np, outputs_np)
    metrics_dict[f'{prefix}rmse'] = np.sqrt(metrics_dict[f'{prefix}mse'])
    metrics_dict[f'{prefix}mae'] = mean_absolute_error(regression_targets_np, outputs_np)
    metrics_dict[f'{prefix}r2'] = r2_score(regression_targets_np, outputs_np)
    metrics_dict[f'{prefix}explained_variance'] = explained_variance_score(regression_targets_np, outputs_np)
    
    return metrics_dict

    
def calc_cls_metrics(targets, outputs: torch.FloatTensor, topk=(1, 5), only_accuracy=False, prefix='',
                     verbose=True, add_doubt_cls: bool = False, calc_tol_accuracy=True,
                     softmaxed_outputs=False, calc_pr_rec_f1=False):
    """
    Adapted from https://github.com/hche11/VGGSound/blob/master/utils.py

    Calculate statistics including mAP, AUC, and d-prime.
        Args:
            targets: 1d tensors, (dataset_size, )
            output: 2d tensors, (dataset_size, classes_num) - before softmax
            topk: tuple
            only_accuracy: bool - if True, only accuracy@k will be calculated
            prefix: str - prefix for the metric names
            verbose: bool - if True, will log warnings i.e if some classes never occured in targets or outputs
        Returns:
            metric_dict: a dict of metrics
    """
    prefix = fix_prefix(prefix)
    metrics_dict = dict()

    dataset_size, num_cls = outputs.shape
    topk = [min(k, num_cls) for k in topk]

    if softmaxed_outputs:
        targets_pred = outputs.clone()
    else:
        targets_pred = torch.softmax(outputs, dim=1)

    if verbose and not torch.isfinite(outputs).all():
        # could raise an error but keeping it a warning for potential debugging runs
        outputs = torch.rand_like(outputs)
        logging.warning('infinity or loss was nan. Replacing with random values.')

    # ids of the predicted classes (same as softmax)
    _, preds = torch.topk(outputs, k=max(topk), dim=1)
    unique_preds = sorted(list(set(preds[:, 0].tolist())))  # picking only the top prediction
    if verbose and (len(unique_preds) < num_cls):
        logging.warning(f'Some classes never occured in _outputs_. {prefix} pred classes: {unique_preds}')

    # accuracy@k
    for k in topk:
        if num_cls == 2:  # binary classification
            if k == 2:  # silence the warning
                continue
            metrics_dict[f'{prefix}accuracy_{k}'] = top_k_accuracy_score(targets, targets_pred[:, 1], k=k,
                                                                         labels=range(num_cls))
        else:
            metrics_dict[f'{prefix}accuracy_{k}'] = top_k_accuracy_score(targets, targets_pred, k=k,
                                                                         labels=range(num_cls))

    # accuracy@k_tol
    if calc_tol_accuracy:
        # NOTE: we ignore the performance on items that have targets that correspond to conf class
        if add_doubt_cls:
            num_off_cls = num_cls - 1  # we don't need conf cls for metrics with tolerance
            # mask out items with doubt cls target
            doubt_items_mask = targets != num_cls - 1
            targets_for_tol = targets.clone()[doubt_items_mask]
            preds_for_tol = preds.clone()[doubt_items_mask]
        else:
            num_off_cls = num_cls
            targets_for_tol = targets.clone()
            preds_for_tol = preds.clone()
        targets_as_preds = targets_for_tol.unsqueeze(-1).expand_as(preds_for_tol)
        targets_left_tol = (targets_as_preds - 1).clamp(0, num_off_cls-1)
        targets_right_tol = (targets_as_preds + 1).clamp(0, num_off_cls-1)
        targets_for_acc_w_tol = torch.stack([targets_left_tol, targets_as_preds, targets_right_tol])
        correct_for_maxtopk_w_tol = (preds_for_tol == targets_for_acc_w_tol).any(dim=0)
        for k in topk:
            # there might be more than one `True` per item (15, 16 in top2). Preventing overcounting w/ any()
            TPs_w_tol = correct_for_maxtopk_w_tol[:, :k].any(dim=1).sum().item()
            # adding 1e-7 to avoid division by 0 which occurs when all items have confidence class target
            metrics_dict[f'{prefix}accuracy_{k}_tol1'] = TPs_w_tol / (len(correct_for_maxtopk_w_tol) + 1e-7)

        if num_off_cls == 3 and dataset_size > 100:  # 100 is to avoid verbosity on iteration level
            logging.warning('Accuracy with tolerance is not reliable as num of offset classes is 3.')

    if only_accuracy:
        return metrics_dict

    # if there are no targets of some classes, the metrics will be wrong, replacing with dummy values
    unique_targets = sorted(list(set(targets.tolist())))
    if len(unique_targets) < num_cls:
        logging.warning(f'Some classes never occured in targets. {prefix} target classes: {unique_targets}')
        # some dummy values just for the sake of error prevention
        metrics_dict[f'{prefix}mAP'] = 0.0
        metrics_dict[f'{prefix}mROCAUC'] = 0.5
        metrics_dict[f'{prefix}dprime'] = 0.0
        return metrics_dict

    # avg precision, average roc_auc, and dprime
    targets_1hot = torch.nn.functional.one_hot(targets, num_classes=num_cls)

    targets_1hot = targets_1hot.numpy()
    targets_pred = targets_pred.numpy()

    # one-vs-rest
    avg_p = [average_precision_score(targets_1hot[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
    roc_aucs = [roc_auc_score(targets_1hot[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]

    metrics_dict[f'{prefix}mAP'] = np.mean(avg_p)
    metrics_dict[f'{prefix}mROCAUC'] = np.mean(roc_aucs)
    # Percent point function (ppf) (inverse of cdf — percentiles).
    metrics_dict[f'{prefix}dprime'] = scipy.stats.norm().ppf(metrics_dict[f'{prefix}mROCAUC'])*np.sqrt(2)

    if calc_pr_rec_f1:
        metrics_dict[f'{prefix}precision'] = precision_score(targets, preds[:, 0], zero_division=0.0)
        metrics_dict[f'{prefix}recall'] = recall_score(targets, preds[:, 0], zero_division=0.0)
        metrics_dict[f'{prefix}f1'] = f1_score(targets, preds[:, 0], zero_division=0.0)

    return metrics_dict


def adjust_state_dict(state_dict_1, state_dict_2):
    """
    Adjusts the parameters in state_dict_2 to match the shapes in state_dict_1.

    For sequential modules, if the target has more layers, the source layers are
    mapped into the middle layers of the target, as per the specified mapping.

    Args:
        state_dict_1 (dict): The target state dict (e.g., from the model).
        state_dict_2 (dict): The source state dict to be adjusted.

    Returns:
        dict: The adjusted state_dict_2.
    """

    adjusted_state_dict = {}
    pattern = re.compile(r'^(.*?)(\d+)(\..*)$')  # Pattern to match layer numbers in parameter names

    # Identify sequential layers and their parameters in both state dicts
    def get_sequential_layers(state_dict):
        sequential_layers = {}
        for name in state_dict.keys():
            match = pattern.match(name)
            if match:
                prefix, layer_num, suffix = match.groups()
                layer_num = int(layer_num)
                sequential_layers.setdefault(prefix, []).append((layer_num, name))
        return sequential_layers

    seq_layers_1 = get_sequential_layers(state_dict_1)
    seq_layers_2 = get_sequential_layers(state_dict_2)

    # Process each sequential module
    for prefix in seq_layers_1:
        layers_1 = sorted(seq_layers_1[prefix], key=lambda x: x[0])
        layers_2 = sorted(seq_layers_2.get(prefix, []), key=lambda x: x[0])

        num_layers_1 = len(layers_1)
        num_layers_2 = len(layers_2)
        num_extra_layers = num_layers_1 - num_layers_2

        # Determine the number of extra layers before and after
        if num_extra_layers > 0:
            extra_layers_before = (num_extra_layers + 1) // 2 if num_extra_layers % 2 else num_extra_layers // 2
            extra_layers_after = num_extra_layers - extra_layers_before
        else:
            extra_layers_before = extra_layers_after = 0

        # Map source layers to target layers
        layer_mapping = {}
        for idx, (layer_num_1, name_1) in enumerate(layers_1):
            if extra_layers_before <= idx < num_layers_1 - extra_layers_after:
                idx_source = idx - extra_layers_before
                if idx_source < num_layers_2:
                    layer_num_2, _ = layers_2[idx_source]
                    layer_mapping[layer_num_1] = layer_num_2
                else:
                    layer_mapping[layer_num_1] = None  # No corresponding source layer
            else:
                layer_mapping[layer_num_1] = None  # Extra layer (null)

        # Adjust parameters for this sequential module
        for layer_num_1, name_1 in layers_1:
            source_layer_num = layer_mapping[layer_num_1]
            p = state_dict_1[name_1]

            if source_layer_num is not None:
                # Build the source parameter name
                source_param_name = f"{prefix}{source_layer_num}{pattern.match(name_1).group(3)}"
                if source_param_name in state_dict_2:
                    param_from_state_dict_2 = state_dict_2[source_param_name]
                    adjusted_state_dict[name_1] = adjust_parameter(p, param_from_state_dict_2, name_1)
                    continue  # Move to the next parameter

            # Initialize new parameter if no corresponding source parameter
            adjusted_state_dict[name_1] = initialize_new_parameter(p, name_1)

    # Handle non-sequential parameters
    for name, p in state_dict_1.items():
        if name in adjusted_state_dict:
            continue  # Already processed
        if name in state_dict_2:
            param_from_state_dict_2 = state_dict_2[name]
            adjusted_state_dict[name] = adjust_parameter(p, param_from_state_dict_2, name)
        else:
            adjusted_state_dict[name] = initialize_new_parameter(p, name)

    return adjusted_state_dict

def adjust_parameter(target_param, source_param, name):
    """
    Adjusts a single parameter tensor to match the target shape, copying overlapping data.
    """
    if source_param.shape != target_param.shape:
        target_shape = target_param.shape
        new_param = torch.empty(target_shape, device=target_param.device, dtype=target_param.dtype)
        initialize_param(new_param, name)

        # Determine slices for overlapping data
        slices = tuple(slice(0, min(s_old, s_new)) for s_old, s_new in zip(source_param.shape, target_shape))

        # Adjust dimensions if necessary
        param = source_param
        while len(param.shape) < len(target_shape):
            param = param.unsqueeze(-1)
        while len(param.shape) > len(target_shape):
            param = param.squeeze(-1)

        new_param[slices] = param[slices]
        return new_param
    else:
        return source_param  # Shapes match, copy directly

def initialize_new_parameter(target_param, name):
    """
    Initializes a new parameter tensor with the target shape.
    """
    target_shape = target_param.shape
    new_param = torch.empty(target_shape, device=target_param.device, dtype=target_param.dtype)
    initialize_param(new_param, name)
    return new_param

def initialize_param(param, name):
    """
    Initializes a parameter tensor based on its name.
    """
    if 'weight' in name:
        if len(param.shape) < 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)
        else:
            torch.nn.init.kaiming_normal_(param)
    elif 'bias' in name:
        torch.nn.init.zeros_(param)
    else:
        torch.nn.init.normal_(param, mean=0.0, std=0.01)