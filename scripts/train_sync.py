import copy
import math
import os
import time
import shutil
import logging
from pathlib import Path

import tqdm

from omegaconf import OmegaConf
import torch
import re
import torch.distributed as dist

from utils.logger import LoggerWithTBoard
from scripts.train_utils import (EarlyStopper, AverageMeter, adjust_state_dict,
                                 broadcast_obj, get_batch_sizes, get_datasets,
                                 get_device, get_loaders, get_lr_scheduler,
                                 get_model, get_optimizer, get_transforms,
                                 init_ddp, is_master, log_window_metrics, make_optim_step,
                                 make_backward_and_optim_step, prepare_inputs,
                                 set_seed, toggle_mode, verbose_epoch_progress, apply_fn_recursive)
from utils.utils import show_cfg_diffs


def validate(cfg, model, loaders, datasets, model_without_ddp, optimizer, lr_scheduler, epoch, device, global_rank, logger, target_key, world_size, early_stoppers):
    torch.cuda.empty_cache()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.training.use_half_precision):
        phase = 'valid'
        toggle_mode(cfg, model, phase)
        running_results = dict()
        losses_m = {'loss_total': AverageMeter()}
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        iter_time_m = AverageMeter()
        end = time.time()
        data_time_m.reset()
        batch_time_m.reset()
        iter_time_m.reset()
        total_samples = len(loaders[phase].dataset)
        log_max_items = cfg.logging.get('log_max_items', total_samples)
        #ids_to_cache = set(torch.arange(0, total_samples, total_samples/log_max_items).long().tolist())
        num_samples = 0

        for i, batch in enumerate(loaders[phase]):
            iter_step = i
            aud, vid, orig_audio, targets, others = prepare_inputs(batch, device, phase)
            num_samples += len(vid) * cfg.training.world_size
            data_time_m.update(time.time() - end)

            with torch.set_grad_enabled(False):
                losses_dict, logits = model(vid, aud, targets, others=others, aud_unshifted=orig_audio if cfg.training.get("predict_aud_repr_shift", False) else None, loss_fns=cfg.training.loss_fns)
                loss = sum(losses_dict.values())

            batch_time_m.update(time.time() - end)

            if not torch.isfinite(loss).all():
                logging.error(f'Loss is not finite: {loss}')
                raise RuntimeError(f'Worker (#{global_rank}): Loss is not finite: {loss}')

            iter_results = dict(
                logits=[apply_fn_recursive(logits, lambda x: x.detach().cpu())],
                targets=[apply_fn_recursive(targets[target_key], lambda x: x.cpu())],
                path_suffix=[str(Path(p).relative_to(datasets[phase].vids_dir)) for p in batch['path']],
            )
            losses_m['loss_total'].update(loss.item(), len(vid))
            
            for loss_key in losses_dict:
                if loss_key not in losses_m:
                    losses_m[loss_key] = AverageMeter()
                losses_m[loss_key].update(losses_dict[loss_key].item(), len(vid))

            if is_master(global_rank) and i % cfg.logging.log_frequency == 0 and i > 0:
                iter_loss = losses_m['loss_total'].val
                batch_size = len(vid)
                samples_per_epoch = len(loaders[phase].dataset)
                percent_done = 100.0 * (i+1) / len(loaders[phase])
                eta = iter_time_m.avg * len(loaders[phase]) * (1 - (i+1) / len(loaders[phase]))
                sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
                samples_per_s = batch_size * world_size / batch_time_m.avg
                samples_per_s_per_gpu = batch_size / batch_time_m.avg
                msg = f'{phase} ({epoch})'
                msg += f' [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_done:.0f}%, ETA {eta/60:.1f}m)]'
                msg += f' Data (t): {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                msg += f' Batch (t): {batch_time_m.val:.3f} ({batch_time_m.avg:.3f},'
                msg += f' {samples_per_s:#g}/s, {samples_per_s_per_gpu:#g}/s/gpu)'
                msg += f' Log (t): {iter_time_m.val:.3f} ({iter_time_m.avg:.3f})'
                msg += f' Loss: {iter_loss:.3E}'

                for key in losses_m:
                    if key == "loss_total":
                        continue
                    msg += f' {key}: {losses_m[key].val:.3E}'

                logging.info(msg)

            #if i in ids_to_cache:
            for res_k, res_v in iter_results.items():
                default_obj = [] if isinstance(res_v, list) else 0
                running_results[res_k] = running_results.get(res_k, default_obj) + iter_results[res_k]

            iter_time_m.update(time.time() - end)
            end = time.time()

        if is_master(global_rank):
            logging.info(f'({phase}) Done')
        
        for loss_name, loss_meter in losses_m.items():
            running_results[loss_name] = loss_meter.avg

        if dist.is_initialized():
            dist.barrier()
        epoch_loss, metrics = verbose_epoch_progress(global_rank, logger, running_results, phase, epoch,
                                                        datasets[phase], False)
    
        if phase == cfg.training.early_stop_phase:
            has_model_improved = [False for early_stopper in early_stoppers]
            for e, early_stopper in enumerate(early_stoppers):
                has_model_improved[e] = early_stopper.is_new_model_better_than_curr(metrics)
                if has_model_improved[e]:
                    early_stopper.reset_patience(global_rank, metrics)
                else:
                    early_stopper.increment_patience(global_rank)

        if is_master(global_rank):
            
            # Early stopping update
            for e, early_stopper in enumerate(early_stoppers):
                if phase == cfg.training.early_stop_phase and has_model_improved[e]:
                    logger.log_best_model(model_without_ddp, epoch_loss, iter_step, epoch, optimizer,
                                            lr_scheduler, metrics, cfg, early_stopper.metric_name)

        if dist.is_initialized():
            dist.barrier()

        if is_master(global_rank):
            logging.info('Finished Validation')
    del logits, losses_dict, batch, iter_results
    torch.cuda.empty_cache()
    import gc; gc.collect()
    if all([early_stopper.triggered for early_stopper in early_stoppers]):
        if is_master(global_rank):
            logging.info(f'Training is early stopped @ {epoch}; RANK: {global_rank}')
        return True
    return False




def train(cfg):
    init_ddp(cfg)
    global_rank = cfg.training.global_rank
    world_size = cfg.training.world_size
    target_key = 'offset_target'
    logger = LoggerWithTBoard(global_rank, cfg)

    if is_master(global_rank):
        logging.info(f'Config: \n{OmegaConf.to_yaml(cfg)}')

    torch.backends.cudnn.benchmark = True
    device, num_gpus = get_device(cfg)
    cfg.ckpt_path = broadcast_obj(cfg.ckpt_path, global_rank, device)
    assert hasattr(cfg, 'ckpt_path'), f'No ckpt_path in the config: {cfg} for worker {global_rank}'
    set_seed(cfg.training.seed)  # same seed for all workers for model init
    model, model_without_ddp = get_model(cfg, device)

    if (cfg.training.lora.enabled and not cfg.training.lora.load_first):
        model.add_lora_adapters(cfg.training)

    optimizer = get_optimizer(cfg, model, num_gpus)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)
    set_seed(cfg.training.seed + global_rank)
    batch_sizes = get_batch_sizes(cfg, num_gpus)
    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms)
    loaders = get_loaders(cfg, datasets, batch_sizes)
    early_stoppers = [
        EarlyStopper(cfg.training.patience, cfg.training.to_max_metric[i], cfg.training.metric_name[i]) for i in range(len(cfg.training.metric_name))
    ]

    if cfg.training.run_test_only or cfg.training.resume or cfg.training.finetune:
        ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        start_epoch = ckpt['epoch'] + 1
        #if (cfg.training.resume and "step" in ckpt):
        #    skip_iters = ckpt['step']
        #else:
        skip_iters = 0
        ckpt_cfg = ckpt['args']
        ckpt_metrics = ckpt['metrics']
        if (hasattr(cfg.training, "del_modules") and cfg.training.del_modules is not None):
            del_keys = {}
            for key in ckpt["model"]:
                for p, pattern in enumerate(cfg.training.del_modules):
                    if re.search(f"^{pattern}.*", key):
                        del_keys[key] = None
            for key in del_keys:
                del ckpt["model"][key]
        if (hasattr(cfg.training, "module_map") and cfg.training.module_map.original_keys is not None):
            map_keys = {}
            for key in ckpt["model"]:
                for p, pattern in enumerate(cfg.training.module_map.original_keys):
                    if re.search(f"^{pattern}.*", key):
                        new_pattern = key.replace(pattern, cfg.training.module_map.map_keys[p])
                        new_pattern = new_pattern.split(".")
                        if (new_pattern[0] == ""):
                            new_pattern = ".".join(new_pattern[1:])
                        map_keys[key] = new_pattern
                        print(key, "->", new_pattern)
            for key, value in map_keys.items():
                ckpt["model"][value] = ckpt["model"][key]
        if hasattr(cfg.training, "dilate_modules") and cfg.training.dilate_modules.patterns is not None:
            # Assume:
            #    cfg.training.dilate_modules.patterns: a list of top-level regex strings for the target modules.
            #    cfg.training.dilate_modules.factors: a list of dilation factors corresponding to each regex.
            dilate_keys = {}
            for key in list(ckpt["model"].keys()):
                for p, pattern in enumerate(cfg.training.dilate_modules.patterns):
                    # The regex matches the pattern at the beginning of the key followed by a number.
                    m = re.search(f"^({pattern})\\.(\\d+).*", key)
                    if m:
                        base = m.group(1)
                        old_index = int(m.group(2))
                        dilation_factor = cfg.training.dilate_modules.factors[p]
                        # For example, with dilation_factor 2, layer 0 -> 1, layer 1 -> 3, layer 2 -> 5, etc.
                        new_index = old_index * dilation_factor + 1
                        new_key = key.replace(f"{base}.{old_index}", f"{base}.{new_index}", 1)
                        dilate_keys[key] = new_key
            for old_key, new_key in dilate_keys.items():
                ckpt["model"][new_key] = ckpt["model"].pop(old_key)
                print(old_key, "->", new_key)
            

        if cfg.training.finetune:
            try:
                map_missing_weights = cfg.model.get("map_missing_weights", dict())
                for n in ckpt["model"]:
                    if (n.split(".")[0] in map_missing_weights):
                        np = ".".join([map_missing_weights[n.split(".")[0]]] + n.split(".")[1:])
                        if (np not in ckpt["model"][np]):
                            print(f'{n} -> {np}: {ckpt["model"][n].shape}')
                            ckpt["model"][np] = copy.deepcopy(ckpt["model"][n])
                _ckpt_load_status = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
            except:
                ckpt["model"] = adjust_state_dict(model_without_ddp.state_dict(), ckpt["model"])
                _ckpt_load_status = model_without_ddp.load_state_dict(ckpt['model'], strict=False)

            if len(_ckpt_load_status.missing_keys) > 0 or len(_ckpt_load_status.unexpected_keys) > 0:
                logging.warning(f'ckpt load status: {_ckpt_load_status}')
                logging.warning('Check if the above missing keys are expected. If so, ignore this warning.'
                                'Otherwise (e.g. lots of unmatched keys), fix the ckpt or the model.'
                                'Expectable keys could be heads during fine-tuning')
                
            logging.info(f'Finetuning from: {ckpt_cfg.ckpt_path} on {datasets["train"].__class__.__name__}')
            start_epoch = 0
            show_cfg_diffs(ckpt_cfg, cfg, Path(cfg.ckpt_path).parent / 'cfg_diffs.diff')
        else:
            model_without_ddp.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'lr_scheduler' in ckpt:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        if (not cfg.training.reset_metric):
            if (isinstance(ckpt_metrics, list)):
                for e, early_stopper in enumerate(early_stoppers):
                    early_stopper.set_best_metrics(ckpt_metrics[e])

    else:
        start_epoch = 0
    start_epoch += 1

    if (cfg.training.lora.enabled and cfg.training.lora.load_first):
        model.add_lora_adapters(cfg.training)            
        optimizer = get_optimizer(cfg, model, num_gpus)
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    else:
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

    logger.log_param_num(global_rank, model)
    num_epochs = 0 if cfg.training.run_test_only else cfg.training.num_epochs

    if cfg.training.compile:
        model = torch.compile(model)

    startup_validate = cfg.training.get('startup_validate', False)
    phase = 'train'

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        toggle_mode(cfg, model, phase)
        running_results = dict()
        losses_m = {'loss_total': AverageMeter()}
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        iter_time_m = AverageMeter()

        if dist.is_initialized():
            loaders[phase].sampler.set_epoch(epoch)

        data_time_m.reset()
        batch_time_m.reset()
        iter_time_m.reset()
        total_samples = len(loaders[phase].dataset)
        log_max_items = cfg.logging.get('log_max_items', total_samples)
        ids_to_cache = set(torch.arange(0, total_samples, total_samples/log_max_items).long().tolist())
        num_samples = 0
        num_acum_iters = 0
        

        if (cfg.training.resume and skip_iters > 0 and epoch * len(loaders[phase]) < skip_iters):
            loaders[phase].dataset.enable_fast_skip()
        for i, batch in enumerate(loaders[phase]): 
            iter_step = epoch * len(loaders[phase]) + i
            if (cfg.training.resume):
                if (iter_step < skip_iters):
                    num_samples += cfg.training.base_batch_size * cfg.training.world_size
                    continue
                loaders[phase].dataset.disable_fast_skip()
                if (not isinstance(batch, dict)):
                    continue
            num_acum_iters += 1
                
            if (startup_validate):
                validate(cfg, model, loaders, datasets, model_without_ddp, optimizer, lr_scheduler, epoch, device, global_rank, logger, target_key, world_size, early_stoppers)
                torch.cuda.empty_cache()
                toggle_mode(cfg, model, phase)
                startup_validate = False
            if cfg.training.validate_only:
                return

            end = time.time()
            aud, vid, orig_audio, targets, others = prepare_inputs(batch, device, phase, get_orig_audio=cfg.training.get("predict_aud_repr_shift", False))

            num_samples += len(vid) * cfg.training.world_size
            data_time_m.update(time.time() - end)

            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                with torch.set_grad_enabled(True):
                    losses_dict, logits = model(vid, aud, targets, others=others, aud_unshifted=orig_audio if cfg.training.get("predict_aud_repr_shift", False) else None, loss_fns=cfg.training.loss_fns)
                    loss = sum(losses_dict.values())

            acum_steps = cfg.training.get("acum_steps", 1)
        
            if (acum_steps == 1):
                make_backward_and_optim_step(cfg, loss, model, optimizer, lr_scheduler)
                model.zero_grad(set_to_none=True)
                num_acum_iters = 0
            else:
                loss.backward()
                if (num_acum_iters % acum_steps == 0):
                    make_optim_step(cfg, model, optimizer, lr_scheduler)
                    model.zero_grad(set_to_none=True)

            batch_time_m.update(time.time() - end)

            if not torch.isfinite(loss).all():
                logging.error(f'Loss is not finite: {loss}')
                raise RuntimeError(f'Worker (#{global_rank}): Loss is not finite: {loss}')

            iter_results = dict(
                logits=[apply_fn_recursive(logits, lambda x: x.detach().cpu())],
                targets=[apply_fn_recursive(targets[target_key], lambda x: x.cpu())],
                path_suffix=[str(Path(p).relative_to(datasets[phase].vids_dir)) for p in batch['path']],
            )
            losses_m['loss_total'].update(loss.item(), len(vid))

            for loss_key in losses_dict:
                if loss_key not in losses_m:
                    losses_m[loss_key] = AverageMeter()
                losses_m[loss_key].update(losses_dict[loss_key].item(), len(vid))

            if is_master(global_rank) and i % cfg.logging.log_frequency == 0 and i > 0:
                iter_loss = losses_m['loss_total'].val
                logger.log_iter_loss(iter_loss, iter_step, phase, prefix='total')
                logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], iter_step)
                batch_size = len(vid)
                samples_per_epoch = len(loaders[phase].dataset)
                percent_done = 100.0 * (i+1) / len(loaders[phase])
                eta = iter_time_m.avg * len(loaders[phase]) * (1 - (i+1) / len(loaders[phase]))
                sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
                samples_per_s = batch_size * world_size / batch_time_m.avg
                samples_per_s_per_gpu = batch_size / batch_time_m.avg
                msg = f'{phase} ({epoch})'
                msg += f' [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_done:.0f}%, ETA {eta/60:.1f}m)]'
                msg += f' Data (t): {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                msg += f' Batch (t): {batch_time_m.val:.3f} ({batch_time_m.avg:.3f},'
                msg += f' {samples_per_s:#g}/s, {samples_per_s_per_gpu:#g}/s/gpu)'
                msg += f' Log (t): ({iter_time_m.avg:.3f})'
                msg += f' Loss: {iter_loss:.3E}'
                for key in losses_m:
                    if key == "loss_total":
                        continue
                    msg += f' {key}: {losses_m[key].val:.3E}'
                    logger.log_iter_loss(losses_m[key].val, iter_step, phase, prefix=f'{key}')
                logging.info(msg)

            if i in ids_to_cache:
                for res_k, res_v in iter_results.items():
                    default_obj = [] if isinstance(res_v, list) else 0
                    running_results[res_k] = running_results.get(res_k, default_obj) + iter_results[res_k]

            iter_time_m.update(time.time() - end)
            end = time.time()
            if iter_step % cfg.logging.val_frequency == 0 and iter_step > 0:
                early_stop = validate(cfg, model, loaders, datasets, model_without_ddp, optimizer, lr_scheduler, epoch, device, global_rank, logger, target_key, world_size, early_stoppers)
                torch.cuda.empty_cache()
                toggle_mode(cfg, model, phase)
                if (early_stop):
                    break
            if is_master(global_rank):
                if iter_step % (cfg.logging.log_frequency * 2) == 0 and iter_step > 0:
                    logger.log_latest_model(model_without_ddp, 0, iter_step,
                        epoch, optimizer, lr_scheduler, [early_stopper.best_metrics for early_stopper in early_stoppers], cfg)
            if dist.is_initialized():
                dist.barrier()

        if is_master(global_rank):
            logging.info(f'({phase}) Done')

        #for loss_name, loss_meter in losses_m.items():
        #    running_results[loss_name] = loss_meter.avg

        if dist.is_initialized():
            dist.barrier()

        #if all([early_stopper.triggered for early_stopper in early_stoppers]):
        #    if is_master(global_rank):
        #        logging.info(f'Training is early stopped @ {epoch}; RANK: {global_rank}')
        #    break

    if is_master(global_rank):
        logging.info('Finished Training')
    if dist.is_initialized():
        dist.barrier()
    if is_master(global_rank):
        logger.finish_wandb_logging()
