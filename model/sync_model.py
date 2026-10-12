from collections import defaultdict
import csv
import json
import logging
import math
import random
import re
from typing import Any, Dict, List, Mapping, Tuple
import sys
import einops
import torch
from torch.nn import functional as F
from pathlib import Path
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PeftModel, LoraRuntimeConfig
from vector_quantize_pytorch import ResidualFSQ
sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config, instantiate_cfg_from_config
from model.modules.transformer import *
from model.model_utils import *

class BaseSynchformer(torch.nn.Module):
    ''' The module has similar structure to SparseSync (SparseSync) but has a diffrerent
    forward pass. It expects the output of the feature extractors to have global and
    segment-level representations.'''

    def __init__(self, afeat_extractor, vfeat_extractor, aproj, vproj, transformer=None, **kwargs):
        super().__init__()
        self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        self.afeat_extractor = instantiate_from_config(afeat_extractor)
        self.vproj = instantiate_from_config(vproj)
        self.aproj = instantiate_from_config(aproj)
        self.transformer = instantiate_from_config(transformer)
            

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, targets: Dict[str, torch.Tensor] = None, for_loop=False,
                vis_mask: torch.Tensor = None, aud_mask: torch.Tensor = None, others=None, aud_unshifted: torch.Tensor=None):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, S, Tv, C, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, 1, F, Ta)
            for_loop (bool): if True, will use a for loop inside of feat_extractors to iterate over the
                             segments or process them in parallel (False), treating segment dim as batch dim
                             (speed-memory tradeoff).
            vis_mask (torch.Tensor): mask for the visual tokens (as input)
            aud_mask (torch.Tensor): mask for the audio tokens (as input)
        Returns:
            tuple(Tensor, Tensor), Tensor: loss values, logits
        '''
        orig_aud_feats = self.extract_afeats(aud, for_loop, aud_mask=aud_mask)
        orig_vis_feats = self.extract_vfeats(vis, for_loop, vis_mask=vis_mask) # B, S, ... : S=14, but some of S=i info is in S=i+1
        vis_feats = self.vproj(orig_vis_feats)
        aud_feats = self.aproj(orig_aud_feats)
        losses = defaultdict(list)
        B, S, tv, D = vis_feats.shape
        B, S, ta, D = aud_feats.shape
        vis_feats = vis_feats.view(B, S*tv, D)  # (B, S*tv, D)
        aud_feats = aud_feats.view(B, S*ta, D)  # (B, S*ta, D)
        logits, losses = self.transformer(vis_feats, aud_feats, num_seqs=S, losses=losses)
        losses = compute_loss(logits, targets, losses=losses, prefix='av', scale=1)  # (B,); or a tuple of (B,) and (B,)
        for key, loss in losses.items():
            losses[key] = torch.stack(loss).mean()
        return losses, logits
    def extract_vfeats(self, vis, for_loop, vis_mask=None, num_layers=None):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        if vis_mask is not None:
            vis_mask = vis_mask.permute(0, 1, 3, 2, 4, 5)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis, _ = self.vfeat_extractor(vis, for_loop=for_loop, cont_mask=vis_mask, num_layers=num_layers)
        return vis
    def extract_afeats(self, aud, for_loop, aud_mask=None, num_layers=None):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        if aud_mask is not None:
            aud_mask = aud_mask.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud, for_loop=for_loop, cont_mask=aud_mask, num_layers=num_layers)
        return aud
    def add_lora_adapters(self, cfg):
        module_names = cfg.lora.target_modules
        is_per_module_config = not isinstance(cfg.lora.r, int)
        for idx, module_name in enumerate(module_names):
            module = self
            for sub_module in module_name:
                parent = module
                module = getattr(module, sub_module)
            lora_config = LoraConfig(
                r=cfg.lora.r[idx] if is_per_module_config else cfg.lora.r,
                lora_alpha=cfg.lora.alpha[idx] if is_per_module_config else cfg.lora.alpha,
                target_modules=cfg.lora.lora_target_submodules[idx] if is_per_module_config else cfg.lora.lora_target_submodules,
                lora_dropout=cfg.lora.lora_dropout[idx] if is_per_module_config else cfg.lora.lora_dropout,
                bias=cfg.lora.bias[idx] if is_per_module_config else cfg.lora.bias, 
                layers_to_transform=cfg.lora.layers_to_transform[idx] if is_per_module_config else cfg.lora.layers_to_transform,
                layers_pattern=cfg.lora.layers_pattern[idx] if is_per_module_config else cfg.lora.layers_pattern,
                use_dora=cfg.lora.use_dora[idx] if is_per_module_config else cfg.lora.use_dora, 
                runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True),
                init_lora_weights = "olora",
            )
            module = get_peft_model(module, lora_config)
            frozen_submodules = cfg.get("stage_1_frozen_submodules", None)
            if frozen_submodules is not None:
                for name, param in module.named_parameters():
                    name = ".".join([".".join(module_name)] + name.split(".")[2:])
                    for pattern in frozen_submodules:
                        if re.search(pattern, name):
                            param.requires_grad = False
                            break
            unfrozen_submodules = cfg.get("stage_2_unfrozen_submodules", None)
            if unfrozen_submodules is not None:
                for name, param in module.named_parameters():
                    name = ".".join([".".join(module_name)] + name.split(".")[2:])
                    for pattern in unfrozen_submodules:
                        if re.search(pattern, name):
                            param.requires_grad = True
                            break
            frozen_submodules = cfg.get("stage_3_frozen_submodules", None)
            if frozen_submodules is not None:
                for name, param in module.named_parameters():
                    name = ".".join([".".join(module_name)] + name.split(".")[2:])
                    for pattern in frozen_submodules:
                        if re.search(pattern, name):
                            param.requires_grad = False
                            break
            unfrozen_submodules = cfg.get("stage_4_unfrozen_submodules", None)
            if unfrozen_submodules is not None:
                for name, param in module.named_parameters():
                    name = ".".join([".".join(module_name)] + name.split(".")[2:])
                    for pattern in unfrozen_submodules:
                        if re.search(pattern, name):
                            param.requires_grad = True
                            break
            setattr(parent, module_name[-1], module)
        for name, param in self.named_parameters():
            if (param.requires_grad):
                print("Enabled:", name)

class StreamingSynchformer(BaseSynchformer):
    ''' The module has similar structure to SparseSync (SparseSync) but has a diffrerent
    forward pass. It expects the output of the feature extractors to have global and
    segment-level representations.
    
    NEW: In addition to the streaming head, we now compute an auxiliary contrastive loss.
         For each window we process the visual and audio features through a single transformer layer
         (with learned prepended tokens and positional embeddings) to obtain a set of summary tokens.
         These auxiliary summaries are then compared against the set of CLS tokens produced by the main transformer.
         In the audio branch, if an aud_unshifted signal is provided, we also process it and use its summaries as negatives.
    '''
    def __init__(self, afeat_extractor, vfeat_extractor, aproj, vproj, transformer=None, streaming_head=None, streaming_contrastive_head=None, encode_segments_per_batch=None, **kwargs):
        super().__init__(afeat_extractor, vfeat_extractor, aproj, vproj, transformer)
        self.use_contextual_selectors = streaming_head.params.use_contextual_selectors
        self.streaming_head = instantiate_from_config(streaming_head)
        self.encode_segments_per_batch = encode_segments_per_batch
        self.skip_embed = nn.Parameter(torch.randn(1, transformer.params.num_stream_tokens_per_window, streaming_head.params.n_embd))
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        D = transformer.params.n_embd
        # project concatenated (vis,aud) → D for contrastive
        self.align_proj = nn.Sequential(
            nn.Linear(2 * D, 2*D),
            nn.GELU(),
            nn.Linear(2*D, D)
        )
        # reconstruct back to (vis,aud)
        #self.align_reconstruct = nn.Linear(2*D, 2 * D)
        self.stream_proj = nn.Linear(D, D)
        self.align_temp = 0.07
        self.cls_div_temp = 0.2
        self.align_weight = 1e2
        self.hard_k        = 10      # how many hard negatives to mine per anchor
        self.hard_weight   = 1.0     # relative weight of the hard-neg loss
        self.cls_div_weight = 1e1

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, targets: dict = None, for_loop=False,
                vis_mask: torch.Tensor = None, aud_mask: torch.Tensor = None, others=None, aud_unshifted: torch.Tensor=None, loss_fns=None, **kwargs):
        orig_aud_feats = []
        orig_vis_feats = []

        step = self.encode_segments_per_batch
        for i in range(0, aud.shape[1], step):
            orig_aud_feats.append(
                self.extract_afeats(aud[:, i:i+step].contiguous(), for_loop, aud_mask=aud_mask, num_layers=None)
            )
            orig_vis_feats.append(
                self.extract_vfeats(vis[:, i:i+step].contiguous(), for_loop, vis_mask=vis_mask, num_layers=None)
            )
            
        orig_aud_feats = torch.cat(orig_aud_feats, dim=1)
        orig_vis_feats = torch.cat(orig_vis_feats, dim=1)
        all_vis_feats = self.vproj(orig_vis_feats)
        all_aud_feats = self.aproj(orig_aud_feats)
        
        if (aud_unshifted is not None):
            with torch.no_grad():
                orig_aud_unshifted_feats = []
                for i in range(0, aud.shape[1], step):
                    orig_aud_unshifted_feats.append(
                        self.extract_afeats(aud_unshifted[:, i:i+step].contiguous(), for_loop, aud_mask=aud_mask, num_layers=None)
                    )
                orig_aud_unshifted_feats = torch.cat(orig_aud_unshifted_feats, dim=1)
                all_aud_unshifted_feats = self.aproj(orig_aud_unshifted_feats)
        else:
            all_aud_unshifted_feats = None

        losses = defaultdict(list)
        
        per_window_aud_unshifted_feats = []
        #dec = random.randint(0, all_vis_feats.shape[1] - 20) if self.training else 0
        sampled_sets = [list(range(0, max(4, all_vis_feats.shape[1] - 14 + 1), 1))]
        B = len(all_aud_feats)
        if (self.training):
            for k in range(2):
                sampled_sets.append(
                    random.choices(list(range(0, all_vis_feats.shape[1] - 14 + 1, 1)), k=random.randint(2, all_vis_feats.shape[1]-1))
                )
        all_perf_preds = []
        all_perf_targets = []

        window_len = 14
        M = all_vis_feats.shape[1] - window_len + 1
        test_sampled_sets = [
            list(range(0, M, 2)),                              # even indices (stride-2)
            list(range(1, M, 2)),                              # odd indices
            [0, M // 2],                                        # coarse half sampling
            [M // 3, (2 * M) // 3],                             # two midpoints
            list(range(0, M, 3)),                              # stride-3
            list(range(1, M, 3)),                              # offset stride-3
            [0, M - 1],                                         # first and last only
            list(range(M // 4, M // 4 + 2)) + list(range(M - 2, M)),  # two discontiguous pairs
            random.sample(range(M), min(3, M)),                 # small random subset
            [0, M // 4, M - M // 4 - 1, M - 1],                  # endpoints plus midpoints
            list(range(M)),                                    # full contiguous window
        ]
        #sampled_sets = test_sampled_sets
        for ss, sampled_set in enumerate(sampled_sets):
            seen_samples = []
            all_logits = []
            all_cls_tokens = []
            all_cls_tokens_no_skips = []
            for iter in range(0, all_vis_feats.shape[1] - 14 + 1, 1):
                if (iter not in sampled_set):
                    all_cls_tokens.append(
                        einops.repeat(self.skip_embed, '1 k d -> b k d', b=B)
                    )
                    continue
                vis_feats = all_vis_feats[:, iter:iter+14]
                aud_feats = all_aud_feats[:, iter:iter+14]
                B, S, tv, D = vis_feats.shape
                B, S, ta, D = aud_feats.shape
                # reshape segment features for each window into sequences
                vis_feats = vis_feats.view(B, S*tv, D)  # (B, S*tv, D)
                aud_feats = aud_feats.view(B, S*ta, D)  # (B, S*ta, D)

                if (aud_unshifted is not None):
                    aud_unshift_feats = all_aud_unshifted_feats[:, iter:iter+14]
                    aud_unshift_feats = aud_unshift_feats.view(B, S*ta, D)  # (B, S*ta, D)
                    per_window_aud_unshifted_feats.append(aud_unshift_feats)
                if (not self.use_contextual_selectors):
                    logits, streaming_cls, losses = self.transformer(vis_feats, aud_feats, a_u=aud_unshift_feats if aud_unshifted is not None else None, losses=losses)
                    losses, _ = compute_loss(logits, targets, losses=losses, prefix=f'av_iter', scale=1.0, loss_fns=loss_fns)
                else:
                    logits, _, losses = self.transformer(vis_feats, aud_feats, a_u=aud_unshift_feats if aud_unshifted is not None else None, losses=losses)
                    losses, _ = compute_loss(logits, targets, losses=losses, prefix=f'av_iter', scale=1.0, loss_fns=loss_fns)
                    logits, streaming_cls, losses = self.transformer(vis_feats, aud_feats, a_u=aud_unshift_feats if aud_unshifted is not None else None, selections=(all_cls_tokens, seen_samples), losses=losses)
                    losses, _ = compute_loss(logits, targets, losses=losses, prefix=f'av_hist', scale=1.0, loss_fns=loss_fns)
                seen_samples.append(iter)
                all_logits.append(logits)
                all_cls_tokens.append(streaming_cls)
                all_cls_tokens_no_skips.append(streaming_cls)
                if "align_repr" in loss_fns: 
                    B, K, D = streaming_cls.shape
                    flat_vis_feats = vis_feats.view(B, -1, D)
                    flat_aud_feats = aud_feats.view(B, -1, D)
                    M = flat_vis_feats.size(1)
                    N = flat_aud_feats.size(1)

                    # 1) build all M×N joint reps
                    vis_e = flat_vis_feats.unsqueeze(2).expand(-1, M, N, -1)
                    aud_e = flat_aud_feats.unsqueeze(1).expand(-1, M, N, -1)
                    joint = torch.cat([vis_e, aud_e], dim=-1).view(B, M * N, 2 * D)

                    # 2) project & normalize
                    joint_proj = self.align_proj(joint)       # (B, P, D), where P = M*N
                    cls_norm   = F.normalize(self.stream_proj(streaming_cls), dim=-1)  # (B, K, D)
                    joint_norm = F.normalize(joint_proj,       dim=-1)                     # (B, P, D)

                    # 3) similarity matrix
                    sims = torch.bmm(cls_norm, joint_norm.transpose(1,2)) / self.align_temp  # (B, K, P)

                    # 4) positive indices
                    with torch.no_grad():
                        pos_idx = sims.argmax(dim=2, keepdim=True)  # (B, K, 1)

                    # 5) gather positive scores
                    pos_s = sims.gather(2, pos_idx)               # (B, K, 1)

                    # ── STANDARD InfoNCE ──
                    all_exp = sims.exp().sum(dim=2, keepdim=True)  # sum over P
                    loss_std = -torch.log(pos_s.exp() / all_exp)   # (B, K, 1)

                    # normalize so that random → 1.0 on average
                    P = M * N
                    if P > 1:
                        norm_std = 1.0 / math.log(P)
                        loss_std = loss_std.mean() * norm_std
                    else:
                        loss_std = loss_std.mean()

                    losses[f'loss_align_repr'].append(self.align_weight * loss_std)
                 
                    # ── HARD-NEGATIVE InfoNCE ──
                    mask     = torch.zeros_like(sims, dtype=torch.bool).scatter_(2, pos_idx, True)
                    neg_sims = sims.masked_fill(mask, float('-inf'))        # (B, K, P)
                    hard_vals, _ = neg_sims.topk(self.hard_k, dim=2)        # (B, K, hard_k)
                    hard_exp = hard_vals.exp().sum(dim=2, keepdim=True)     # (B, K, 1)

                    loss_hard = -torch.log(pos_s.exp() / (pos_s.exp() + hard_exp))

                    # normalize so random among (hard_k+1) → 1.0
                    if self.hard_k + 1 > 1:
                        norm_hard = 1.0 / math.log(self.hard_k + 1)
                        loss_hard = loss_hard.mean() * norm_hard
                    else:
                        loss_hard = loss_hard.mean()

                    losses[f'loss_align_repr_hard'].append(
                        self.align_weight * self.hard_weight * loss_hard
                    )
                cls_tokens = torch.stack(all_cls_tokens, dim=1)
                stream_logits, perf_preds, losses = self.streaming_head(cls_tokens, seen_samples, losses=losses)
                losses, perf_targets = compute_loss(stream_logits, targets, losses=losses, prefix=f'av_stream', scale=1, loss_fns=loss_fns)
                all_perf_preds.append(perf_preds.view(B))
                all_perf_targets.append(perf_targets.view(B))
            logits = torch.stack(all_logits, dim=0).mean(dim=0)
            # cls_tokens has shape (B, W, K, D) where W = number of windows and K = num tokens per window.
            cls_tokens = torch.stack(all_cls_tokens, dim=1)
            if "cls_contrastive" in loss_fns:    
                cls_tokens_no_skips = torch.stack(all_cls_tokens_no_skips, dim=1)[:, :, 1:]
                flat_cls_no_skips = cls_tokens.view(B, -1, cls_tokens_no_skips.size(-1))  # (B, N, D)
                
                B, K, D = flat_cls_no_skips.shape
                cls_norm   = F.normalize(self.stream_proj(flat_cls_no_skips), dim=-1)  # (B, K, D)
            
                # ── SELF-CONTRASTIVE ON streaming_cls ──
                # reuse cls_norm from above (B, K, D)
                sims_cls = torch.bmm(cls_norm, cls_norm.transpose(1,2)) / self.cls_div_temp  # (B, K, K)

                with torch.no_grad():
                    idx = torch.arange(K, device=sims_cls.device)
                    pos_idx_cls = idx.view(1, K, 1).expand(B, K, 1)  # (B, K, 1)

                pos_s_cls  = sims_cls.gather(2, pos_idx_cls)                # (B, K, 1)
                all_exp_cls = sims_cls.exp().sum(dim=2, keepdim=True)       # (B, K, 1)
                loss_cls_con = -torch.log(pos_s_cls.exp() / all_exp_cls)    # (B, K, 1)

                # normalize so random among K → 1.0
                if K > 1:
                    norm_self = 1.0 / math.log(K)
                    loss_cls_con = loss_cls_con.mean() * norm_self
                else:
                    loss_cls_con = loss_cls_con.mean()
            
                losses[f'loss_cls_contrastive'].append(self.cls_div_weight * loss_cls_con)

            

        all_perf_preds = torch.stack(all_perf_preds, dim=1)
        all_perf_targets = torch.stack(all_perf_targets, dim=1)
        if "reg_sml1" in loss_fns:
            losses["loss_reg_sml1"].append(
                self.criterion(all_perf_preds, all_perf_targets)
            )
        if (self.training):
            losses["loss_perf_std_true"].append(torch.std(all_perf_targets.detach(), dim=1).mean())
            losses["loss_perf_std_pred"].append(torch.std(all_perf_preds.detach(), dim=1).mean())

        for key, loss_val in losses.items():
            losses[key] = torch.stack(loss_val).mean()
        return losses, stream_logits
    
    
    

class SyncHead(nn.Module):
    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, conf_pred_bins, n_layer, n_head, n_embd, num_stream_tokens_per_window,
                 num_codebooks, codebook_size, pos_emb_cfg=None, off_head_cfg=None) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop,
                             attn_pdrop=attn_pdrop, n_layer=n_layer, n_head=n_head, n_embd=n_embd)

        self.vis_in_lnorm = nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = nn.LayerNorm(self.config.n_embd)
        self.OFF_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        self.STREAM_tok = nn.Parameter(torch.randn(1, num_stream_tokens_per_window, n_embd))
        self.MOD_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = nn.Dropout1d(tok_pdrop)
        self.pos_emb_cfg = instantiate_from_config(pos_emb_cfg)
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        
        self.off_head = instantiate_from_config(off_head_cfg)

        self.selection_pos = SinusoidalPositionalEmbedding(
            n_embd=self.config.n_embd,
            pos_dim=self.config.n_embd // 2,
            max_period=200
        )

        self.apply(init_weights)
    
    def forward(self, v: torch.Tensor, a: torch.Tensor, a_u: torch.Tensor=None, selections: torch.Tensor=None, soft_mask=None, losses=None):
        if losses is None:
            losses = defaultdict(list)
            
        B, Sv, D = v.shape
        B, Sa, D = a.shape
        
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        mod_tok = einops.repeat(self.MOD_tok, '1 1 d -> b 1 d', b=B)
        stream_tok = einops.repeat(self.STREAM_tok, '1 k d -> b k d', b=B)
        
        v = self.vis_in_lnorm(v)
        a = self.aud_in_lnorm(a)
        
        if self.tok_pdrop > 0:
            v = self.tok_drop_vis(v)
            a = self.tok_drop_aud(a)
        
        x = torch.cat((off_tok, v, mod_tok, a), dim=1)
        x = self.pos_emb_cfg(x)
        if (selections and len(selections[0])):
            selections, keep_set = selections
            selections = torch.stack(selections, dim=1)
            _, L, K, _ = selections.shape
            S = L*K
            
            selections = selections.view(B, -1, D)
            pos_emb = self.selection_pos(torch.zeros_like(selections)).flip([1])
            selections = selections + pos_emb

            selections = selections.reshape(B, L, K, D)
            keep_idx = torch.tensor(sorted(keep_set), dtype=torch.long, device=selections.device)
            selections = selections[:, keep_idx, :, :]
            selections = selections.reshape(B, -1, D)
            
            x = torch.cat((x, selections), dim=1)
        x = self.drop(x)
        
        
        x = torch.cat((x, stream_tok), dim=1)
        for block in self.blocks:
            x = block(x)
        off_tok_final = self.ln_f(x[:, 0])
        off_pred = self.off_head(off_tok_final)
        stream_embd = self.ln_f(x[:, -stream_tok.shape[1]:])

        return off_pred, stream_embd, losses



class StreamingHead(nn.Module):
    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop,
                 n_layer, n_head, n_embd, max_period, pos_dim, conf_pred_bins, off_head_cfg=None, **kwargs) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop,
                             resid_pdrop=resid_pdrop,
                             attn_pdrop=attn_pdrop,
                             n_layer=n_layer,
                             n_head=n_head,
                             n_embd=n_embd)
                             
        self.in_lnorm = nn.LayerNorm(self.config.n_embd)
        self.OFF_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        self.PERF_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = nn.Dropout1d(tok_pdrop)

        self.sinusoidal_pos_emb = SinusoidalPositionalEmbedding(
            n_embd=self.config.n_embd,
            pos_dim=pos_dim,
            max_period=max_period
        )
        
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        self.off_head = instantiate_from_config(off_head_cfg)
        self.perf_head = nn.Sequential(
            nn.Linear(self.config.n_embd, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.apply(init_weights)

    def forward(self, selections, keep_set, soft_mask=None, num_seqs=None, losses=None, spacing=None):
        """
        Args:
            toks (Tensor): Input token embeddings of shape (B, S, n_embd).
            spacing (Tensor, optional): A tensor of shape (B, S) that specifies how many units separate
                                        successive token indices. (Note: the off token will always be at 0.)
                                        If not provided, default spacing of 1 is assumed.
        """
        B, L, K, D = selections.shape
        S = L*K
        toks = selections.view(B, -1, D)
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        perf_tok = einops.repeat(self.PERF_tok, '1 1 d -> b 1 d', b=B)
        toks = self.in_lnorm(toks)
        if self.tok_pdrop > 0:
            toks = self.tok_drop_vis(toks)
        
        pos_emb = self.sinusoidal_pos_emb(torch.zeros((B, S//8, D), device=toks.device), spacing=spacing).flip([1]).unsqueeze(2)
        pos_emb = einops.repeat(pos_emb, 'b s 1 d -> b s k d', k=8).reshape(B, -1, D)
        toks = toks + pos_emb

        toks = toks.reshape(B, L, K, D)
        keep_idx = torch.tensor(sorted(keep_set), dtype=torch.long, device=toks.device)
        toks = toks[:, keep_idx, :, :]
        toks = toks.reshape(B, -1, D)

        toks = self.drop(toks)
        x = torch.cat((off_tok, perf_tok, toks), dim=1)

        for block in self.blocks:
            x = block(x)
        off_tok_final = self.ln_f(x[:, 0])
        off_pred = self.off_head(off_tok_final)
        perf_tok_final = self.ln_f(x[:, 1])
        perf_pred = self.perf_head(perf_tok_final)
        return off_pred, perf_pred, losses
    


'''
def forward_with_logging(self,
            vis: torch.Tensor,
            aud: torch.Tensor,
            targets: dict = None,
            for_loop=False,
            vis_mask: torch.Tensor = None,
            aud_mask: torch.Tensor = None,
            others=None):
        top_level_dir = Path.home() / "av_experiment_results"
        top_level_dir.mkdir(parents=True, exist_ok=True)

        # This list will only store metrics from the current call
        current_metrics = []

        # Extract features
        orig_aud_feats = self.extract_afeats(aud, for_loop, aud_mask=aud_mask)
        orig_vis_feats = self.extract_vfeats(vis, for_loop, vis_mask=vis_mask)
        all_vis_feats = self.vproj(orig_vis_feats)
        all_aud_feats = self.aproj(orig_aud_feats)

        effective_inputs = [orig_vis_feats, orig_aud_feats]
        losses = defaultdict(list)
        focus_map_dict = {}

        all_iteration_logits = []
        all_iteration_streaming_cls = []

        iteration_indices = range(0, all_vis_feats.shape[1] - 14 + 2, 2)
        for iter_idx in iteration_indices:
            vis_feats = all_vis_feats[:, iter_idx:iter_idx+14]
            aud_feats = all_aud_feats[:, iter_idx:iter_idx+14]

            B, S, tv, D = vis_feats.shape
            B, S, ta, D = aud_feats.shape

            vis_feats = vis_feats.view(B, S*tv, D)
            aud_feats = aud_feats.view(B, S*ta, D)

            logits, streaming_cls, losses = self.transformer(
                vis_feats, aud_feats, losses=losses
            )
            losses = compute_loss(
                logits, targets, losses=losses, prefix='av_iter', scale=1.0
            )
            all_iteration_logits.append(logits)
            all_iteration_streaming_cls.append(streaming_cls)

        logits_all_iters = torch.stack(all_iteration_logits, dim=0)
        stream_logits = logits_all_iters.mean(dim=0)
        losses = compute_loss(stream_logits, targets,
                                losses=losses, prefix='av_stream', scale=1.0)

        for key in focus_map_dict.keys():
            focus_map_dict[key].detach()

        for key, val in losses.items():
            losses[key] = torch.stack(val).mean()

        B = vis.shape[0]
        full_vid = others["full_video"]
        full_aud = others["full_audio"]

        for b in range(B):
            sample_idx = find_next_index(top_level_dir)
            sample_dir = top_level_dir / f"sample_{sample_idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save the full A/V
            full_av_path = sample_dir / "full_window.mp4"
            save_full_av_video(full_vid[b], full_aud[b], full_av_path)

            # Save the full mel spectrogram
            melspec_path = sample_dir / "melspectrogram_full.png"
            save_melspectrogram_as_png(full_aud[b], sample_rate=16000, out_path=melspec_path)

            window_metrics = []
            num_iters = logits_all_iters.shape[0]
            for iter_i in range(num_iters):
                # Slice the original full video/audio into per-window segments
                window_start_frame = iter_i * 16 // 2  # 2 segments * 16 frames/segment
                window_end_frame = window_start_frame + ((14+1) * 16 // 2)  # 14 segments, 16 frames each
                window_start_audio = window_start_frame * 16000 // 25  # Audio aligned with video
                window_end_audio = window_end_frame * 16000 // 25

                # Clamp to valid ranges
                window_frames = full_vid[b][window_start_frame:window_end_frame]  # (frames, 3, H, W)
                window_audio = full_aud[b][window_start_audio:window_end_audio]   # (samples,)

                # Save the windowed A/V
                window_av_path = sample_dir / f"window_{iter_i}.mp4"
                save_full_av_video(window_frames, window_audio, window_av_path)

                # Save the windowed mel spectrogram
                window_melspec_path = sample_dir / f"melspectrogram_window_{iter_i}.png"
                save_melspectrogram_as_png(window_audio, sample_rate=16000, out_path=window_melspec_path)

                # Compute metrics for the current window
                logits_iter = logits_all_iters[iter_i, b]  # shape (n_classes,)
                probs = F.softmax(logits_iter, dim=-1)
                top_class = int(torch.argmax(probs))
                confidence = float(probs[top_class])
                offset_target = targets["offset_target"][b].item() if targets and "offset_target" in targets else None
                correct = (top_class == offset_target) if offset_target is not None else None

                metrics = {
                    "iteration": iter_i,
                    "ground_truth": offset_target,
                    "predicted_class": top_class,
                    "correct": correct,
                    "confidence": confidence,
                    "softmax_probs": probs.detach().cpu().numpy().tolist(),
                    "video_path": str(window_av_path),
                    "melspectrogram_path": str(window_melspec_path)
                }
                if offset_target is not None:
                    correct_prob = probs[offset_target]
                    metrics.update({
                        "cross_entropy": float(F.cross_entropy(
                            logits_iter.unsqueeze(0),
                            torch.tensor([offset_target]).to(logits_iter.device)
                        )),
                        "rank_of_correct_class": int((probs >= correct_prob).sum().item()),
                        "offset_from_correct": abs(top_class - offset_target)
                    })
                window_metrics.append(metrics)

            # Final streaming-level metrics
            final_probs = F.softmax(stream_logits[b], dim=-1)
            final_top_class = int(torch.argmax(final_probs))
            final_confidence = float(final_probs[final_top_class])
            final_correct = (final_top_class == offset_target) if offset_target is not None else None

            final_metrics = {
                "ground_truth": offset_target,
                "predicted_class": final_top_class,
                "correct": final_correct,
                "confidence": final_confidence,
                "softmax_probs": final_probs.detach().cpu().numpy().tolist()
            }
            if offset_target is not None:
                correct_prob = final_probs[offset_target]
                final_metrics.update({
                    "cross_entropy": float(F.cross_entropy(
                        stream_logits[b].unsqueeze(0),
                        torch.tensor([offset_target]).to(stream_logits.device)
                    )),
                    "rank_of_correct_class": int((final_probs >= correct_prob).sum().item()),
                    "offset_from_correct": abs(final_top_class - offset_target)
                })

            # Collect all metrics for this sample
            sample_metrics = {
                "sample_idx": sample_idx,
                "window_metrics": window_metrics,
                "final_metrics": final_metrics
            }

            # Write sample-specific JSON
            with open(sample_dir / "metrics.json", "w") as f:
                json.dump(sample_metrics, f, indent=4)

            # Append sample_metrics to the *current* forward call's list
            current_metrics.append(sample_metrics)

        # ----------------------------------------------------
        # Now we append our current_metrics to the global JSON
        # ----------------------------------------------------
        global_json_path = top_level_dir / "global_metrics.json"
        if global_json_path.exists():
            # If the global file already exists, read & update
            with open(global_json_path, "r") as f:
                global_metrics = json.load(f)
        else:
            # Otherwise, start fresh
            global_metrics = []

        # Extend the old list with current metrics
        global_metrics.extend(current_metrics)

        # Write everything back out
        with open(global_json_path, "w") as f:
            json.dump(global_metrics, f, indent=4)

        # ----------------------------------------------------
        # Now we append our current_metrics to the global CSV
        # ----------------------------------------------------
        global_csv_path = top_level_dir / "global_metrics.csv"
        file_exists = global_csv_path.exists()

        with open(global_csv_path, "a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Only write header if this file didn't exist before
            if not file_exists:
                csv_writer.writerow([
                    "sample_idx",
                    "iteration",  # -1 for final metrics
                    "ground_truth",
                    "predicted_class",
                    "correct",
                    "confidence",
                    "cross_entropy",
                    "offset_from_correct",
                    "rank_of_correct_class",
                    "softmax_probs",   # as JSON string
                    "video_path",      # Only for per-window
                    "melspectrogram_path"
                ])

            # For each sample in our newly computed metrics
            for sample in current_metrics:
                sample_idx = sample["sample_idx"]
                # Write per-window metrics
                for window in sample["window_metrics"]:
                    csv_writer.writerow([
                        sample_idx,
                        window["iteration"],
                        window.get("ground_truth", None),
                        window["predicted_class"],
                        window["correct"],
                        window["confidence"],
                        window.get("cross_entropy", None),
                        window.get("offset_from_correct", None),
                        window.get("rank_of_correct_class", None),
                        json.dumps(window["softmax_probs"]),
                        window.get("video_path", None),
                        window.get("melspectrogram_path", None)
                    ])

                # Write final metrics as iteration = -1
                final = sample["final_metrics"]
                csv_writer.writerow([
                    sample_idx,
                    -1,  # indicates final metrics
                    final.get("ground_truth", None),
                    final["predicted_class"],
                    final["correct"],
                    final["confidence"],
                    final.get("cross_entropy", None),
                    final.get("offset_from_correct", None),
                    final.get("rank_of_correct_class", None),
                    json.dumps(final["softmax_probs"]),
                    None,
                    None
                ])

        return losses, stream_logits, focus_map_dict, effective_inputs
'''