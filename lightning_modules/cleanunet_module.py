import functools
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import Augmenter
from cleanunet.blend import blend
from cleanunet.cleanspecnet import safe_atan2
from cleanunet.cleanunet2 import CleanUNet2
from cleanunet.metric_disc import MetricDiscriminator, batch_pesq, discriminator_loss, generator_loss
from evaluation.metrics import evaluate_many, summarize
from losses import (AntiWrappingPhaseLoss, CleanUNet2Loss, CompressedMagnitudeLoss, MultiResolutionSTFTLoss,
                    phase_losses)

SPEC_TERMS = ("l1", "mse", "wave", "phase", "complex", "consistency")


class CleanUNetLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training CleanUNet2 (and its waveform-only / spectrogram-only stages).

    Config sections (see configs/*.yaml):
      model:      architecture (CleanUNet2 kwargs), init (pretrained weights), train_cleanunet / train_cleanspecnet
      loss:       mode = hybrid | waveform | spec, waveform/spectral/phase weights, MR-STFT settings,
                  cmag (compressed-magnitude loss), metric_gan (metric discriminator)
      optim:      AdamW + linear warmup + cosine decay, per-submodule LR scales
      augment:    Remix / BandMask / Shift / Gain / SNR probabilities (applied on GPU)
      validation: whole-utterance metrics (PESQ-WB, STOI, SI-SDR) on the validation split
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        cfg = config
        model_cfg = cfg.get("model", {})

        # 1. Model
        self.model = CleanUNet2(**model_cfg.get("architecture", {}))
        self._load_pretrained(model_cfg.get("init", {}) or {})

        # 2. Freezing (glue layers are always trainable)
        self.train_cleanunet = model_cfg.get("train_cleanunet", True)
        self.train_cleanspecnet = model_cfg.get("train_cleanspecnet", True)
        self._set_requires_grad(self.model.clean_unet, self.train_cleanunet)
        if self.model.use_spec_branch:
            self._set_requires_grad(self.model.clean_spec_net, self.train_cleanspecnet)
        print(f"[INFO] CleanUNet: {'TRAIN' if self.train_cleanunet else 'FREEZE'} | CleanSpecNet: "
              f"{'TRAIN' if self.train_cleanspecnet else 'FREEZE'} | trainable params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # 3. Losses
        loss_cfg = cfg.get("loss", {})
        self.loss_mode = loss_cfg.get("mode", "hybrid")
        if self.loss_mode not in ("hybrid", "waveform", "spec"):
            raise ValueError(f"loss.mode must be hybrid | waveform | spec, got '{self.loss_mode}'")
        if self.loss_mode in ("hybrid", "spec") and not self.model.use_spec_branch:
            raise ValueError(f"loss.mode='{self.loss_mode}' needs a model with use_spec_branch=True")
        stft_cfg = loss_cfg.get("stft_config", {})
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg.get("fft_sizes", [512, 1024, 2048]),
            hop_sizes=stft_cfg.get("hop_sizes", [50, 120, 240]),
            win_lengths=stft_cfg.get("win_lengths", [240, 600, 1200]),
            sc_lambda=loss_cfg.get("sc_lambda", 0.5),
            mag_lambda=loss_cfg.get("mag_lambda", 0.5),
            band=loss_cfg.get("stft_band", "full"),
        )
        self.criterion = CleanUNet2Loss(
            ell_p=loss_cfg.get("ell_p", 1),
            ell_p_lambda=loss_cfg.get("ell_p_lambda", 1.0),
            stft_lambda=loss_cfg.get("stft_lambda", 1.0),
            mrstftloss=self.mrstft,
        )
        self.phase_loss = AntiWrappingPhaseLoss(n_fft=1024, hop_length=256, win_length=1024)
        self.weight_waveform = float(loss_cfg.get("weight_waveform", 1.0))
        self.weight_spec = float(loss_cfg.get("weight_spec", 1.0))
        self.weight_phase = float(loss_cfg.get("weight_phase", 0.0))
        # Spectral-branch loss (IMPROVEMENT_PLAN.md, N4/N8), a weighted sum of:
        #   l1 / mse     on the compressed magnitude (default: l1 only = the original CleanSpecNet loss)
        #   wave         L1 + MR-STFT of the resynthesized signal (predicted phase if any, else noisy phase)
        #   phase        anti-wrapping phase losses (needs a phase head)
        #   complex      MSE between compressed complex spectra (needs a phase head)
        #   consistency  STFT(iSTFT(estimate)) vs estimate (needs a phase head)
        self.spec_terms = {k: float(v) for k, v in (loss_cfg.get("spec_terms") or {"l1": 1.0}).items()}
        unknown = set(self.spec_terms) - set(SPEC_TERMS)
        if unknown:
            raise ValueError(f"Unknown loss.spec_terms {sorted(unknown)} (use {SPEC_TERMS})")
        # Power-law compressed magnitude loss on the waveform output, on input-normalized signals (plan, W2)
        cmag = loss_cfg.get("cmag") or {}
        self.cmag_weight = float(cmag.get("weight", 0.0))
        self.cmag = None
        if self.cmag_weight > 0:
            self.cmag = CompressedMagnitudeLoss(n_fft=int(cmag.get("n_fft", 400)),
                                                hop_length=int(cmag.get("hop_length", 100)),
                                                power=float(cmag.get("power", 0.3)))

        # 4. Augmentation and validation settings
        self.sample_rate = int(cfg.get("data", {}).get("sample_rate", 16000))
        self.augmenter = Augmenter(sample_rate=self.sample_rate, **(cfg.get("augment", {}) or {}))
        val_cfg = cfg.get("validation", {}) or {}
        self.val_metric_workers = int(val_cfg.get("metric_workers", 8))
        self.val_log_audio = int(val_cfg.get("log_audio_examples", 3))
        self._val_outputs = []

        # 5. Oracle-condition curriculum for fusion models (IMPROVEMENT_PLAN.md, T8):
        #    CleanSpecNet sees the clean spectrogram with probability p, decayed linearly to 0 over `steps`.
        cur = cfg.get("fusion_curriculum") or {}
        self.oracle_prob0 = float(cur.get("oracle_prob", 0.0))
        self.oracle_steps = int(cur.get("steps", 1))

        # 5b. Polarity augmentation (IMPROVEMENT_PLAN.md, 14.3): with probability p a training pair is multiplied by
        #     -1, so the absolute polarity of the target speech carries no information. Models trained with MUSAN
        #     babble without it keep the speech of VoiceBank's polarity, remove the other one and fail on an inverted
        #     input. Distillation teachers always see the original polarity (their target is flipped instead).
        self.polarity_p = float(cfg.get("augment_polarity", 0.0) or 0.0)
        if not 0.0 <= self.polarity_p <= 1.0:
            raise ValueError(f"augment_polarity must be in [0, 1], got {self.polarity_p}")

        # 6. Condition dropout (IMPROVEMENT_PLAN.md, V2): with probability p a training batch runs with every
        #    conditioning path off (fusion_enabled=False), so the waveform branch stays usable without the condition.
        #    The spectral branch still runs and gets its own loss.
        self.fusion_dropout = float(cfg.get("fusion_dropout", 0.0) or 0.0)
        if not 0.0 <= self.fusion_dropout < 1.0:
            raise ValueError(f"fusion_dropout must be in [0, 1), got {self.fusion_dropout}")

        # 7. Distillation (IMPROVEMENT_PLAN.md, V6 / W3): frozen teachers (CleanUNet2 Lightning checkpoints, each one
        #    optionally blended with a CleanSpecNet like evaluate.py --blend_spec) enhance the same input; the weighted
        #    mean of their outputs is the target of `weight` x waveform loss, added to the loss against the clean
        #    target.
        #    distill: {weight, teacher_checkpoint}  or  {weight, teachers: [{checkpoint, blend_spec, blend_w, weight}]}
        dist = cfg.get("distill") or {}
        self.distill_weight = float(dist.get("weight", 0.0))
        self.teacher = None
        if self.distill_weight > 0:
            specs = dist.get("teachers")
            if specs is None:
                if not dist.get("teacher_checkpoint"):
                    raise ValueError("distill.weight > 0 needs distill.teacher_checkpoint or distill.teachers")
                specs = [{"checkpoint": dist["teacher_checkpoint"]}]
            if not specs or any(not s.get("checkpoint") for s in specs):
                raise ValueError("every entry of distill.teachers needs a checkpoint")
            total_w = sum(float(s.get("weight", 1.0)) for s in specs)
            # a list: the teachers are not registered submodules, so the student's checkpoints do not grow
            self.teacher, self._teacher_plan = [], []
            for s in specs:
                i, j = len(self.teacher), None
                self.teacher.append(self._load_teacher(s["checkpoint"]))
                if s.get("blend_spec"):
                    j = len(self.teacher)
                    self.teacher.append(self._load_teacher(s["blend_spec"]))
                share = float(s.get("weight", 1.0)) / total_w
                self._teacher_plan.append((i, j, float(s.get("blend_w", 0.25)), share))

        # 8. Metric discriminator (IMPROVEMENT_PLAN.md, W2): MetricGAN+-style adversarial PESQ loss. Two optimizers,
        #    so the module switches to manual optimization and clips the gradients itself (train.py then leaves
        #    trainer.gradient_clip_val out). The first `warmup_steps` steps train the discriminator alone.
        mg = loss_cfg.get("metric_gan") or {}
        self.mgan_weight = float(mg.get("weight", 0.0))
        self.metric_disc = None
        if self.mgan_weight > 0:
            if self.loss_mode == "spec":
                raise ValueError("loss.metric_gan needs a waveform output (loss.mode waveform or hybrid)")
            self.automatic_optimization = False
            # highpass_hz: band-limited PESQ target and discriminator input (cleanunet/metric_disc.py)
            hp = float(mg.get("highpass_hz", 0.0))
            self.metric_disc = MetricDiscriminator(**(mg.get("disc") or {}), highpass_hz=hp,
                                                   sample_rate=self.sample_rate)
            self.mgan_warmup = int(mg.get("warmup_steps", 300))
            self.mgan_d_lr = float(mg.get("d_lr", 5e-4))
            self.mgan_workers = int(mg.get("pesq_workers", 8))
            self.clip_val = float((cfg.get("trainer") or {}).get("gradient_clip_val") or 0.0)
            self.register_buffer("mgan_step", torch.zeros((), dtype=torch.long))   # survives a resume
            self._pesq_pool = None
            # (ref, est, pool) -> normalized PESQ per row; replaceable in tests
            self.pesq_fn = functools.partial(batch_pesq, sample_rate=self.sample_rate, highpass_hz=hp)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _set_requires_grad(module: nn.Module, requires: bool):
        for p in module.parameters():
            p.requires_grad = requires

    def _load_pretrained(self, init: dict):
        """Load pretrained weights. Errors are raised (never silently ignored)."""
        if init.get("full_checkpoint"):
            self.model.load_cleanunet2_weights(init["full_checkpoint"], strict=init.get("strict", True))
        if init.get("cleanunet_from_cleanumamba"):
            self.model.load_cleanumamba_weights(init["cleanunet_from_cleanumamba"])
        if init.get("cleanunet_checkpoint"):
            self.model.load_cleanunet_weights(init["cleanunet_checkpoint"])
        if init.get("cleanspecnet_checkpoint"):
            self.model.load_cleanspecnet_weights(init["cleanspecnet_checkpoint"])

    @staticmethod
    def _load_teacher(path: str) -> nn.Module:
        """Frozen CleanUNet2 from a Lightning checkpoint (architecture read from its stored config). It is kept out
        of the module's state dict, so the student's checkpoints do not grow."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        teacher = CleanUNet2(**ckpt["hyper_parameters"]["model"]["architecture"])
        teacher.load_cleanunet2_weights(path)
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher.eval()

    def _teacher_output(self, noisy: torch.Tensor) -> torch.Tensor:
        for teacher in self.teacher:
            if next(teacher.parameters()).device != noisy.device:
                teacher.to(noisy.device)
        x = noisy.view(noisy.shape[0], 1, -1)
        with torch.no_grad():
            target = 0.0
            for i, j, blend_w, weight in self._teacher_plan:
                y = self.teacher[i](x)["wav"].float()
                if j is not None:
                    y = blend(y, x, self.teacher[j], blend_w)
                target = target + weight * y
            return target.detach()

    def _get_pesq_pool(self):
        """Process pool for the PESQ scores of the metric discriminator (created at the first use: fork, the
        workers never touch CUDA, as in evaluation.metrics.evaluate_many)."""
        if self.mgan_workers <= 1:
            return None
        if self._pesq_pool is None:
            from multiprocessing import get_context
            from evaluation.metrics import _limit_threads
            self._pesq_pool = get_context("fork").Pool(self.mgan_workers, initializer=_limit_threads)
        return self._pesq_pool

    def on_train_end(self):
        if getattr(self, "_pesq_pool", None) is not None:
            self._pesq_pool.close()
            self._pesq_pool.join()
            self._pesq_pool = None

    def _frozen_modules(self):
        mods = []
        if not self.train_cleanunet:
            mods.append(self.model.clean_unet)
        if self.model.use_spec_branch and not self.train_cleanspecnet:
            mods.append(self.model.clean_spec_net)
        return mods

    def on_train_epoch_start(self):
        # Frozen submodules stay in eval mode (no dropout / BatchNorm statistics updates)
        for m in self._frozen_modules():
            m.eval()

    def forward(self, noisy: torch.Tensor):
        return self.model(noisy)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer is not None and self.trainer.training and self.augmenter.enabled:
            noisy, clean, ids = batch
            noisy, clean = self.augmenter(noisy, clean)
            batch = (noisy, clean, ids)
        return batch

    def _spectral_loss(self, out, noisy, clean, logs):
        """Weighted spectral-branch loss (see `self.spec_terms`)."""
        t, m = self.spec_terms, self.model
        scale = out["scale"]
        target = m.target_spectrogram(clean, scale)
        T = min(out["spec"].shape[-1], target.shape[-1])
        spec, target = out["spec"].float()[..., :T], target[..., :T]
        phase = out.get("phase")
        phase = None if phase is None else phase.float()[..., :T]
        total = 0.0

        def add(name, value):
            nonlocal total
            total = total + t[name] * value
            logs[f"spec/{name}"] = value.detach()

        if t.get("l1"):
            add("l1", F.l1_loss(spec, target))
        if t.get("mse"):
            add("mse", F.mse_loss(spec, target))
        if phase is not None and (t.get("phase") or t.get("complex")):
            ref_phase = torch.angle(m.target_stft(clean, scale))[..., :T]
            if t.get("phase"):
                add("phase", phase_losses(phase, ref_phase, weight=target))
            if t.get("complex"):
                est_c = torch.polar(spec.clamp(min=0), phase)
                ref_c = torch.polar(target, ref_phase)
                add("complex", (est_c - ref_c).abs().pow(2).mean())
        if t.get("wave") or (phase is not None and t.get("consistency")):
            x_norm = noisy.view(noisy.shape[0], 1, -1).float() / scale
            y = m.spec_to_wave(spec, x_norm, clean.shape[-1], phase=phase)
            if t.get("wave"):
                add("wave", self.criterion(clean.view_as(y), y * scale))
            if phase is not None and t.get("consistency"):
                S = m.spectrogram
                Y = S.complex_stft(y)[..., :T]
                resyn = torch.polar(S.compress(Y.abs() * S.scale), safe_atan2(Y.imag, Y.real))
                add("consistency", (resyn - torch.polar(spec.clamp(min=0), phase)).abs().pow(2).mean())
        return total

    def _flip_polarity(self, noisy, clean):
        """Training pair with a random per-example sign (augment_polarity) and the signs (B, 1, 1), or None."""
        if not (self.training and self.polarity_p > 0):
            return noisy, clean, None
        flip = torch.rand(noisy.shape[0], 1, 1, device=noisy.device) < self.polarity_p
        sign = 1.0 - 2.0 * flip.to(noisy.dtype)
        return noisy * sign, clean * sign.view(-1, *([1] * (clean.dim() - 1))), sign

    def _compute_losses(self, noisy, clean, teacher_input=None):
        """`teacher_input` = (noisy with its original polarity, signs) when the pair was flipped: the distillation
        target is then sign x teacher(original input)."""
        logs = {}
        if self.loss_mode == "spec":
            out = self.model(noisy, spec_only=True)
            loss_spec = self._spectral_loss(out, noisy, clean, logs)
            logs["spec_loss"] = loss_spec.detach()
            return loss_spec, out, logs

        fusion_enabled = True
        if self.training and self.fusion_dropout > 0:
            fusion_enabled = bool(torch.rand(()) >= self.fusion_dropout)
            logs["fusion_dropped"] = torch.tensor(float(not fusion_enabled))
        out = self.model(noisy, clean=clean, fusion_enabled=fusion_enabled)
        enhanced = out["wav"]
        loss_wave, comps = self.criterion(clean, enhanced, return_components=True)
        total = self.weight_waveform * loss_wave
        logs["waveform_loss"] = loss_wave.detach()
        logs.update({f"waveform/{k}": v for k, v in comps.items()})
        if self.cmag is not None:
            scale = out["scale"]
            loss_cm = self.cmag(enhanced / scale, clean.view_as(enhanced) / scale)
            total = total + self.cmag_weight * loss_cm
            logs["cmag_loss"] = loss_cm.detach()
        if self.teacher is not None and self.training:
            if teacher_input is None:
                target = self._teacher_output(noisy)
            else:
                original, sign = teacher_input
                target = self._teacher_output(original) * sign.view(-1, 1, 1).float()
            target = target.to(enhanced.dtype)
            loss_kd = self.criterion(target.view_as(enhanced), enhanced)
            total = total + self.distill_weight * loss_kd
            logs["distill_loss"] = loss_kd.detach()
        if self.weight_phase > 0:
            loss_phase = self.phase_loss(enhanced, clean)
            total = total + self.weight_phase * loss_phase
            logs["phase_loss"] = loss_phase.detach()
        if self.loss_mode == "hybrid" and self.weight_spec > 0 and out["spec"] is not None:
            loss_spec = self._spectral_loss(out, noisy, clean, logs)
            total = total + self.weight_spec * loss_spec
            logs["spec_loss"] = loss_spec.detach()
        return total, out, logs

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        noisy, clean, _ = batch
        if self.oracle_prob0 > 0:
            self.model.oracle_prob = self.oracle_prob0 * max(0.0, 1.0 - self.global_step / self.oracle_steps)
            self.log("train/oracle_prob", self.model.oracle_prob, on_step=True, batch_size=noisy.shape[0])
        original = noisy
        noisy, clean, sign = self._flip_polarity(noisy, clean)
        teacher_input = None if sign is None else (original, sign)
        if self.metric_disc is not None:
            total, logs = self._metric_gan_step(noisy, clean, teacher_input)
        else:
            total, _, logs = self._compute_losses(noisy, clean, teacher_input)
        for k, v in logs.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, batch_size=noisy.shape[0])
        self.log("train/total_loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=noisy.shape[0])
        return total.detach() if self.metric_disc is not None else total

    def _metric_gan_step(self, noisy, clean, teacher_input=None):
        """One manual-optimization step: the model with its usual losses + weight x (D(clean, est) - 1)^2, then the
        discriminator on the measured PESQ of the (detached) estimate. Signals are input-normalized."""
        opt_g, opt_d = self.optimizers()
        sched = self.lr_schedulers()
        total, out, logs = self._compute_losses(noisy, clean, teacher_input)
        scale = out["scale"].float()
        est = out["wav"].float() / scale
        ref = clean.view_as(out["wav"]).float() / scale
        if int(self.mgan_step) >= self.mgan_warmup:
            self.metric_disc.requires_grad_(False)
            g_loss = generator_loss(self.metric_disc, ref, est)
            self.metric_disc.requires_grad_(True)
            total = total + self.mgan_weight * g_loss
            logs["mgan/g_loss"] = g_loss.detach()
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(total)
        if self.clip_val > 0:
            self.clip_gradients(opt_g, gradient_clip_val=self.clip_val, gradient_clip_algorithm="norm")
        opt_g.step()
        if sched is not None:
            sched.step()

        scores = self.pesq_fn(ref, est, pool=self._get_pesq_pool())
        d_loss = discriminator_loss(self.metric_disc, ref, est.detach(), scores)
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.mgan_step += 1
        ok = torch.isfinite(scores)
        logs["mgan/d_loss"] = d_loss.detach()
        logs["mgan/pesq_valid"] = ok.float().mean()
        if ok.any():
            logs["mgan/pesq"] = scores[ok].mean() * 3.5 + 1.0
        return total, logs

    # -------------------------
    # Validation (whole utterances, batch_size=1)
    # -------------------------
    def validation_step(self, batch, batch_idx):
        noisy, clean, ids = batch
        total, out, logs = self._compute_losses(noisy, clean)
        bs = noisy.shape[0]
        self.log("val_loss", total, on_epoch=True, sync_dist=True, batch_size=bs)
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True, batch_size=bs)

        if out["wav"] is not None:
            enhanced = out["wav"].float().squeeze(1).cpu().numpy()
            ref = clean.float().squeeze(1).cpu().numpy()
            for i in range(bs):
                self._val_outputs.append((ref[i], enhanced[i]))
            if batch_idx < self.val_log_audio:
                self._log_audio(batch_idx, noisy[0], out["wav"][0], clean[0])
        return total

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        results = evaluate_many(self._val_outputs, sample_rate=self.sample_rate, full=False,
                                num_workers=self.val_metric_workers)
        self._val_outputs = []
        s = summarize(results)
        if s["n_files"] == 0:
            print("[WARNING] Validation metrics failed for every file (e.g. silent outputs).")
            pesq_v, stoi_v, sisdr_v = 1.0, 0.0, -50.0
        else:
            pesq_v, stoi_v, sisdr_v = s["pesq_wb"], s["stoi"], s["si_sdr"]
        score = (stoi_v + pesq_v / 4.5 + sisdr_v / 30.0) / 3.0
        self.log_dict({"val/pesq": pesq_v, "val/stoi": stoi_v, "val/si_sdr": sisdr_v, "val/weighted_score": score,
                       "val/metric_errors": float(s["n_errors"])}, prog_bar=True, sync_dist=True)

    def _log_audio(self, idx, noisy, enhanced, clean):
        if self.logger is None:
            return
        tracks = {"noisy": noisy, "enhanced": enhanced, "clean": clean}
        exp = getattr(self.logger, "experiment", None)
        try:
            if hasattr(exp, "add_audio"):                                   # TensorBoard
                for name, wav in tracks.items():
                    exp.add_audio(f"val/sample_{idx}/{name}", wav.float().reshape(1, -1).cpu(),
                                  global_step=self.global_step, sample_rate=self.sample_rate)
            elif hasattr(self.logger, "log_audio"):                         # Weights & Biases
                self.logger.log_audio(key=f"val/sample_{idx}",
                                      audios=[w.float().reshape(-1).cpu().numpy() for w in tracks.values()],
                                      caption=list(tracks), sample_rate=[self.sample_rate] * 3)
        except Exception as err:
            print(f"[WARNING] Could not log audio example {idx}: {err}")

    # -------------------------
    # Optimizers
    # -------------------------
    def configure_optimizers(self):
        optim_cfg = self.hparams.get("optim", {}) or {}
        lr = float(optim_cfg.get("lr", 1e-4))
        weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        lr_scale = optim_cfg.get("lr_scale", {}) or {}

        # Weight decay only on >=2D weights, never on biases, norms, Mamba A_log / D
        groups = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            top = name.split(".")[0]
            scale = float(lr_scale.get(top, 1.0))
            no_decay = p.dim() < 2 or getattr(p, "_no_weight_decay", False) or name.endswith("A_log")
            key = (scale, no_decay)
            groups.setdefault(key, []).append(p)
        param_groups = [{"params": ps, "lr": lr * s, "weight_decay": 0.0 if nd else weight_decay}
                        for (s, nd), ps in groups.items()]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
                                      eps=float(optim_cfg.get("eps", 1e-8)))

        opt_d = None
        if self.metric_disc is not None:
            opt_d = torch.optim.AdamW(self.metric_disc.parameters(), lr=self.mgan_d_lr, betas=(0.8, 0.99))
        if optim_cfg.get("schedule", "cosine") == "constant":
            return optimizer if opt_d is None else [optimizer, opt_d]
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = int(float(optim_cfg.get("warmup_ratio", 0.05)) * total_steps)
        min_ratio = float(optim_cfg.get("min_lr_ratio", 0.01))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if opt_d is not None:          # manual optimization: the scheduler is stepped in _metric_gan_step
            return [optimizer, opt_d], [{"scheduler": scheduler, "interval": "step"}]
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
