"""
PyTorch Lightning module for CleanUNet2 Stage-2 training with SSL Embeddings.

Stage-2: Training without SSL embedding extraction, replicating latent vectors from Stage-1.
The model learns to predict the fused latents without using the embedding extractor,
enabling fast inference while maintaining the benefits of speaker information.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import torchaudio

from cleanunet.cleanunet2_with_ssl_embeddings import CleanUNet2WithSSLEmbeddings
from cleanunet.ssl_extractor_factory import ssl_args_from_config, fusion_args_from_config, latent_predictor_args_from_config
from losses import CleanUNet2Loss, MultiResolutionSTFTLoss, AntiWrappingPhaseLoss
from spec_dataset import latent_cache_key

# Import TorchMetrics
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class CleanUNet2SSLEmbeddingsStage2Module(pl.LightningModule):
    """
    Lightning module for Stage-2 training with SSL Embeddings (multi-backbone).

    Trains the model to replicate Stage-1's fused latent vectors using
    only the noisy audio (no SSL embedding extractor).
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        print("=" * 80)
        print("STAGE-2: Replicating Latents (SSL Embeddings - Inference Mode)")
        print("=" * 80)

        # ===== Model Initialization =====
        model_config = config.get('model', {})

        # Generic SSL embedding configuration (model.ssl.*), with legacy fallback.
        # Note: Stage 2 doesn't run the extractor, but the model is built with the
        # same args so the architecture (integration block, dims) matches Stage 1.
        model_args = {
            'stage': 'stage2',
            'conditioning_type': model_config.get('conditioning_type', 'addition'),
            'cleanunet_params': model_config.get('cleanunet_params', {}),
            'cleanspecnet_params': model_config.get('cleanspecnet_params', {}),
            **ssl_args_from_config(model_config),
            # Fusion type must match Stage 1 so the architecture/weights line up.
            **fusion_args_from_config(model_config),
            # Stage-2-only latent predictor architecture.
            **latent_predictor_args_from_config(model_config),
        }

        self.model = CleanUNet2WithSSLEmbeddings(**model_args)

        # ===== Load Stage-1 Checkpoint =====
        stage1_ckpt = config.get('stage1_checkpoint') or config.get('pipeline', {}).get('stage1_checkpoint')
        if stage1_ckpt:
            print(f"[Stage-2] Loading Stage-1 checkpoint: {stage1_ckpt}")
            self.model.load_stage1_weights(stage1_ckpt)
        else:
            print("[WARNING] No Stage-1 checkpoint provided. Training from scratch.")

        # ===== Loss Initialization =====
        loss_cfg = config.get('losses') or config.get('loss', {})

        # Multi-Resolution STFT Loss
        stft_cfg = loss_cfg.get('stft_config', {})
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg.get('fft_sizes', [512, 1024, 2048]),
            hop_sizes=stft_cfg.get('hop_sizes', [128, 256, 512]),
            win_lengths=stft_cfg.get('win_lengths', [512, 1024, 2048]),
            sc_lambda=loss_cfg.get('sc_lambda', 0.1),
            mag_lambda=loss_cfg.get('mag_lambda', 0.1)
        )

        # Primary waveform loss
        self.criterion = CleanUNet2Loss(
            ell_p=loss_cfg.get('ell_p', 1),
            ell_p_lambda=loss_cfg.get('ell_p_lambda', 1.0),
            stft_lambda=loss_cfg.get('stft_lambda', 1.0),
            mrstftloss=mrstft_loss
        )

        # Phase loss
        self.phase_loss = AntiWrappingPhaseLoss(
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )

        # Loss weights
        self.weight_waveform = float(loss_cfg.get('weight_waveform', 10.0))
        self.weight_spec = float(loss_cfg.get('weight_spec', 1.0))
        self.weight_phase = float(loss_cfg.get('weight_phase', 1.0))
        self.gamma_latent = float(loss_cfg.get('gamma_latent', 0.05))  # From paper

        print(f"[Stage-2] Loss weights: waveform={self.weight_waveform}, "
              f"spec={self.weight_spec}, phase={self.weight_phase}")
        print(f"[Stage-2] Latent replication weight (γ): {self.gamma_latent}")

        # ===== Metrics Initialization =====
        sr = config.get('audio', {}).get('sample_rate', 16000)
        self.sample_rate = sr

        # PESQ only supports 8kHz or 16kHz
        if sr < 8000:
            raise ValueError(
                f"[Stage-2] ERROR: Sample rate {sr} Hz is too low for PESQ metric. "
                f"PESQ requires at least 8000 Hz. Please use sample_rate >= 8000 in your config."
            )
        elif sr in [8000, 16000]:
            # Use sample rate directly for PESQ
            self.pesq_sample_rate = sr
            print(f"[Stage-2] Using sample rate {sr} Hz for PESQ metric")
        else:
            # Sample rate > 16000: downsample to 16kHz for PESQ calculation
            self.pesq_sample_rate = 16000
            print(f"[Stage-2] ⚠️  WARNING: Sample rate is {sr} Hz, but PESQ only supports 8kHz/16kHz.")
            print(f"[Stage-2] Audio will be downsampled to {self.pesq_sample_rate} Hz for PESQ calculation.")

        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=self.pesq_sample_rate, mode='wb')
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=False)
        self.val_sisdr = ScaleInvariantSignalNoiseRatio()

        # Store resampler config (but don't create the resampler itself yet)
        # This avoids saving it in the checkpoint, preventing compatibility issues
        self._pesq_resampler_config = {
            'needed': self.sample_rate != self.pesq_sample_rate,
            'orig_freq': self.sample_rate,
            'new_freq': self.pesq_sample_rate
        }
        self._pesq_resampler_cache = None

        # ===== Per-file latent caches (keyed by MD5 of the clean-audio path) =====
        # Generated by generate_latents.py from a trained Stage-1 checkpoint. Both training
        # and validation use the same path-keyed lookup; alignment relies on the matching
        # deterministic crop (applied to train AND val) so each sample lines up with its
        # cached fused-latent target.
        #
        # TRAIN cache is REQUIRED: Stage-2 trains ONLY if it exists (distillation targets).
        train_dir = config.get('train_latents_dir')
        if not train_dir:
            raise ValueError(
                "[Stage-2] 'train_latents_dir' is not set in the config. Stage-2 distillation "
                "requires the Stage-1 TRAIN latents. Generate them with "
                "`generate_latents.py --split train` and set 'train_latents_dir'."
            )
        self.train_latents_dir, self.train_latent_keys = self._load_latent_cache(train_dir)
        self._train_latent_mem = {}
        if not self.train_latent_keys:
            raise FileNotFoundError(
                f"[Stage-2] No train latents found in '{self.train_latents_dir}'. Stage-2 will not "
                f"train without them. Generate first:\n"
                f"  python generate_latents.py --split train --config <stage1-config> "
                f"--checkpoint <stage1.ckpt> --output-dir {self.train_latents_dir}"
            )
        print(f"[Stage-2] Train latent cache: {len(self.train_latent_keys)} files "
              f"in {self.train_latents_dir} (training-time latent loss ENABLED)")
        self._assert_latents_aligned(config, 'train_list_path', self.train_latents_dir,
                                     self.train_latent_keys, required=True)

        # VAL cache is OPTIONAL (metric only): if present, the validation latent loss is
        # computed the same path-keyed way; if absent, val/loss_latent stays 0.
        self.val_latents_dir, self.val_latent_keys = self._load_latent_cache(config.get('val_latents_dir'))
        self._val_latent_mem = {}
        if self.val_latent_keys:
            print(f"[Stage-2] Val latent cache: {len(self.val_latent_keys)} files "
                  f"in {self.val_latents_dir} (validation latent metric ENABLED)")
            self._assert_latents_aligned(config, 'val_list_path', self.val_latents_dir,
                                         self.val_latent_keys, required=False)
        else:
            print(f"[Stage-2] No val latent cache (val_latents_dir={config.get('val_latents_dir')}); "
                  f"validation latent metric = 0. Generate with `generate_latents.py --split val`.")

        # Audio samples for logging (6 samples: noisy, clean, denoised)
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _load_latent_cache(self, latents_dir):
        """Return (Path|None, set_of_stems) for a per-file <md5>.pt latent cache."""
        d = Path(latents_dir) if latents_dir else None
        keys = {p.stem for p in d.glob('*.pt')} if (d is not None and d.exists()) else set()
        return d, keys

    def _get_cached_latent(self, clean_path, latents_dir, keys, mem):
        """Lazily load the cached Stage-1 fused latent (C, T) for a clean-audio path,
        or None if not cached."""
        key = latent_cache_key(clean_path)
        if key not in keys:
            return None
        if key not in mem:
            mem[key] = torch.load(latents_dir / f"{key}.pt", map_location='cpu').float()
        return mem[key]

    def _cached_latent_loss(self, predicted_latent, clean_paths, latents_dir, keys, mem):
        """MSE between the predicted latent and the cached Stage-1 fused latents for this
        batch. Returns 0 if the cache is disabled, paths are unavailable, or any sample in
        the batch is not cached (alignment must be complete to be valid)."""
        zero = torch.zeros((), device=self.device)
        if predicted_latent is None or clean_paths is None or not keys:
            return zero
        targets = []
        for p in clean_paths:
            t = self._get_cached_latent(p, latents_dir, keys, mem)
            if t is None:
                return zero
            targets.append(t)
        target = torch.stack(targets).to(device=predicted_latent.device,
                                         dtype=predicted_latent.dtype)
        min_t = min(predicted_latent.size(-1), target.size(-1))
        return F.mse_loss(predicted_latent[..., :min_t], target[..., :min_t])

    def _assert_latents_aligned(self, config, list_key, latents_dir, keys,
                                required, min_ratio=0.9):
        """Verify the cache keys align with a split's file paths (md5(os.path.join(
        data_dir, rel))). Abort if `required` and misaligned; otherwise warn. Skips
        multi-dataset / pathless configs (can't reconstruct the same keys here)."""
        import os
        data_cfg = config.get('data', {})
        list_path = data_cfg.get(list_key)
        data_dir = data_cfg.get('data_dir', '.')

        if not list_path or data_cfg.get('datasets'):
            print(f"[Stage-2] Skipping latent-alignment check for {list_key} "
                  f"(no single {list_key}).")
            return

        try:
            from spec_dataset import get_dataset_filelist
            pairs = get_dataset_filelist(list_path)
        except Exception as e:
            print(f"[Stage-2] Could not read {list_key} for alignment check ({e}); skipping.")
            return
        if not pairs:
            return

        hit = sum(latent_cache_key(os.path.join(data_dir, c)) in keys for c, _ in pairs)
        ratio = hit / len(pairs)
        if ratio < min_ratio:
            msg = (f"[Stage-2] Latents MISALIGNED with {list_key}: only {hit}/{len(pairs)} "
                   f"paths have a cached latent in '{latents_dir}' "
                   f"({ratio:.0%} < {min_ratio:.0%}). Likely a wrong cache dir or a "
                   f"data_dir/{list_key} mismatch. Regenerate with generate_latents.py "
                   f"using the SAME config.")
            if required:
                raise RuntimeError(msg + " The training latent loss would be silently zeroed.")
            print("[WARNING] " + msg + " The validation latent metric will be ~0.")
            return
        print(f"[Stage-2] Latents aligned with {list_key}: {hit}/{len(pairs)} ({ratio:.0%}).")

    def _get_pesq_resampler(self):
        """
        Lazily creates and returns the PESQ resampler.
        This avoids saving it in checkpoints, preventing compatibility issues.
        """
        if not self._pesq_resampler_config['needed']:
            return None

        if self._pesq_resampler_cache is None:
            self._pesq_resampler_cache = torchaudio.transforms.Resample(
                orig_freq=self._pesq_resampler_config['orig_freq'],
                new_freq=self._pesq_resampler_config['new_freq']
            )
        return self._pesq_resampler_cache

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom state_dict loading that filters out incompatible keys from old checkpoints.

        This handles cases where old checkpoints have keys that don't exist in the current model,
        such as _pesq_resampler_cache which was changed from a saved object to None.
        """
        # List of keys to ignore (known incompatibilities from old checkpoints)
        keys_to_ignore = [
            '_pesq_resampler_cache.kernel',
            '_pesq_resampler_cache.width',
            '_pesq_resampler_cache',
        ]

        # Filter out incompatible keys
        filtered_state_dict = {}
        ignored_keys = []

        for key, value in state_dict.items():
            # Check if this key should be ignored
            should_ignore = any(key.startswith(ignore_key) for ignore_key in keys_to_ignore)

            if should_ignore:
                ignored_keys.append(key)
            else:
                filtered_state_dict[key] = value

        # Print info about ignored keys
        if ignored_keys:
            print(f"[INFO] Ignoring {len(ignored_keys)} incompatible keys from checkpoint:")
            for key in ignored_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(ignored_keys) > 5:
                print(f"  ... and {len(ignored_keys) - 5} more")

        # Call parent's load_state_dict with filtered dict
        return super().load_state_dict(filtered_state_dict, strict=strict)

    def forward(self, noisy_wav, noisy_spec):
        return self.model(noisy_wav, noisy_spec, clean_audio=None)

    def training_step(self, batch, batch_idx):
        # Handle both dataset formats: with and without file paths
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, clean_audio_paths = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch
            clean_audio_paths = None

        # Forward without X-Vectors
        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_audio=None,
            return_latents=True
        )

        predicted_latent = latents['predicted_latent']

        # ===== Compute Reconstruction Losses =====
        loss_waveform = self.criterion(clean_wav, enhanced)

        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        loss_phase = self.phase_loss(enhanced, clean_wav)

        loss_recon = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

        # ===== Compute Latent Replication Loss =====
        # Training-time latent distillation: match the predicted latent to the cached
        # Stage-1 fused latent for each sample (keyed by clean path, aligned via the
        # deterministic train crop). Falls back to 0 if the cache is unavailable.
        loss_latent = self._cached_latent_loss(
            predicted_latent, clean_audio_paths,
            self.train_latents_dir, self.train_latent_keys, self._train_latent_mem)

        # Total loss (Eq. 5 from paper)
        total_loss = loss_recon + self.gamma_latent * loss_latent

        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_recon', loss_recon, on_step=False, on_epoch=True)
        self.log('train/loss_waveform', loss_waveform, on_step=False, on_epoch=True)
        self.log('train/loss_spec', loss_spec, on_step=False, on_epoch=True)
        self.log('train/loss_phase', loss_phase, on_step=False, on_epoch=True)
        self.log('train/loss_latent', loss_latent, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Handle both dataset formats: with and without file paths
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, clean_audio_paths = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch
            clean_audio_paths = None

        # Forward without X-Vectors
        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_audio=None,
            return_latents=True
        )

        predicted_latent = latents['predicted_latent']

        # ===== Compute Reconstruction Losses =====
        loss_waveform = self.criterion(clean_wav, enhanced)

        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        loss_phase = self.phase_loss(enhanced, clean_wav)

        loss_recon = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

        # ===== Compute Latent Replication Loss (path-keyed, same mechanism as training) =====
        # Aligns each val sample with its cached Stage-1 fused latent via the deterministic
        # val crop. 0 if the val cache is absent/misaligned (metric only — see __init__).
        loss_latent = self._cached_latent_loss(
            predicted_latent, clean_audio_paths,
            self.val_latents_dir, self.val_latent_keys, self._val_latent_mem)

        # Total loss
        total_loss = loss_recon + self.gamma_latent * loss_latent

        # ===== Compute Metrics (Safe Mode) =====
        # Disable autocast for metrics computation to ensure float32 precision
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Convert to float32 for metrics (AMP uses float16, but metrics need float32)
            preds = enhanced.squeeze(1).float()
            target = clean_wav.squeeze(1).float()

            is_silent_or_nan = (preds.abs().max() < 1e-5) or torch.isnan(preds).any()

            if is_silent_or_nan:
                val_pesq = torch.tensor(1.0, device=self.device)
                val_stoi = torch.tensor(1e-5, device=self.device)
                val_sisdr = torch.tensor(-50.0, device=self.device)
            else:
                try:
                    # Resample for PESQ if needed
                    pesq_resampler = self._get_pesq_resampler()
                    if pesq_resampler is not None:
                        preds_pesq = pesq_resampler(preds)
                        target_pesq = pesq_resampler(target)
                    else:
                        preds_pesq = preds
                        target_pesq = target

                    # CORRECTED: PESQ expects (reference, degraded) order, i.e., (clean, enhanced)
                    # Move to CPU for PESQ calculation (PESQ internal weights are on CPU)
                    preds_pesq_cpu = preds_pesq.cpu()
                    target_pesq_cpu = target_pesq.cpu()

                    val_pesq = self.val_pesq(target_pesq_cpu, preds_pesq_cpu)
                except Exception as e:
                    print(f"[WARNING] PESQ computation failed: {e}")
                    val_pesq = torch.tensor(1.0, device=self.device)

                try:
                    # CORRECTED: STOI expects (reference, degraded) order
                    val_stoi = self.val_stoi(target, preds)
                except Exception as e:
                    print(f"[WARNING] STOI computation failed: {e}")
                    val_stoi = torch.tensor(1e-5, device=self.device)

                try:
                    # CORRECTED: SI-SDR expects (reference, degraded) order
                    val_sisdr = self.val_sisdr(target, preds)
                except Exception as e:
                    print(f"[WARNING] SI-SDR computation failed: {e}")
                    val_sisdr = torch.tensor(-50.0, device=self.device)

        # Weighted score
        weighted_score = (val_stoi + (val_pesq / 4.5) + (val_sisdr / 30.0)) / 3.0

        # ===== Collect Audio Samples for Logging =====
        if len(self.val_audio_samples) < self.max_audio_samples:
            # Collect first sample from batch
            self.val_audio_samples.append({
                'noisy': noisy_wav[0].detach().cpu(),
                'clean': clean_wav[0].detach().cpu(),
                'denoised': enhanced[0].detach().cpu()
            })

        # Logging
        self.log('val_loss', total_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log('val/loss', total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/loss_recon', loss_recon, on_epoch=True, sync_dist=True)
        self.log('val/loss_latent', loss_latent, on_epoch=True, sync_dist=True)
        self.log('val/pesq', val_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/stoi', val_stoi, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/si_sdr', val_sisdr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/weighted_score', weighted_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        # ===== Log Audio Samples =====
        if len(self.val_audio_samples) > 0:
            # Use the sample rate saved during initialization
            sr = self.sample_rate

            # Handle both single logger and multiple loggers (list)
            loggers = self.logger if isinstance(self.logger, list) else [self.logger] if self.logger else []

            for idx, sample in enumerate(self.val_audio_samples):
                # Iterate over all loggers
                for logger in loggers:
                    if logger is None:
                        continue

                    try:
                        # TensorBoard logger
                        if hasattr(logger.experiment, 'add_audio'):
                            logger.experiment.add_audio(
                                f'audio/sample_{idx}_noisy',
                                sample['noisy'],
                                self.current_epoch,
                                sample_rate=sr
                            )
                            logger.experiment.add_audio(
                                f'audio/sample_{idx}_clean',
                                sample['clean'],
                                self.current_epoch,
                                sample_rate=sr
                            )
                            logger.experiment.add_audio(
                                f'audio/sample_{idx}_denoised',
                                sample['denoised'],
                                self.current_epoch,
                                sample_rate=sr
                            )

                        # WandB logger
                        try:
                            import wandb
                            if isinstance(logger.experiment, wandb.sdk.wandb_run.Run):
                                logger.experiment.log({
                                    f'audio/sample_{idx}_noisy': wandb.Audio(
                                        sample['noisy'].numpy(), sample_rate=sr, caption=f'Noisy {idx}'
                                    ),
                                    f'audio/sample_{idx}_clean': wandb.Audio(
                                        sample['clean'].numpy(), sample_rate=sr, caption=f'Clean {idx}'
                                    ),
                                    f'audio/sample_{idx}_denoised': wandb.Audio(
                                        sample['denoised'].numpy(), sample_rate=sr, caption=f'Denoised {idx}'
                                    )
                                })
                        except (ImportError, AttributeError):
                            pass  # WandB not available

                    except Exception as e:
                        print(f"[Stage-2] Warning: Could not log audio sample {idx}: {e}")

            print(f"[Stage-2] Logged {len(self.val_audio_samples)} audio samples")

            # Clear samples for next epoch
            self.val_audio_samples = []

    def configure_optimizers(self):
        optimizer_cfg = self.config.get('optimizer', {})
        lr = float(optimizer_cfg.get('lr', optimizer_cfg.get('learning_rate', 1e-4)))
        betas = optimizer_cfg.get('betas', [0.9, 0.999])

        # All parameters are trainable in Stage-2 (no frozen X-Vector extractor)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)

        print(f"[Stage-2] Optimizer: AdamW(lr={lr}, betas={betas})")

        return optimizer

