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
from cleanunet.ssl_extractor_factory import ssl_args_from_config
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

        # ===== Load Stored VAL Latents from Stage-1 (optional) =====
        # These align by validation batch index and feed the validation-time latent
        # metric. Optional: the training-time latent loss uses the per-file train cache
        # below, so a missing val cache only disables the val metric.
        self.latents_dir = Path(config.get('latents_dir', 'stored_latents_stage1'))
        if self.latents_dir.exists():
            self.stored_latents = self._load_stored_latents()
            print(f"[Stage-2] Loaded {len(self.stored_latents)} VAL latent files from Stage-1")
        else:
            print(f"[Stage-2] VAL latents dir not found ({self.latents_dir}); "
                  f"validation latent metric disabled.")
            self.stored_latents = {}

        # ===== Per-file TRAIN latent cache (REQUIRED for the training-time latent loss) =====
        # Generated by generate_latents.py from a trained Stage-1 checkpoint, keyed
        # by MD5 of the clean-audio path. Requires deterministic_crop on the train set so
        # each sample matches the segment its cached target was computed from.
        # Stage-2 trains ONLY if these latents exist (otherwise distillation is impossible).
        train_dir = config.get('train_latents_dir')
        if not train_dir:
            raise ValueError(
                "[Stage-2] 'train_latents_dir' is not set in the config. Stage-2 distillation "
                "requires the Stage-1 TRAIN latents. Generate them with generate_latents.py "
                "and set 'train_latents_dir' in the config."
            )
        self.train_latents_dir = Path(train_dir)
        self._train_latent_mem = {}
        self.train_latent_keys = (
            {p.stem for p in self.train_latents_dir.glob('*.pt')}
            if self.train_latents_dir.exists() else set()
        )
        if not self.train_latent_keys:
            raise FileNotFoundError(
                f"[Stage-2] No train latents found in '{self.train_latents_dir}'. Stage-2 will not "
                f"train without them. Generate first:\n"
                f"  python generate_latents.py --config <stage1-config> "
                f"--checkpoint <stage1.ckpt> --output-dir {self.train_latents_dir}"
            )
        print(f"[Stage-2] Train latent cache: {len(self.train_latent_keys)} files "
              f"in {self.train_latents_dir} (training-time latent loss ENABLED)")

        # Validation batch counter
        self.global_val_batch_idx = 0

        # Audio samples for logging (6 samples: noisy, clean, denoised)
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _load_stored_latents(self):
        """Load stored VAL latents from Stage-1. Optional: returns an empty dict (and
        disables the validation latent metric) if the directory holds no val_batch_*.pt,
        so an empty/leftover latents_dir never blocks Stage-2 training."""
        latent_files = sorted(self.latents_dir.glob('val_batch_*.pt'))

        if not latent_files:
            print(f"[Stage-2] No val_batch_*.pt in {self.latents_dir}; "
                  f"validation latent metric disabled.")
            return {}

        stored_latents = {}

        for latent_file in latent_files:
            try:
                data = torch.load(latent_file, map_location='cpu')
                batch_idx = data.get('batch_idx', None)

                if batch_idx is not None:
                    stored_latents[batch_idx] = data
                else:
                    # Fallback: extract batch_idx from filename
                    filename = latent_file.stem  # val_batch_000123
                    idx = int(filename.split('_')[-1])
                    stored_latents[idx] = data

            except Exception as e:
                print(f"[WARNING] Failed to load {latent_file}: {e}")

        print(f"[Stage-2] Successfully loaded {len(stored_latents)} latent files")
        return stored_latents

    def _get_train_latent(self, clean_path):
        """Lazily load the cached Stage-1 fused latent for a clean-audio path.
        Returns a CPU float tensor (C, T), or None if not cached."""
        key = latent_cache_key(clean_path)
        if key not in self.train_latent_keys:
            return None
        if key not in self._train_latent_mem:
            self._train_latent_mem[key] = torch.load(
                self.train_latents_dir / f"{key}.pt", map_location='cpu'
            ).float()
        return self._train_latent_mem[key]

    def _train_latent_loss(self, predicted_latent, clean_paths):
        """MSE between the predicted latent and the cached Stage-1 fused latents for
        this batch. Returns 0 if the cache is disabled, paths are unavailable, or any
        sample in the batch is not cached (alignment must be complete to be valid)."""
        zero = torch.zeros((), device=self.device)
        if predicted_latent is None or clean_paths is None or not self.train_latent_keys:
            return zero
        targets = []
        for p in clean_paths:
            t = self._get_train_latent(p)
            if t is None:
                return zero
            targets.append(t)
        target = torch.stack(targets).to(device=predicted_latent.device,
                                         dtype=predicted_latent.dtype)
        min_t = min(predicted_latent.size(-1), target.size(-1))
        return F.mse_loss(predicted_latent[..., :min_t], target[..., :min_t])

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
        loss_latent = self._train_latent_loss(predicted_latent, clean_audio_paths)

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
        if self.global_val_batch_idx in self.stored_latents:
            stored_data = self.stored_latents[self.global_val_batch_idx]
            stored_latent = stored_data['fused_latent'].to(self.device)

            # Handle batch size mismatch (e.g., last batch may be smaller)
            batch_size_pred = predicted_latent.size(0)
            batch_size_stored = stored_latent.size(0)

            if batch_size_pred != batch_size_stored:
                # Use the minimum batch size to compare only matching samples
                min_batch_size = min(batch_size_pred, batch_size_stored)
                predicted_latent_slice = predicted_latent[:min_batch_size]
                stored_latent_slice = stored_latent[:min_batch_size]
                loss_latent = F.mse_loss(predicted_latent_slice, stored_latent_slice)
            else:
                # L2 loss between predicted and stored latents
                loss_latent = F.mse_loss(predicted_latent, stored_latent)
        else:
            loss_latent = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = loss_recon + self.gamma_latent * loss_latent

        self.global_val_batch_idx += 1

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

        # Reset counter for next epoch
        self.global_val_batch_idx = 0

    def configure_optimizers(self):
        optimizer_cfg = self.config.get('optimizer', {})
        lr = float(optimizer_cfg.get('lr', optimizer_cfg.get('learning_rate', 1e-4)))
        betas = optimizer_cfg.get('betas', [0.9, 0.999])

        # All parameters are trainable in Stage-2 (no frozen X-Vector extractor)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)

        print(f"[Stage-2] Optimizer: AdamW(lr={lr}, betas={betas})")

        return optimizer

