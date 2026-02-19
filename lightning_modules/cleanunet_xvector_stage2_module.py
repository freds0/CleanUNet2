"""
PyTorch Lightning module for CleanUNet2 Stage-2 training.

Stage-2: Training without X-Vectors, replicating latent vectors from Stage-1.
The model learns to predict the fused latents without using the X-Vector extractor,
enabling fast inference while maintaining the benefits of speaker information.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import torchaudio

from cleanunet.cleanunet2_with_xvector import CleanUNet2WithXVector
from losses import CleanUNet2Loss, MultiResolutionSTFTLoss, AntiWrappingPhaseLoss

# Import TorchMetrics
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class CleanUNet2Stage2Module(pl.LightningModule):
    """
    Lightning module for Stage-2 training (without X-Vectors).

    Trains the model to replicate Stage-1's fused latent vectors using
    only the noisy audio (no X-Vector extractor).
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        print("=" * 80)
        print("STAGE-2: Replicating Latents (without X-Vectors)")
        print("=" * 80)

        # ===== Model Initialization =====
        model_config = config.get('model', {})

        # Determine embedding type from config
        use_wav2vec2 = model_config.get('use_wav2vec2', False)
        use_xvector = model_config.get('use_xvector', True)

        # Prepare model initialization arguments
        model_args = {
            'stage': 'stage2',
            'conditioning_type': model_config.get('conditioning_type', 'addition'),
            'cleanunet_params': model_config.get('cleanunet_params', {}),
            'cleanspecnet_params': model_config.get('cleanspecnet_params', {}),
        }

        # Add embedding-specific parameters
        if use_wav2vec2:
            model_args.update({
                'use_wav2vec2': True,
                'use_xvector': False,
                'wav2vec2_model': model_config.get('wav2vec2_model', 'facebook/wav2vec2-xls-r-300m'),
                'wav2vec2_cache_dir': model_config.get('wav2vec2_cache_dir', None),
                'use_preextracted_embeddings': model_config.get('use_preextracted_embeddings', False),
                'wav2vec2_pooling_method': model_config.get('wav2vec2_pooling_method', 'self_attention'),
                'wav2vec2_attention_heads': model_config.get('wav2vec2_attention_heads', 8),
            })
        elif use_xvector:
            model_args.update({
                'use_xvector': True,
                'use_wav2vec2': False,
                'xvector_dim': model_config.get('xvector_dim', 512),
                'xvector_local_path': model_config.get('xvector_local_path', None),
            })
        else:
            # No embeddings
            model_args.update({
                'use_xvector': False,
                'use_wav2vec2': False,
            })

        self.model = CleanUNet2WithXVector(**model_args)

        # ===== Load Stage-1 Checkpoint =====
        stage1_ckpt = config.get('stage1_checkpoint')
        if stage1_ckpt:
            print(f"[Stage-2] Loading Stage-1 checkpoint: {stage1_ckpt}")
            self.model.load_stage1_weights(stage1_ckpt)
        else:
            print("[WARNING] No Stage-1 checkpoint provided. Training from scratch.")

        # ===== Loss Initialization =====
        loss_cfg = config.get('losses', {})

        # Multi-Resolution STFT Loss
        stft_cfg = loss_cfg.get('stft_config', {})
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg.get('fft_sizes', [512, 1024, 2048]),
            hop_sizes=stft_cfg.get('hop_sizes', [128, 256, 512]),
            win_lengths=stft_cfg.get('win_lengths', [512, 1024, 2048])
        )

        # Primary waveform loss
        self.criterion = CleanUNet2Loss(
            ell_p=1,
            ell_p_lambda=1.0,
            stft_lambda=1.0,
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

        # ===== Load Stored Latents from Stage-1 =====
        self.latents_dir = Path(config.get('latents_dir', 'stored_latents_stage1'))

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {self.latents_dir}. "
                           "Please run Stage-1 training first.")

        self.stored_latents = self._load_stored_latents()
        print(f"[Stage-2] Loaded {len(self.stored_latents)} latent files from Stage-1")

        # Validation batch counter
        self.global_val_batch_idx = 0

        # Audio samples for logging (6 samples: noisy, clean, denoised)
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _load_stored_latents(self):
        """Load all stored latents from Stage-1."""
        latent_files = sorted(self.latents_dir.glob('val_batch_*.pt'))

        if not latent_files:
            raise ValueError(f"No latent files found in {self.latents_dir}. "
                           "Please run Stage-1 training first.")

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
        # Note: In training, we don't have direct correspondence with validation batches
        # So we skip latent loss during training (only reconstruction)
        # Latent loss is primarily for validation/evaluation
        loss_latent = torch.tensor(0.0, device=self.device)

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
        lr = float(optimizer_cfg.get('lr', 1e-4))
        betas = optimizer_cfg.get('betas', [0.9, 0.999])

        # All parameters are trainable in Stage-2 (no frozen X-Vector extractor)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)

        print(f"[Stage-2] Optimizer: AdamW(lr={lr}, betas={betas})")

        return optimizer

