"""
PyTorch Lightning module for CleanUNet2 Stage-1 training with X-Vectors.

Stage-1: Training with X-Vectors injected into latent space.
The model learns to denoise using speaker embeddings as guidance.
Latent vectors are saved for Stage-2 training.
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


class CleanUNet2Stage1Module(pl.LightningModule):
    """
    Lightning module for Stage-1 training (with X-Vectors).

    Trains the model using X-Vector embeddings extracted from clean audio.
    Saves fused latent vectors for Stage-2 training.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        print("=" * 80)
        print("STAGE-1: Training with X-Vectors")
        print("=" * 80)

        # ===== Model Initialization =====
        model_config = config.get('model', {})

        # Determine embedding type (X-Vector or Wav2Vec2)
        use_wav2vec2 = model_config.get('use_wav2vec2', False)
        use_xvector = model_config.get('use_xvector', not use_wav2vec2)  # Default to xvector if wav2vec2 not specified

        # X-Vector cache configuration
        xvector_cache_enabled = model_config.get('xvector_cache_enabled', False)
        xvector_cache_dir = model_config.get('xvector_cache_dir', 'xvector_cache')

        # Wav2Vec2 configuration
        wav2vec2_model = model_config.get('wav2vec2_model', 'facebook/wav2vec2-xls-r-300m')
        wav2vec2_cache_dir = model_config.get('wav2vec2_cache_dir', 'wav2vec2_embeddings')
        use_preextracted_embeddings = model_config.get('use_preextracted_embeddings', False)

        self.model = CleanUNet2WithXVector(
            stage='stage1',
            use_xvector=use_xvector,
            xvector_dim=model_config.get('xvector_dim', 512),
            conditioning_type=model_config.get('conditioning_type', 'addition'),
            cleanunet_params=model_config.get('cleanunet_params', {}),
            cleanspecnet_params=model_config.get('cleanspecnet_params', {}),
            xvector_local_path=model_config.get('xvector_local_path', None),
            xvector_cache_enabled=xvector_cache_enabled,
            xvector_cache_dir=xvector_cache_dir,
            # Wav2Vec2 parameters
            use_wav2vec2=use_wav2vec2,
            wav2vec2_model=wav2vec2_model,
            wav2vec2_cache_dir=wav2vec2_cache_dir,
            use_preextracted_embeddings=use_preextracted_embeddings
        )

        # ===== Load Vanilla Checkpoint (Optional) =====
        vanilla_ckpt = model_config.get('vanilla_checkpoint')
        if vanilla_ckpt:
            print(f"\n[Stage-1] Loading vanilla checkpoint for warm start: {vanilla_ckpt}")
            self.model.load_vanilla_checkpoint(vanilla_ckpt)
        else:
            print("[Stage-1] No vanilla checkpoint specified. Training from scratch.")

        # ===== Loss Initialization =====
        loss_cfg = config.get('losses', {})

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

        print(f"[Stage-1] Loss weights: waveform={self.weight_waveform}, "
              f"spec={self.weight_spec}, phase={self.weight_phase}")

        # ===== Metrics Initialization =====
        sr = config.get('audio', {}).get('sample_rate', 16000)
        self.sample_rate = sr

        # PESQ only supports 8kHz or 16kHz
        if sr < 8000:
            raise ValueError(
                f"[Stage-1] ERROR: Sample rate {sr} Hz is too low for PESQ metric. "
                f"PESQ requires at least 8000 Hz. Please use sample_rate >= 8000 in your config."
            )
        elif sr in [8000, 16000]:
            # Use sample rate directly for PESQ
            self.pesq_sample_rate = sr
            print(f"[Stage-1] Using sample rate {sr} Hz for PESQ metric")
        else:
            # Sample rate > 16000: downsample to 16kHz for PESQ calculation
            self.pesq_sample_rate = 16000
            print(f"[Stage-1] ⚠️  WARNING: Sample rate is {sr} Hz, but PESQ only supports 8kHz/16kHz.")
            print(f"[Stage-1] Audio will be downsampled to {self.pesq_sample_rate} Hz for PESQ calculation.")

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

        # ===== Latents Storage =====
        self.latents_dir = Path(config.get('latents_dir', 'stored_latents_stage1'))
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Stage-1] Latents will be saved to: {self.latents_dir}")

        # Counter for unique batch identification
        self.global_val_batch_idx = 0

        # Audio samples for logging (6 samples: noisy, clean, denoised)
        self.val_audio_samples = []
        self.max_audio_samples = 6

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
            ).to(self.device)  # Move to same device as model
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

    def forward(self, noisy_wav, noisy_spec, clean_wav=None):
        return self.model(noisy_wav, noisy_spec, clean_wav)

    def training_step(self, batch, batch_idx):
        # Unpack batch (may include file paths for caching)
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, clean_paths = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch
            clean_paths = None

        # Forward with X-Vectors (with optional caching)
        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_wav,
            clean_audio_paths=clean_paths,
            return_latents=True
        )

        # Compute losses
        loss_waveform = self.criterion(clean_wav, enhanced)

        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        loss_phase = self.phase_loss(enhanced, clean_wav)

        total_loss = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_waveform', loss_waveform, on_step=False, on_epoch=True)
        self.log('train/loss_spec', loss_spec, on_step=False, on_epoch=True)
        self.log('train/loss_phase', loss_phase, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch (may include file paths for caching)
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, clean_paths = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch
            clean_paths = None

        # Forward with X-Vectors (with optional caching)
        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_wav,
            clean_audio_paths=clean_paths,
            return_latents=True
        )

        # ===== Save Latents for Stage-2 =====
        latent_path = self.latents_dir / f"val_batch_{self.global_val_batch_idx:06d}.pt"
        torch.save({
            'fused_latent': latents['fused_latent'].cpu(),
            'embedding': latents['embedding'].cpu(),  # Works for both xvector and wav2vec2
            'latent': latents['latent'].cpu(),
            'noisy_wav': noisy_wav.cpu(),
            'clean_wav': clean_wav.cpu(),
            'batch_idx': self.global_val_batch_idx
        }, latent_path)

        self.global_val_batch_idx += 1

        # ===== Compute Losses =====
        loss_waveform = self.criterion(clean_wav, enhanced)

        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        loss_phase = self.phase_loss(enhanced, clean_wav)

        total_loss = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

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

                    # Move to CPU for PESQ calculation (PESQ internal weights are on CPU)
                    preds_pesq_cpu = preds_pesq.cpu()
                    target_pesq_cpu = target_pesq.cpu()

                    # CORRECTED: PESQ expects (reference, degraded) order, i.e., (clean, enhanced)
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
        self.log('val/pesq', val_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/stoi', val_stoi, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/si_sdr', val_sisdr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/weighted_score', weighted_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        print(f"\n[Stage-1] Validation epoch ended")
        print(f"[Stage-1] Latents saved to: {self.latents_dir}")
        print(f"[Stage-1] Total latent files: {len(list(self.latents_dir.glob('*.pt')))}")
        print(f"[Stage-1] Global val batch index: {self.global_val_batch_idx}")

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
                        print(f"[Stage-1] Warning: Could not log audio sample {idx}: {e}")

            print(f"[Stage-1] Logged {len(self.val_audio_samples)} audio samples\n")

            # Clear samples for next epoch
            self.val_audio_samples = []

    def configure_optimizers(self):
        optimizer_cfg = self.config.get('optimizer', {})
        lr = float(optimizer_cfg.get('lr', 1e-4))
        betas = optimizer_cfg.get('betas', [0.9, 0.999])

        # Filter trainable parameters (X-Vector extractor is frozen)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=betas)

        print(f"[Stage-1] Optimizer: AdamW(lr={lr}, betas={betas})")

        return optimizer
