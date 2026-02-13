import torch
import pytorch_lightning as pl
import itertools
import torch.nn.functional as F
import torchaudio

# Project imports
from cleanunet.cleanunet2 import CleanUNet2
try:
    from cleanunet.hifigan_components import HiFiGANDiscriminatorWrapper as HiFiGANDiscriminator
except ImportError:
    from cleanunet.hifigan_discriminator import HiFiGANDiscriminator

# Import losses
from losses import (
    MultiResolutionSTFTLoss, 
    generator_loss, 
    discriminator_loss, 
    feature_loss, 
    AntiWrappingPhaseLoss
)

# Metrics
from torchmetrics.audio import (
    ScaleInvariantSignalNoiseRatio, 
    ShortTimeObjectiveIntelligibility, 
    PerceptualEvaluationSpeechQuality
)

class CleanUNetGANModule(pl.LightningModule):
    """
    LightningModule implementing the GAN training loop using CleanUNet2 as Generator
    and HiFi-GAN Discriminators (MPD + MSD).
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Disable automatic optimization to handle the GAN training loop manually
        self.automatic_optimization = False

        # ---------------------------------------
        # 1. Initialize Generator (CleanUNet2)
        # ---------------------------------------
        conditioning = getattr(self.hparams, "conditioning_type", "addition")
        
        # Retrieve parameter dictionaries from config
        cleanunet_params = getattr(self.hparams, "cleanunet_params", {})
        cleanspecnet_params = getattr(self.hparams, "cleanspecnet_params", {})

        self.generator = CleanUNet2(
            conditioning_type=conditioning,
            cleanunet_params=cleanunet_params,
            cleanspecnet_params=cleanspecnet_params
        )

        # ---------------------------------------
        # 1.1 Load Complete CleanUNet2 Checkpoint (from Vanilla training)
        # ---------------------------------------
        ckpt_cleanunet2 = getattr(self.hparams, "cleanunet2_checkpoint", None)

        if ckpt_cleanunet2:
            print(f"[INFO] Loading complete CleanUNet2 model from Vanilla checkpoint: {ckpt_cleanunet2}")
            try:
                self._load_cleanunet2_from_vanilla(ckpt_cleanunet2)
                print("[SUCCESS] CleanUNet2 weights loaded successfully from Vanilla checkpoint.")
            except Exception as e:
                print(f"[WARNING] Could not load CleanUNet2 from Vanilla checkpoint: {e}")
                import traceback
                traceback.print_exc()

        # ---------------------------------------
        # 1.2 Load Generator Sub-module Checkpoints (Alternative)
        # ---------------------------------------
        # These are used if you want to load individual components instead of full model
        ckpt_cleanunet = getattr(self.hparams, "cleanunet_checkpoint", None)
        ckpt_cleanspecnet = getattr(self.hparams, "cleanspecnet_checkpoint", None)

        if ckpt_cleanunet and not ckpt_cleanunet2:
            print(f"[INFO] Loading CleanUNet weights from: {ckpt_cleanunet}")
            try:
                self.generator.load_cleanunet_weights(ckpt_cleanunet)
            except Exception as e:
                print(f"[WARNING] Could not load CleanUNet weights: {e}")

        if ckpt_cleanspecnet and not ckpt_cleanunet2:
            print(f"[INFO] Loading CleanSpecNet weights from: {ckpt_cleanspecnet}")
            try:
                self.generator.load_cleanspecnet_weights(ckpt_cleanspecnet)
            except Exception as e:
                print(f"[WARNING] Could not load CleanSpecNet weights: {e}")

        # ---------------------------------------
        # 1.2 Handle Freezing (Optional)
        # ---------------------------------------
        # Check if modules should be trained or frozen (default: True = train)
        train_cleanunet = getattr(self.hparams, "train_cleanunet", True)
        train_cleanspecnet = getattr(self.hparams, "train_cleanspecnet", True)

        self._set_requires_grad(self.generator.clean_unet, train_cleanunet)
        self._set_requires_grad(self.generator.clean_spec_net, train_cleanspecnet)
        
        # Always train 'Conditioner' and 'Upsampler' which connect the two models
        if hasattr(self.generator, "conditioner"):
            self._set_requires_grad(self.generator.conditioner, True)
        if hasattr(self.generator, "spec_upsampler"):
            self._set_requires_grad(self.generator.spec_upsampler, True)
        
        # ---------------------------------------
        # 2. Initialize Discriminator (HiFi-GAN)
        # ---------------------------------------
        self.discriminator = HiFiGANDiscriminator()
        
        # ---------------------------------------
        # 3. Initialize Losses
        # ---------------------------------------
        loss_cfg = getattr(self.hparams, "loss_config", {})
        stft_cfg = loss_cfg.get("stft_config", {})
        
        # 3.1 Main Reconstruction Loss (MR-STFT)
        self.mrstft = MultiResolutionSTFTLoss(**stft_cfg)
        
        # 3.2 Auxiliary Losses
        self.criterion = torch.nn.L1Loss()          # Waveform Loss
        self.phase_loss = AntiWrappingPhaseLoss()   # Phase Loss
        
        # 3.3 Loss Weights
        self.lambda_mel = float(loss_cfg.get("lambda_mel", 45.0))
        self.lambda_fm = float(loss_cfg.get("lambda_fm", 2.0))
        self.lambda_adv = float(loss_cfg.get("lambda_adv", 1.0))

        self.weight_waveform = float(loss_cfg.get("weight_waveform", 1.0))
        self.weight_spec = float(loss_cfg.get("weight_spec", 1.0))
        self.weight_phase = float(loss_cfg.get("weight_phase", 1.0))

        # ---------------------------------------
        # 4. Initialize Validation Metrics
        # ---------------------------------------
        # SI-SDR runs on GPU (Fast)
        # Get sample rate from config
        sr = int(getattr(hparams, "sampling_rate", 16000))
        self.sample_rate = sr

        # PESQ only supports 8kHz or 16kHz
        if sr < 8000:
            raise ValueError(
                f"[GAN] ERROR: Sample rate {sr} Hz is too low for PESQ metric. "
                f"PESQ requires at least 8000 Hz. Please use sampling_rate >= 8000 in your config."
            )
        elif sr in [8000, 16000]:
            # Use sample rate directly for PESQ
            self.pesq_sample_rate = sr
            print(f"[GAN] Using sample rate {sr} Hz for PESQ metric")
        else:
            # Sample rate > 16000: downsample to 16kHz for PESQ calculation
            self.pesq_sample_rate = 16000
            print(f"[GAN] ⚠️  WARNING: Sample rate is {sr} Hz, but PESQ only supports 8kHz/16kHz.")
            print(f"[GAN] Audio will be downsampled to {self.pesq_sample_rate} Hz for PESQ calculation.")

        self.val_sisdr = ScaleInvariantSignalNoiseRatio()
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=False)
        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=self.pesq_sample_rate, mode='wb')

        # Store resampler config (but don't create the resampler itself yet)
        # This avoids saving it in the checkpoint, preventing compatibility issues
        self._pesq_resampler_config = {
            'needed': self.sample_rate != self.pesq_sample_rate,
            'orig_freq': self.sample_rate,
            'new_freq': self.pesq_sample_rate
        }
        self._pesq_resampler_cache = None

        # ---------------------------------------
        # 5. Audio Samples for Logging (6 samples: noisy, clean, denoised)
        # ---------------------------------------
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _set_requires_grad(self, module, requires_grad):
        """Helper to freeze/unfreeze weights safely."""
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad
        status = "Training" if requires_grad else "Frozen"
        print(f"[INFO] Module {type(module).__name__}: {status}")

    def _load_cleanunet2_from_vanilla(self, checkpoint_path):
        """
        Load a complete CleanUNet2 checkpoint from Vanilla (LightningModule) training.

        This method handles:
        - PyTorch Lightning checkpoints (with 'state_dict' key)
        - Direct state_dict files
        - Removal of 'model.' prefix added by Lightning

        Args:
            checkpoint_path: Path to the Vanilla checkpoint file (.ckpt)
        """
        print(f"[INFO] Loading Vanilla checkpoint from: {checkpoint_path}")

        # Load checkpoint to CPU to avoid device compatibility issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict from Lightning checkpoint structure
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"[INFO] Found 'state_dict' key in checkpoint (Lightning format)")
        else:
            state_dict = checkpoint
            print(f"[INFO] Using checkpoint directly as state_dict")

        # Remove 'model.' prefix from keys (Lightning adds this)
        cleanunet2_state_dict = {}
        prefix = "model."

        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Remove 'model.' prefix
                new_key = key[len(prefix):]
                cleanunet2_state_dict[new_key] = value
            else:
                # Keep keys that don't have the prefix (shouldn't happen, but safe)
                cleanunet2_state_dict[key] = value

        if not cleanunet2_state_dict:
            raise ValueError(
                f"No compatible keys found after prefix removal. "
                f"Expected keys starting with '{prefix}'. "
                f"Found keys: {list(state_dict.keys())[:5]}..."
            )

        # Load into generator
        missing_keys, unexpected_keys = self.generator.load_state_dict(cleanunet2_state_dict, strict=False)

        # Report loading status
        if missing_keys:
            print(f"[WARNING] Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys}")

        print(f"[SUCCESS] Loaded {len(cleanunet2_state_dict)} parameters into CleanUNet2 generator")

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

    def forward(self, noisy, noisy_spec):
        """Forward pass of the generator."""
        return self.generator(noisy, noisy_spec)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        noisy, noisy_spec, clean, clean_spec = batch
        
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

        # Retrieve config parameters
        warmup_epochs = int(getattr(self.hparams, "warmup_epochs", 0))
        grad_clip_threshold = float(getattr(self.hparams, "grad_clip_threshold", 5.0))
        
        IS_WARMUP = self.current_epoch < warmup_epochs

        # ==================================================================
        # PHASE 1: Train Discriminator
        # ==================================================================
        if not IS_WARMUP:
            with torch.no_grad():
                fake_audio, _ = self.generator(noisy, noisy_spec)

            y_d_rs, y_d_gs, _, _ = self.discriminator(clean, fake_audio.detach())
            
            loss_d, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)
            
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            self.clip_gradients(opt_d, gradient_clip_val=grad_clip_threshold, gradient_clip_algorithm="norm")
            opt_d.step()
            
            self.log("train/loss_d", loss_d, prog_bar=True, logger=True)
            self.log("train/loss_d_real", sum(r_losses)/len(r_losses), prog_bar=False, logger=True)
            self.log("train/loss_d_fake", sum(g_losses)/len(g_losses), prog_bar=False, logger=True)

        # ==================================================================
        # PHASE 2: Train Generator
        # ==================================================================
        fake_audio, fake_spec = self.generator(noisy, noisy_spec)

        # --- 1. Reconstruction Losses ---
        loss_waveform = self.criterion(fake_audio, clean)
        loss_spec = F.l1_loss(fake_spec, clean_spec)
        loss_phase = self.phase_loss(fake_audio, clean)
        loss_mel_sc, loss_mel_mag = self.mrstft(fake_audio.squeeze(1), clean.squeeze(1))
        loss_mel = loss_mel_sc + loss_mel_mag
        
        loss_g_recon = (self.lambda_mel * loss_mel) + \
                       (self.weight_waveform * loss_waveform) + \
                       (self.weight_spec * loss_spec) + \
                       (self.weight_phase * loss_phase)

        if IS_WARMUP:
            loss_gen_adv = 0.0
            loss_fm = 0.0
            loss_g = loss_g_recon
        else:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(clean, fake_audio)
            loss_gen_adv, _ = generator_loss(y_d_gs)
            loss_fm = feature_loss(fmap_rs, fmap_gs)
            
            loss_g = (self.lambda_adv * loss_gen_adv) + \
                     (self.lambda_fm * loss_fm) + \
                     loss_g_recon
            
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=grad_clip_threshold, gradient_clip_algorithm="norm")
        opt_g.step()
        
        self.log("train/loss_g", loss_g, prog_bar=True, logger=True)
        self.log("train/loss_g_mel", loss_mel, prog_bar=True, logger=True)
        self.log("train/loss_wav", loss_waveform, prog_bar=False, logger=True)
        self.log("train/loss_spec", loss_spec, prog_bar=False, logger=True)
        self.log("train/loss_phase", loss_phase, prog_bar=False, logger=True)
        
        if not IS_WARMUP:
            self.log("train/loss_g_adv", loss_gen_adv, prog_bar=True, logger=True)
            self.log("train/loss_g_fm", loss_fm, prog_bar=True, logger=True)

    # ----------------------------------------------------------------------
    # CRITICAL: Manual Step for Schedulers
    # ----------------------------------------------------------------------
    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        Essential for stepping Learning Rate Schedulers when using Manual Optimization.
        """
        sch_g, sch_d = self.lr_schedulers()

        # Step Generator Scheduler
        if isinstance(sch_g, torch.optim.lr_scheduler.ExponentialLR):
            sch_g.step()
        
        # Step Discriminator Scheduler
        if isinstance(sch_d, torch.optim.lr_scheduler.ExponentialLR):
            sch_d.step()

    def configure_optimizers(self):
        """
        Configure AdamW optimizers and Learning Rate Schedulers.
        """
        lr_g = float(getattr(self.hparams, "lr_g", 2e-4))
        lr_d = float(getattr(self.hparams, "lr_d", 2e-4))
        b1 = float(getattr(self.hparams, "adam_b1", 0.8))
        b2 = float(getattr(self.hparams, "adam_b2", 0.99))
        lr_decay = float(getattr(self.hparams, "lr_decay", 0.999))

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr_d, betas=(b1, b2))

        # Exponential LR Decay
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=lr_decay)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]


    def validation_step(self, batch, batch_idx):
        """
        Validation loop: calculate Mel Loss and metrics.
        Metrics (PESQ, STOI, SI-SDR) are now calculated on every validation step.
        """
        noisy, noisy_spec, clean, clean_spec = batch

        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

        enhanced, _ = self(noisy, noisy_spec)

        # 1. GPU Loss (Always computed for monitoring)
        loss_mel_sc, loss_mel_mag = self.mrstft(enhanced.squeeze(1), clean.squeeze(1))
        total_mel_loss = loss_mel_sc + loss_mel_mag

        # Log val_loss explicitly for ModelCheckpoint
        self.log("val_loss", total_mel_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_mel", total_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. Calculate Metrics (Safe Mode) - Now on every validation step
        # Note: Input shape to metrics should be (Batch, Time). Squeeze channels.
        # Disable autocast for metrics computation to ensure float32 precision
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Convert to float32 for metrics (AMP uses float16, but metrics need float32)
            preds = enhanced.squeeze(1).float()
            target = clean.squeeze(1).float()

            # Check for Silence or NaNs to prevent PESQ crashes (NoUtterancesError)
            # If the max amplitude is too low, PESQ considers it empty.
            is_silent_or_nan = (preds.abs().max() < 1e-5) or torch.isnan(preds).any()

            if is_silent_or_nan:
                # Assign worst-case values if model collapsed
                val_pesq = torch.tensor(1.0, device=self.device)   # Min PESQ is ~1.0
                val_stoi = torch.tensor(1e-5, device=self.device)  # Min STOI is 0.0
                val_sisdr = torch.tensor(-50.0, device=self.device) # Very low SI-SDR
            else:
                # PESQ calculation
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
                    val_pesq = torch.tensor(1.0, device=self.device)

                # STOI calculation
                try:
                    # CORRECTED: STOI expects (reference, degraded) order
                    val_stoi = self.val_stoi(target, preds)
                except Exception as e:
                    print(f"[WARNING] STOI computation failed: {e}")
                    val_stoi = torch.tensor(1e-5, device=self.device)

                # SI-SDR calculation
                try:
                    # CORRECTED: SI-SDR expects (reference, degraded) order
                    val_sisdr = self.val_sisdr(target, preds)
                except Exception as e:
                    print(f"[WARNING] SI-SDR computation failed: {e}")
                    val_sisdr = torch.tensor(-50.0, device=self.device)

        # 3. Calculate Custom Weighted Score
        # Formula: (STOI + PESQ/4.5 + SI_SDR/30.0) / 3.0
        weighted_score = (val_stoi + (val_pesq / 4.5) + (val_sisdr / 30.0)) / 3.0

        # 4. Collect Audio Samples for Logging
        if len(self.val_audio_samples) < self.max_audio_samples:
            # Collect first sample from batch
            self.val_audio_samples.append({
                'noisy': noisy[0].detach().cpu(),
                'clean': clean[0].detach().cpu(),
                'denoised': enhanced[0].detach().cpu()
            })

        # 5. Logging Metrics
        self.log("val/pesq", val_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/stoi", val_stoi, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/si_sdr", val_sisdr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/weighted_score", weighted_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        Logs collected audio samples to TensorBoard and WandB.
        """
        if len(self.val_audio_samples) > 0:
            # Use the sample rate saved during initialization
            sample_rate = self.sample_rate

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
                                sample_rate=sample_rate
                            )
                            logger.experiment.add_audio(
                                f'audio/sample_{idx}_clean',
                                sample['clean'],
                                self.current_epoch,
                                sample_rate=sample_rate
                            )
                            logger.experiment.add_audio(
                                f'audio/sample_{idx}_denoised',
                                sample['denoised'],
                                self.current_epoch,
                                sample_rate=sample_rate
                            )

                        # WandB logger
                        try:
                            import wandb
                            if isinstance(logger.experiment, wandb.sdk.wandb_run.Run):
                                logger.experiment.log({
                                    f'audio/sample_{idx}_noisy': wandb.Audio(
                                        sample['noisy'].numpy(), sample_rate=sample_rate, caption=f'Noisy {idx}'
                                    ),
                                    f'audio/sample_{idx}_clean': wandb.Audio(
                                        sample['clean'].numpy(), sample_rate=sample_rate, caption=f'Clean {idx}'
                                    ),
                                    f'audio/sample_{idx}_denoised': wandb.Audio(
                                        sample['denoised'].numpy(), sample_rate=sample_rate, caption=f'Denoised {idx}'
                                    )
                                })
                        except (ImportError, AttributeError):
                            pass  # WandB not available

                    except Exception as e:
                        print(f"[WARNING] Could not log audio sample {idx}: {e}")

            print(f"[INFO] Logged {len(self.val_audio_samples)} audio samples to TensorBoard/WandB")

            # Clear samples for next epoch
            self.val_audio_samples = []