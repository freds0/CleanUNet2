import torch
import pytorch_lightning as pl
import itertools
import torch.nn.functional as F

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
        # 1.1 Load Generator Sub-module Checkpoints
        # ---------------------------------------
        ckpt_cleanunet = getattr(self.hparams, "cleanunet_checkpoint", None)
        ckpt_cleanspecnet = getattr(self.hparams, "cleanspecnet_checkpoint", None)

        if ckpt_cleanunet:
            print(f"[INFO] Loading CleanUNet weights from: {ckpt_cleanunet}")
            try:
                self.generator.load_cleanunet_weights(ckpt_cleanunet)
            except Exception as e:
                print(f"[WARNING] Could not load CleanUNet weights: {e}")

        if ckpt_cleanspecnet:
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
        self.val_sisdr = ScaleInvariantSignalNoiseRatio()
        # PESQ and STOI run on CPU (Slow)
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')

    def _set_requires_grad(self, module, requires_grad):
        """Helper to freeze/unfreeze weights safely."""
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad
        status = "Training" if requires_grad else "Frozen"
        print(f"[INFO] Module {type(module).__name__}: {status}")

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
        The execution of metrics (PESQ, STOI, SI-SDR) is controlled by `metrics_interval_epochs`.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
        
        enhanced, _ = self(noisy, noisy_spec)
        
        # 1. GPU Loss (Always computed for monitoring)
        loss_mel_sc, loss_mel_mag = self.mrstft(enhanced.squeeze(1), clean.squeeze(1))
        total_mel_loss = loss_mel_sc + loss_mel_mag
        self.log("val/loss_mel", total_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. Metrics Calculation (Configurable Interval)
        # Default: 5 epochs if not specified
        val_metrics_interval = int(getattr(self.hparams, "val_metrics_interval_epochs", 5))
        
        # Calculate metrics if it's the current epoch OR if it's the last epoch (sanity check)
        should_run_metrics = (self.current_epoch % val_metrics_interval == 0)

        if should_run_metrics:
            try:
                preds = enhanced.squeeze(1)
                target = clean.squeeze(1)

                # SI-SDR (Fast, GPU)
                self.val_sisdr(preds, target)
                self.log("val/si_sdr", self.val_sisdr, on_step=False, on_epoch=True, prog_bar=True)

                # STOI & PESQ (Slow, CPU)
                self.val_stoi(preds, target)
                self.log("val/stoi", self.val_stoi, on_step=False, on_epoch=True, prog_bar=True)
                
                self.val_pesq(preds, target)
                self.log("val/pesq", self.val_pesq, on_step=False, on_epoch=True, prog_bar=True)
            except Exception as e:
                pass

        # 3. Sparse Audio Logging
        LOG_AUDIO_EVERY_N_EPOCHS = 100
        should_log_audio = (self.current_epoch % LOG_AUDIO_EVERY_N_EPOCHS == 0)

        if should_log_audio and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_audio"):
            if batch_idx == 0:
                MAX_AUDIO_LOGS = 4
                sr = int(getattr(self.hparams, "sampling_rate", 16000))
                count = min(noisy.shape[0], MAX_AUDIO_LOGS)
                
                for i in range(count):
                    noisy_audio = noisy[i].squeeze().cpu().float().clamp(-1, 1)
                    clean_audio = clean[i].squeeze().cpu().float().clamp(-1, 1)
                    enhanced_audio = enhanced[i].squeeze().cpu().float().clamp(-1, 1)
                    
                    self.logger.experiment.add_audio(f"val_{i}/noisy", noisy_audio, self.global_step, sr)
                    self.logger.experiment.add_audio(f"val_{i}/clean", clean_audio, self.global_step, sr)
                    self.logger.experiment.add_audio(f"val_{i}/enhanced", enhanced_audio, self.global_step, sr)