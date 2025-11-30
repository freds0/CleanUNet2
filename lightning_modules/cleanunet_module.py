import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from cleanunet.cleanunet2 import CleanUNet2
from losses import AntiWrappingPhaseLoss, MultiResolutionSTFTLoss, CleanUNet2Loss
from metrics import ObjectiveMetricsPredictor


class CleanUNetLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training CleanUNet2.

    - Combines waveform-domain loss (CleanUNet2Loss), a log-magnitude spectrogram loss
      and an anti-wrapping phase loss.
    - Computes audio quality metrics (PESQ/STOI/SI-SDR) on validation.
    - Provides safe loading of submodule checkpoints and flexible freezing based on config.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # -------------------------
        # 1. Initialize Model
        # -------------------------
        # Retrieve conditioning type from hparams (default to 'addition')
        conditioning_type = getattr(self.hparams, "conditioning_type", "addition")
        print(f"[INFO] Initializing CleanUNet2 with conditioning: {conditioning_type}")
        
        self.model = CleanUNet2(conditioning_type=conditioning_type)

        # -------------------------
        # 2. Load Checkpoints (Configurable)
        # -------------------------
        # Retrieve paths from config. If None or empty, skip loading.
        ckpt_cleanunet = getattr(self.hparams, "cleanunet_checkpoint", None)
        ckpt_cleanspecnet = getattr(self.hparams, "cleanspecnet_checkpoint", None)

        # Load CleanUNet weights
        if ckpt_cleanunet:
            try:
                self.model.load_cleanunet_weights(ckpt_cleanunet)
                print(f"[INFO] Loaded CleanUNet weights from: {ckpt_cleanunet}")
            except Exception as e:
                print(f"[WARNING] Failed to load CleanUNet weights: {e}")
        else:
            print("[INFO] No CleanUNet checkpoint provided. Initializing with random weights.")

        # Load CleanSpecNet weights
        if ckpt_cleanspecnet:
            try:
                self.model.load_cleanspecnet_weights(ckpt_cleanspecnet)
                print(f"[INFO] Loaded CleanSpecNet weights from: {ckpt_cleanspecnet}")
            except Exception as e:
                print(f"[WARNING] Failed to load CleanSpecNet weights: {e}")
        else:
            print("[INFO] No CleanSpecNet checkpoint provided. Initializing with random weights.")

        # -------------------------
        # 3. Freezing / Unfreezing
        # -------------------------
        # Determine if submodules should be trained or frozen based on config.
        # Default: False (Frozen) if not specified.
        train_cleanunet = getattr(self.hparams, "train_cleanunet", False)
        train_cleanspecnet = getattr(self.hparams, "train_cleanspecnet", False)

        print(f"[INFO] Training Config -> CleanUNet: {'TRAIN' if train_cleanunet else 'FREEZE'}, "
              f"CleanSpecNet: {'TRAIN' if train_cleanspecnet else 'FREEZE'}")

        # Apply gradients setting
        self._set_requires_grad(self.model.clean_unet, train_cleanunet)
        self._set_requires_grad(self.model.clean_spec_net, train_cleanspecnet)

        # IMPORTANT: Always ensure the 'glue' components (Conditioner & Upsampler) are trainable.
        # These modules need to learn how to fuse the representations from the two submodels.
        if hasattr(self.model, "conditioner"):
            self._set_requires_grad(self.model.conditioner, True)
        elif hasattr(self.model, "WaveformConditioner"): # Fallback for older naming
            self._set_requires_grad(self.model.WaveformConditioner, True)
            
        if hasattr(self.model, "spec_upsampler"):
            self._set_requires_grad(self.model.spec_upsampler, True)

        # -------------------------
        # 4. Losses and Metrics
        # -------------------------
        # Primary waveform loss wrapper (may internally include STFT/MRSTFT)
        self.criterion = CleanUNet2Loss(
            ell_p=1,
            ell_p_lambda=1.0,
            stft_lambda=1.0,
            mrstftloss=MultiResolutionSTFTLoss()
        )

        # Anti-wrapping phase loss (weights phase differences by magnitude)
        self.phase_loss = AntiWrappingPhaseLoss(n_fft=1024, hop_length=256, win_length=1024)

        # Optional MR-STFT instance if you want to use it separately
        self.mrstft = MultiResolutionSTFTLoss()

        # Objective metric helper (PESQ/STOI/SI-SDR)
        self.obj_metrics = ObjectiveMetricsPredictor()

        # Loss combination weights (tweak these experimentally via config)
        self.weight_waveform = float(getattr(self.hparams, "weight_waveform", 10.0))
        self.weight_spec = float(getattr(self.hparams, "weight_spec", 1.0))
        self.weight_phase = float(getattr(self.hparams, "weight_phase", 1.0))
        # Weight for consistency loss (can be 0.0 to disable)
        self.weight_consistency = float(getattr(self.hparams, "weight_consistency", 1.0))

    # -------------------------
    # Helpers
    # -------------------------
    def _set_requires_grad(self, module: nn.Module, requires: bool):
        """Utility to set requires_grad for all parameters in a module safely."""
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor):
        """Forward pass delegated to the underlying CleanUNet2 model."""
        return self.model(waveform, spectrogram)

    # -------------------------
    # Training / Validation
    # -------------------------
    def training_step(self, batch, batch_idx):
        """
        Single training step:
          - Run model to get enhanced waveform + enhanced spectrogram.
          - Compute losses: Waveform L1/MRSTFT, Spectrogram Log-L1, Phase Loss.
          - Combine losses and log scalars.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)

        # 1. Waveform-domain loss (CleanUNet2Loss)
        loss_waveform = self.criterion(clean, enhanced)

        # 2. Spectrogram branch loss
        # Use log1p for better numerical stability near zero.
        # Scaling by 1000 moves spectrogram values into a range where log works effectively.
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        # 3. Phase-aware loss
        loss_phase = self.phase_loss(enhanced, clean)

        '''
        # 4. Consistency Loss (Optional but Recommended)
        # Forces CleanSpecNet to predict a spectrogram consistent with the audio generated by CleanUNet.
        spec_from_audio = torch.stft(
            enhanced.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, 
            window=torch.hann_window(1024).to(enhanced.device), return_complex=True
        ).abs()
        
        loss_consistency = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(spec_from_audio * 1000)
        )
        '''
        # Weighted sum of components
        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase) # + \
                     #(self.weight_consistency * loss_consistency)

        # Logging (train step scalars)
        self.log("train/waveform_loss", loss_waveform, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/spec_loss", loss_spec, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/phase_loss", loss_phase, on_step=True, on_epoch=True, prog_bar=False)
        #self.log("train/consistency_loss", loss_consistency, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)

        # Calculate Losses (Same as training for monitoring)
        loss_waveform = self.criterion(clean, enhanced)
        
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )
        
        loss_phase = self.phase_loss(enhanced, clean)

        # Consistency loss can be omitted from val calculation to save compute if desired,
        # but included here for completeness.
        # spec_from_audio calculation omitted for brevity/speed unless needed.
        
        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase)
        
        # Compute objective metrics safely
        batch_stoi, batch_pesq, batch_sisdr = [], [], []
        batch_size = clean.shape[0]
        
        # Limit metrics calculation to a subset if batch is too large to speed up validation
        metrics_limit = min(batch_size, 8) 
        
        for i in range(metrics_limit):
            try:
                clean_wave = clean[i]
                enhanced_wave = enhanced[i]
                metrics = self.obj_metrics.predict_metrics(clean_wave, enhanced_wave)
                batch_stoi.append(metrics.get("stoi", 0.0))
                batch_pesq.append(metrics.get("pesq", 0.0))
                batch_sisdr.append(metrics.get("si_sdr", 0.0))
            except Exception as e:
                # print(f"Warning: metric computation failed for index {i}: {e}")
                pass

        mean_stoi = float(np.mean(batch_stoi)) if batch_stoi else 0.0
        mean_pesq = float(np.mean(batch_pesq)) if batch_pesq else 0.0
        mean_sisdr = float(np.mean(batch_sisdr)) if batch_sisdr else 0.0

        # Log validation scalars
        # Log 'val_loss' explicitly for ModelCheckpoint compatibility
        self.log("val_loss", total_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        self.log("val/stoi", mean_stoi, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/pesq", mean_pesq, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/si_sdr", mean_sisdr, prog_bar=True, on_epoch=True, sync_dist=True)

        # Log a few audio examples to TensorBoard on the first validation batch
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            num_examples = min(4, batch_size)
            sample_rate = int(getattr(self.hparams, "sample_rate", 16000))
            for i in range(num_examples):
                try:
                    tb.add_audio(f"val/sample_{i}/noisy", noisy[i].squeeze().cpu(), global_step=self.global_step, sample_rate=sample_rate)
                    tb.add_audio(f"val/sample_{i}/enhanced", enhanced[i].squeeze().cpu(), global_step=self.global_step, sample_rate=sample_rate)
                    tb.add_audio(f"val/sample_{i}/clean", clean[i].squeeze().cpu(), global_step=self.global_step, sample_rate=sample_rate)
                except Exception as e:
                    print(f"[WARNING] Failed to add audio to TensorBoard for index {i}: {e}")

        return total_loss

    # -------------------------
    # Optimizers
    # -------------------------
    def configure_optimizers(self):
        """
        Configure optimizer. Uses AdamW.
        Only passes parameters that require gradients to the optimizer.
        """
        lr = float(getattr(self.hparams, "lr", 1e-4))
        
        # Filter parameters to only update those with requires_grad=True
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        return optimizer
