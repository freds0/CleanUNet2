import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

# Import TorchMetrics for Audio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from cleanunet.cleanunet2 import CleanUNet2
from losses import AntiWrappingPhaseLoss, MultiResolutionSTFTLoss, CleanUNet2Loss
from metrics import ObjectiveMetricsPredictor


class CleanUNetLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training CleanUNet2.

    - Combines waveform-domain loss (CleanUNet2Loss), a log-magnitude spectrogram loss,
      and an anti-wrapping phase loss.
    - Computes audio quality metrics (PESQ/STOI/SI-SDR) on validation using TorchMetrics.
    - Provides safe loading of submodule checkpoints and flexible freezing.
    """

    def __init__(self, hparams, freeze_cleanunet: bool = False, freeze_cleanspecnet: bool = True):
        super().__init__()
        self.save_hyperparameters(hparams)

        # ------------------------------------------------------------------
        # 1. Initialize Model
        # ------------------------------------------------------------------
        # Retrieve conditioning type from hparams (default to 'addition')
        conditioning_type = getattr(self.hparams, "conditioning_type", "addition")
        print(f"[INFO] Initializing CleanUNet2 with conditioning: {conditioning_type}")
        
        self.model = CleanUNet2(conditioning_type=conditioning_type)

        # ------------------------------------------------------------------
        # 2. Load Checkpoints (Dynamic from Config)
        # ------------------------------------------------------------------
        # Retrieve paths from config hparams. If key missing or empty, returns None.
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

        # ------------------------------------------------------------------
        # 3. Freezing / Unfreezing
        # ------------------------------------------------------------------
        # Freeze or unfreeze submodules according to flags passed to init.
        # (You can also change this to read from hparams if preferred)

        # [STEP 3.1] RESET: Unfreeze ALL weights first.
        # This ensures the model starts in a fully trainable state before we apply restrictions.
        print("[INFO] Resetting all model parameters to 'Trainable' before applying config.")
        for param in self.model.parameters():
            param.requires_grad = True

        # [STEP 3.2] Apply specific freezing based on configuration.
        # Determine if submodules should be trained or frozen based on config.
        # Default: False (Frozen) if not specified in config.
        train_cleanunet = getattr(self.hparams, "train_cleanunet", False)
        train_cleanspecnet = getattr(self.hparams, "train_cleanspecnet", False)

        print(f"[INFO] Training Config -> CleanUNet: {'TRAIN' if train_cleanunet else 'FREEZE'}, "
              f"CleanSpecNet: {'TRAIN' if train_cleanspecnet else 'FREEZE'}")

        # Apply gradients setting (if False, it will overwrite the True we set above)
        self._set_requires_grad(self.model.clean_unet, train_cleanunet)
        self._set_requires_grad(self.model.clean_spec_net, train_cleanspecnet)

        # [STEP 3.3] Ensure 'glue' components are ALWAYS trainable.
        # Even if the submodules are frozen, the layers connecting them must learn.
        if hasattr(self.model, "conditioner"):
            self._set_requires_grad(self.model.conditioner, True)
        elif hasattr(self.model, "WaveformConditioner"): # Fallback for legacy naming
            self._set_requires_grad(self.model.WaveformConditioner, True)
            
        if hasattr(self.model, "spec_upsampler"):
            self._set_requires_grad(self.model.spec_upsampler, True)

        # ------------------------------------------------------------------
        # 4. Losses
        # ------------------------------------------------------------------
        self.criterion = CleanUNet2Loss(
            ell_p=1,
            ell_p_lambda=1.0,
            stft_lambda=1.0,
            mrstftloss=MultiResolutionSTFTLoss()
        )

        self.phase_loss = AntiWrappingPhaseLoss(n_fft=1024, hop_length=256, win_length=1024)
        self.mrstft = MultiResolutionSTFTLoss()

        # ------------------------------------------------------------------
        # 5. Metrics (TorchMetrics)
        # ------------------------------------------------------------------
        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
        self.val_sisdr = ScaleInvariantSignalNoiseRatio()

        # Legacy Helper (kept unused for validation now)
        self.obj_metrics = ObjectiveMetricsPredictor()

        # Loss combination weights
        self.weight_waveform = float(getattr(self.hparams, "weight_waveform", 10.0))
        self.weight_spec = float(getattr(self.hparams, "weight_spec", 1.0))
        self.weight_phase = float(getattr(self.hparams, "weight_phase", 1.0))
        # self.weight_consistency = 1.0 

    # -------------------------
    # Helpers
    # -------------------------
    def _set_requires_grad(self, module: nn.Module, requires: bool):
        """Utility to set requires_grad for all parameters in a module safely."""
        if module is None:
            return
        status = "Trainable" if requires else "Frozen"
        # print(f"[INFO] Module {type(module).__name__} is set to: {status}")
        for p in module.parameters():
            p.requires_grad = requires

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor):
        """Forward pass delegated to the underlying CleanUNet2 model."""
        return self.model(waveform, spectrogram)

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)

        # 1. Waveform-domain loss
        loss_waveform = self.criterion(clean, enhanced)

        # 2. Spectrogram branch loss (Log1p for stability)
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        # 3. Phase-aware loss
        loss_phase = self.phase_loss(enhanced, clean)

        # Weighted sum of components
        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase)
                
        # Logging
        self.log("train/waveform_loss", loss_waveform, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/spec_loss", loss_spec, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/phase_loss", loss_phase, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch, batch_idx):
        """
        Validation step:
         - Calculates losses.
         - Calculates objective metrics (PESQ, STOI, SI-SDR) using TorchMetrics safely.
         - Computes the custom 'weighted_score'.
         - Logs everything.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)

        # --- 1. Calculate Losses ---
        loss_waveform = self.criterion(clean, enhanced)
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )
        loss_phase = self.phase_loss(enhanced, clean)

        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase)
        
        # --- 2. Calculate Metrics (Safe Mode) ---
        # Note: Input shape to metrics should be (Batch, Time). Squeeze channels.
        preds = enhanced.squeeze(1)
        target = clean.squeeze(1)

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
                val_pesq = self.val_pesq(preds, target)
            except Exception as e:
                # print(f"[WARNING] PESQ computation failed: {e}")
                val_pesq = torch.tensor(1.0, device=self.device)

            # STOI calculation
            try:
                val_stoi = self.val_stoi(preds, target)
            except Exception:
                val_stoi = torch.tensor(1e-5, device=self.device)

            # SI-SDR calculation
            try:
                val_sisdr = self.val_sisdr(preds, target)
            except Exception:
                val_sisdr = torch.tensor(-50.0, device=self.device)

        # --- 3. Calculate Custom Weighted Score ---
        # Formula: (STOI + PESQ/4.5 + SI_SDR/30.0) / 3.0
        # We ensure values are on the correct device for logging
        weighted_score = (val_stoi + (val_pesq / 4.5) + (val_sisdr / 30.0)) / 3.0

        # --- 4. Logging ---
        # Log 'val_loss' explicitly for ModelCheckpoint
        self.log("val_loss", total_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Log Metrics (on_epoch=True ensures accumulation and averaging)
        self.log("val/pesq", val_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/stoi", val_stoi, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/si_sdr", val_sisdr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/weighted_score", weighted_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- 5. Log Audio Examples (First batch only) ---
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            num_examples = min(4, batch_idx) if batch_idx > 4 else 4
            num_examples = min(num_examples, clean.shape[0])
            
            sample_rate = int(getattr(self.hparams, "sample_rate", 16000))
            for i in range(num_examples):
                try:
                    tb.add_audio(f"val/sample_{i}/noisy", noisy[i].squeeze().cpu().unsqueeze(0), global_step=self.global_step, sample_rate=sample_rate)
                    tb.add_audio(f"val/sample_{i}/enhanced", enhanced[i].squeeze().cpu().unsqueeze(0), global_step=self.global_step, sample_rate=sample_rate)
                    tb.add_audio(f"val/sample_{i}/clean", clean[i].squeeze().cpu().unsqueeze(0), global_step=self.global_step, sample_rate=sample_rate)
                except Exception as e:
                    print(f"[WARNING] Failed to add audio to TensorBoard for index {i}: {e}")

        return total_loss

    # -------------------------
    # Optimizers
    # -------------------------
    def configure_optimizers(self):
        lr = float(getattr(self.hparams, "lr", 1e-4))
        # Filter parameters to only update those with requires_grad=True
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        return optimizer