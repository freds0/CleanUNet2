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
    - Provides safe loading of submodule checkpoints and flexible freezing.
    """

    def __init__(self, hparams, freeze_cleanunet: bool = False, freeze_cleanspecnet: bool = True):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Model (CleanUNet2 combines spec-net + waveform UNet)
        self.model = CleanUNet2()

        # Try to load pretrained checkpoints if available - fail gracefully with a warning.
        try:
            self.model.load_cleanunet_weights('checkpoints/cleanunet/last.ckpt')
            self.model.load_cleanspecnet_weights('checkpoints/cleanspecnet/last.ckpt')
            print("[INFO] Loaded pretrained checkpoints for CleanUNet and CleanSpecNet.")
        except Exception as e:
            # Keep going if checkpoints are missing or incompatible
            print(f"[WARNING] Could not load checkpoints: {e}")

        # -------------------------
        # Freezing / Unfreezing
        # -------------------------
        # Freeze or unfreeze submodules according to flags. Default: unfreeze.
        self._set_requires_grad(self.model.clean_unet, not freeze_cleanunet)
        self._set_requires_grad(self.model.clean_spec_net, not freeze_cleanspecnet)

        # Ensure optional components (upsampler/conditioner) are trainable by default.
        # CORREÇÃO: Nome atualizado para "conditioner" conforme a arquitetura CleanUNet2 correta
        if hasattr(self.model, "conditioner"):
            self._set_requires_grad(self.model.conditioner, True)
        elif hasattr(self.model, "WaveformConditioner"): # Fallback para nomes antigos
            self._set_requires_grad(self.model.WaveformConditioner, True)
            
        if hasattr(self.model, "spec_upsampler"):
            self._set_requires_grad(self.model.spec_upsampler, True)

        # -------------------------
        # Losses and metrics
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

        # Loss combination weights (tweak these experimentally)
        self.weight_waveform = float(getattr(self.hparams, "weight_waveform", 10.0))
        self.weight_spec = float(getattr(self.hparams, "weight_spec", 1.0))
        self.weight_phase = float(getattr(self.hparams, "weight_phase", 1.0))
        # Peso para a loss de consistência (novo)
        self.weight_consistency = 1.0 

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
         - Compute waveform loss (self.criterion).
         - Compute log-magnitude spectrogram L1 loss on spectrogram branch.
         - Compute anti-wrapping phase loss between enhanced and clean waveforms.
         - Combine losses and log scalars.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)

        # 1. Waveform-domain loss (CleanUNet2Loss)
        loss_waveform = self.criterion(clean, enhanced)

        # 2. Spectrogram branch loss (MELHORIA: Usando log1p para estabilidade e definição)
        # Multiplicamos por 1000 para trazer os valores para uma escala onde o log opera bem.
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )

        # 3. Phase-aware loss
        loss_phase = self.phase_loss(enhanced, clean)

        '''
        # 4. Consistency Loss (NOVO: O Pulo do Gato)
        # Calcula o spec do áudio gerado pela UNet para comparar com a saída da SpecNet
        spec_from_audio = torch.stft(
            enhanced.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, 
            window=torch.hann_window(1024).to(enhanced.device), return_complex=True
        ).abs()
        
        loss_consistency = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(spec_from_audio * 1000)
        )

        # Weighted sum of components
        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase) + \
                     (self.weight_consistency * loss_consistency)
        '''
        # Weighted sum of components
        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase) 
                
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

        # Losses (same as training)
        loss_waveform = self.criterion(clean, enhanced)
        
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )
        
        loss_phase = self.phase_loss(enhanced, clean)

        '''
        spec_from_audio = torch.stft(
            enhanced.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, 
            window=torch.hann_window(1024).to(enhanced.device), return_complex=True
        ).abs()
        

        loss_consistency = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(spec_from_audio * 1000)
        )

        total_loss = (self.weight_waveform * loss_waveform) + \
                     (self.weight_spec * loss_spec) + \
                     (self.weight_phase * loss_phase) + \
                     (self.weight_consistency * loss_consistency)
        '''
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
                # Assuming predict_metrics handles CPU conversion internally or here
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
        # CORREÇÃO: Logar 'val_loss' explicitamente para garantir compatibilidade com ModelCheckpoint padrão
        self.log("val_loss", total_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        self.log("val/stoi", mean_stoi, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/pesq", mean_pesq, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/si_sdr", mean_sisdr, prog_bar=True, on_epoch=True, sync_dist=True)

        # Log audio examples
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
                    print(f"Warning: failed to add audio to TensorBoard for index {i}: {e}")

        return total_loss

    # -------------------------
    # Optimizers
    # -------------------------
    def configure_optimizers(self):
        lr = float(getattr(self.hparams, "lr", 1e-4))
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return optimizer