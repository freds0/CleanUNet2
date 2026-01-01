"""
PyTorch Lightning module for training CleanUNet2 with X-Vectors
Implements two-stage training strategy
Modified to include TorchMetrics logs (PESQ, STOI, SI-SDR) and Audio Logging (WandB + TensorBoard support)
"""

import torch
import pytorch_lightning as pl
from cleanunet.cleanunet2_xvector import CleanUNet2XVector
# Import both Loss classes
from losses import CleanUNet2Loss, MultiResolutionSTFTLoss

# --- NEW: Import TorchMetrics ---
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

class CleanUNet2XVectorModule(pl.LightningModule):
    def __init__(self, config, stage='stage1'):
        """
        Args:
            config (dict): Configuration dictionary
            stage (str): 'stage1' (training with x-vectors) or 'stage2' (replicating latents)
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.stage = stage
        
        # --- Model Initialization ---
        model_params = config['model'].copy()
        model_params['xvector_stage'] = stage
        
        if 'use_xvector' not in model_params:
            model_params['use_xvector'] = True

        if 'latent_dim' in model_params:
            model_params['tsfm_d_model'] = model_params.pop('latent_dim')
            
        self.model = CleanUNet2XVector(**model_params)

        # Checkpoint Loading Logic
        ckpt_config = config.get('checkpoint_loading', {})
        if ckpt_config.get('load_cleanunet', False):
            ckpt_path = ckpt_config.get('cleanunet_checkpoint')
            if ckpt_path:
                self.model.load_cleanunet_weights(ckpt_path)
            else:
                print("[Warning] 'load_cleanunet' is True but no path provided.")         
        
        # --- Loss Initialization ---
        loss_cfg = config['losses']
        
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=loss_cfg.get('fft_sizes', [512, 1024, 2048]),
            hop_sizes=loss_cfg.get('hop_sizes', [128, 256, 512]),
            win_lengths=loss_cfg.get('win_lengths', [512, 1024, 2048])
        )
        
        weights = loss_cfg.get('weights', {})
        l1_lambda = weights.get('l1', 1.0)
        stft_lambda = weights.get('stft', 1.0)
        
        self.criterion = CleanUNet2Loss(
            ell_p=1,
            ell_p_lambda=l1_lambda,
            stft_lambda=stft_lambda,
            mrstftloss=mrstft_loss
        )

        # --- NEW: Metrics Initialization ---
        # Using 16kHz as default based on typical CleanUNet usage
        sr = config['data'].get('sample_rate', 16000)
        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=sr, mode='wb')
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=False)
        self.val_sisdr = ScaleInvariantSignalNoiseRatio()

    def forward(self, noisy, clean=None):
        return self.model(noisy, clean)

    def training_step(self, batch, batch_idx):
        clean = batch['clean']
        noisy = batch['noisy']
        
        output = self(noisy, clean)
        
        if isinstance(output, tuple):
            denoised = output[0]
        else:
            denoised = output
            
        loss_dict = self.criterion(clean, denoised)
        
        if isinstance(loss_dict, tuple):
            loss = loss_dict[0]
            metrics = loss_dict[1]
        else:
            loss = loss_dict
            metrics = {}
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if 'stft_sc' in metrics:
            self.log('train/stft_sc', metrics['stft_sc'], on_step=False, on_epoch=True)
        if 'reconstruct' in metrics:
            self.log('train/l1', metrics['reconstruct'], on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        clean = batch['clean']
        noisy = batch['noisy']
        
        # Forward pass
        output = self(noisy, clean)
        
        if isinstance(output, tuple):
            denoised = output[0]
        else:
            denoised = output
            
        # Compute Loss
        loss_dict = self.criterion(clean, denoised)
        
        if isinstance(loss_dict, tuple):
            loss = loss_dict[0]
        else:
            loss = loss_dict
        
        # Log validation loss explicitly
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True) # For legacy callbacks

        # --- NEW: Calculate Metrics (Safe Mode) ---
        # Squeeze channels: (Batch, 1, Time) -> (Batch, Time)
        preds = denoised.squeeze(1)
        target = clean.squeeze(1)

        # Check for Silence or NaNs to prevent crashes
        is_silent_or_nan = (preds.abs().max() < 1e-5) or torch.isnan(preds).any()

        if is_silent_or_nan:
            # Assign worst-case values
            val_pesq = torch.tensor(1.0, device=self.device)
            val_stoi = torch.tensor(1e-5, device=self.device)
            val_sisdr = torch.tensor(-50.0, device=self.device)
        else:
            try:
                val_pesq = self.val_pesq(preds, target)
            except Exception:
                val_pesq = torch.tensor(1.0, device=self.device)

            try:
                val_stoi = self.val_stoi(preds, target)
            except Exception:
                val_stoi = torch.tensor(1e-5, device=self.device)

            try:
                val_sisdr = self.val_sisdr(preds, target)
            except Exception:
                val_sisdr = torch.tensor(-50.0, device=self.device)

        # --- Calculate Custom Weighted Score ---
        # Formula from reference: (STOI + PESQ/4.5 + SI_SDR/30.0) / 3.0
        weighted_score = (val_stoi + (val_pesq / 4.5) + (val_sisdr / 30.0)) / 3.0

        # --- Log Metrics ---
        self.log("val/pesq", val_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/stoi", val_stoi, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/si_sdr", val_sisdr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/weighted_score", weighted_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- NEW: Log Audio Examples (First batch only) ---
        if batch_idx == 0:
            # Limit number of examples
            num_examples = min(4, clean.shape[0])
            sample_rate = self.config['data'].get('sample_rate', 16000)
            
            # Prepare data on CPU to avoid device errors
            noisy_cpu = noisy.detach().cpu()
            enhanced_cpu = denoised.detach().cpu()
            clean_cpu = clean.detach().cpu()

            # --- Logic for WANDB ---
            if isinstance(self.logger, WandbLogger):
                columns = ["Sample ID", "Noisy", "Enhanced", "Clean"]
                data = []
                for i in range(num_examples):
                    # wandb.Audio expects numpy arrays
                    row = [
                        i,
                        wandb.Audio(noisy_cpu[i].squeeze().numpy(), sample_rate=sample_rate, caption="Noisy"),
                        wandb.Audio(enhanced_cpu[i].squeeze().numpy(), sample_rate=sample_rate, caption="Enhanced"),
                        wandb.Audio(clean_cpu[i].squeeze().numpy(), sample_rate=sample_rate, caption="Clean")
                    ]
                    data.append(row)
                
                # Log a table with audio players
                # Note: 'commit=False' is often used if you want to log other metrics in the same step, 
                # but 'commit=True' (default) is safer to ensure it appears.
                self.logger.experiment.log({
                    "val/audio_samples": wandb.Table(data=data, columns=columns)
                })

            # --- Logic for TENSORBOARD ---
            # Using checks to support TensorBoardLogger or other loggers with 'experiment.add_audio'
            elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_audio"):
                tb = self.logger.experiment
                for i in range(num_examples):
                    try:
                        # TensorBoard accepts tensors
                        tb.add_audio(f"val/sample_{i}/noisy", noisy_cpu[i].squeeze().unsqueeze(0), 
                                     global_step=self.global_step, sample_rate=sample_rate)
                        tb.add_audio(f"val/sample_{i}/enhanced", enhanced_cpu[i].squeeze().unsqueeze(0), 
                                     global_step=self.global_step, sample_rate=sample_rate)
                        tb.add_audio(f"val/sample_{i}/clean", clean_cpu[i].squeeze().unsqueeze(0), 
                                     global_step=self.global_step, sample_rate=sample_rate)
                    except Exception as e:
                        print(f"[WARNING] Failed to log audio to TensorBoard: {e}")

        return loss

    def configure_optimizers(self):
        optimizer_config = self.config['optimizer']
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas'])
        )
        return optimizer