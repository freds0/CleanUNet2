"""
PyTorch Lightning module for training CleanUNet2 with X-Vectors
Implements two-stage training strategy
"""

import torch
import pytorch_lightning as pl
from cleanunet.cleanunet2_xvector import CleanUNet2XVector
# Import both Loss classes
from losses import CleanUNet2Loss, MultiResolutionSTFTLoss

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
        
        # 1. Force 'xvector_stage' to match the current training stage
        model_params['xvector_stage'] = stage
        
        # 2. Ensure 'use_xvector' is set correctly
        if 'use_xvector' not in model_params:
            model_params['use_xvector'] = True

        # 3. Map 'latent_dim' from config to 'tsfm_d_model' expected by CleanUNet2XVector
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
        
        # 1. Instantiate MultiResolutionSTFTLoss using config parameters
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=loss_cfg.get('fft_sizes', [512, 1024, 2048]),
            hop_sizes=loss_cfg.get('hop_sizes', [128, 256, 512]),
            win_lengths=loss_cfg.get('win_lengths', [512, 1024, 2048])
        )
        
        # 2. Extract weights
        weights = loss_cfg.get('weights', {})
        l1_lambda = weights.get('l1', 1.0)
        stft_lambda = weights.get('stft', 1.0)
        
        # 3. Instantiate CleanUNet2Loss with required positional arguments
        self.criterion = CleanUNet2Loss(
            ell_p=1,                  # Default to L1 (ell_p=1)
            ell_p_lambda=l1_lambda,
            stft_lambda=stft_lambda,
            mrstftloss=mrstft_loss
        )
        # -----------------------------
        
    def forward(self, noisy, clean=None):
        """
        Forward pass.
        """
        return self.model(noisy, clean)

    def training_step(self, batch, batch_idx):
        clean = batch['clean']
        noisy = batch['noisy']
        
        # Forward pass
        output = self(noisy, clean)
        
        # Handle cases where model returns multiple values (e.g. latent vectors)
        if isinstance(output, tuple):
            denoised = output[0]
        else:
            denoised = output
            
        # Compute Loss
        loss_dict = self.criterion(clean, denoised) # Note: order is (clean, denoised) in your losses.py
        
        # CleanUNet2Loss returns (total_loss, dict_of_sublosses) or just total_loss depending on implementation
        # Based on your losses.py: it returns a tuple (loss, output_dic)
        if isinstance(loss_dict, tuple):
            loss = loss_dict[0]
            metrics = loss_dict[1]
        else:
            loss = loss_dict
            metrics = {}
        
        # Log metrics
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
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate PESQ if available
        try:
            from metrics import pesq_score
            # PESQ usually requires cpu numpy arrays
            current_pesq = pesq_score(denoised, clean, self.config['data']['sample_rate'])
            self.log('val/pesq', current_pesq, on_step=False, on_epoch=True, prog_bar=True)
        except (ImportError, Exception):
            pass

        return loss

    def configure_optimizers(self):
        optimizer_config = self.config['optimizer']
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas'])
        )
        
        return optimizer