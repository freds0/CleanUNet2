import torch
import pytorch_lightning as pl
import itertools
import torch.nn.functional as F

# Imports do projeto
from cleanunet.cleanunet2 import CleanUNet2
from cleanunet.hifigan_discriminator import HiFiGANDiscriminator
# Importando as losses originais do CleanUNet e as novas do HiFi-GAN
from losses import MultiResolutionSTFTLoss, generator_loss, discriminator_loss, feature_loss

class CleanUNetGANModule(pl.LightningModule):
    """
    LightningModule implementing the GAN training loop using CleanUNet2 as Generator
    and HiFi-GAN Discriminators (MPD + MSD).
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Desativar otimização automática para controle manual da GAN
        self.automatic_optimization = False

        # ---------------------------------------
        # 1. Initialize Generator (CleanUNet2)
        # ---------------------------------------
        conditioning = getattr(self.hparams, "conditioning_type", "addition")
        
        # Recupera os dicionários de parâmetros do config
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
        # Lê os caminhos do config (hparams)
        ckpt_cleanunet = getattr(self.hparams, "cleanunet_checkpoint", None)
        ckpt_cleanspecnet = getattr(self.hparams, "cleanspecnet_checkpoint", None)

        # Carrega CleanUNet (Waveform)
        if ckpt_cleanunet:
            print(f"[INFO] Loading CleanUNet weights from: {ckpt_cleanunet}")
            try:
                self.generator.load_cleanunet_weights(ckpt_cleanunet)
            except Exception as e:
                print(f"[WARNING] Could not load CleanUNet weights: {e}")

        # Carrega CleanSpecNet (Spectrogram)
        if ckpt_cleanspecnet:
            print(f"[INFO] Loading CleanSpecNet weights from: {ckpt_cleanspecnet}")
            try:
                self.generator.load_cleanspecnet_weights(ckpt_cleanspecnet)
            except Exception as e:
                print(f"[WARNING] Could not load CleanSpecNet weights: {e}")

        # ---------------------------------------
        # 1.2 Handle Freezing (Optional)
        # ---------------------------------------
        # Verifica se deve treinar ou congelar os módulos (padrão: True = treinar)
        train_cleanunet = getattr(self.hparams, "train_cleanunet", True)
        train_cleanspecnet = getattr(self.hparams, "train_cleanspecnet", True)

        self._set_requires_grad(self.generator.clean_unet, train_cleanunet)
        self._set_requires_grad(self.generator.clean_spec_net, train_cleanspecnet)
        
        # Sempre treina o 'Conditioner' e o 'Upsampler' que ligam os dois modelos
        if hasattr(self.generator, "conditioner"):
            self._set_requires_grad(self.generator.conditioner, True)
        if hasattr(self.generator, "spec_upsampler"):
            self._set_requires_grad(self.generator.spec_upsampler, True)
        
        # ---------------------------------------
        # 2. Initialize Discriminator (HiFi-GAN)
        # ---------------------------------------
        self.discriminator = HiFiGANDiscriminator()
        
        # ---------------------------------------
        # 3. Initialize Reconstruction Losses
        # ---------------------------------------
        loss_cfg = getattr(self.hparams, "loss_config", {})
        stft_cfg = loss_cfg.get("stft_config", {})
        self.mrstft = MultiResolutionSTFTLoss(**stft_cfg)
        
        # Weights for loss components
        self.lambda_mel = float(loss_cfg.get("lambda_mel", 45.0))
        self.lambda_fm = float(loss_cfg.get("lambda_fm", 2.0))
        self.lambda_adv = float(loss_cfg.get("lambda_adv", 1.0))

    def _set_requires_grad(self, module, requires_grad):
        """Helper to freeze/unfreeze weights safely"""
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
        """
        Performs one step of training for both Discriminator and Generator.
        """
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Unpack batch
        # Assumes data loader returns: (noisy_waveform, noisy_spec, clean_waveform, clean_spec)
        noisy, noisy_spec, clean, clean_spec = batch
        
        # Ensure waveform shapes are [Batch, 1, Time] for 1D convolutions
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

        # ==================================================================
        # PHASE 1: Train Discriminator
        # ==================================================================
        
        # Generate fake audio (detach gradients so we don't update Generator yet)
        with torch.no_grad():
            fake_audio, _ = self.generator(noisy, noisy_spec)

        # Run Discriminators (Real and Fake)
        # Returns lists of outputs and feature maps from both MPD and MSD
        y_d_rs, y_d_gs, _, _ = self.discriminator(clean, fake_audio.detach())
        
        # Calculate Discriminator Loss
        loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
        
        # Update Discriminator
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        
        self.clip_gradients(opt_d, gradient_clip_val=5.0, gradient_clip_algorithm="norm")

        # Log Discriminator Metrics
        self.log("train/loss_d", loss_d, prog_bar=True, logger=True)

        # ==================================================================
        # PHASE 2: Train Generator
        # ==================================================================
        
        # Generate fake audio again (this time keeping gradients)
        fake_audio, fake_spec = self.generator(noisy, noisy_spec)
        
        # Run Discriminators again to get outputs for Generator loss calculation
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(clean, fake_audio)
        
        # A. Adversarial Loss (Generator tries to verify as Real)
        loss_gen_adv, _ = generator_loss(y_d_gs)
        
        # B. Feature Matching Loss (Minimize distance in intermediate layers)
        loss_fm = feature_loss(fmap_rs, fmap_gs)
        
        # C. Mel-Spectrogram Reconstruction Loss
        # HiFi-GAN usually applies L1 loss on the mel-spectrogram
        # We use the MR-STFT loss which includes Spectral Convergence and Log-Mag L1
        loss_mel_sc, loss_mel_mag = self.mrstft(fake_audio.squeeze(1), clean.squeeze(1))
        loss_mel = loss_mel_sc + loss_mel_mag
        
        # Total Generator Loss
        loss_g = (self.lambda_adv * loss_gen_adv) + \
                 (self.lambda_fm * loss_fm) + \
                 (self.lambda_mel * loss_mel)
        
        # Update Generator
        opt_g.zero_grad()
        self.manual_backward(loss_g)

        self.clip_gradients(opt_g, gradient_clip_val=5.0, gradient_clip_algorithm="norm")

        opt_g.step()
        
        # Log Generator Metrics
        self.log("train/loss_g", loss_g, prog_bar=True, logger=True)
        self.log("train/loss_g_adv", loss_gen_adv, prog_bar=True, logger=True)
        self.log("train/loss_g_fm", loss_fm, prog_bar=True, logger=True)
        self.log("train/loss_g_mel", loss_mel, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configure AdamW optimizers for G and D.
        HiFi-GAN uses learning rate decay, but here we start with basic config.
        """
        lr_g = float(getattr(self.hparams, "lr", 1e-4))
        lr_d = float(getattr(self.hparams, "lr", 1e-5))

        b1 = 0.8 
        b2 = 0.99

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr_d, betas=(b1, b2))
        
        # Return two lists: optimizers and schedulers (empty for now)
        return [opt_g, opt_d], []
    
    def validation_step(self, batch, batch_idx):
        """
        Validation loop: calculate Mel Loss to monitor reconstruction quality.
        """
        noisy, noisy_spec, clean, clean_spec = batch
        
        # Ensure shape
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        
        enhanced, _ = self(noisy, noisy_spec)
        
        loss_mel_sc, loss_mel_mag = self.mrstft(enhanced.squeeze(1), clean.squeeze(1))
        total_mel_loss = loss_mel_sc + loss_mel_mag
        
        self.log("val/loss_mel", total_mel_loss, on_epoch=True, prog_bar=True)

        # ------------------------------------------------------------
        # AUDIO LOGGING (Otimizado)
        # ------------------------------------------------------------
        # OTIMIZAÇÃO: Logar áudio consome muito tempo de I/O.
        # Aqui configuramos para logar apenas a cada 10 épocas.
        # Se estiver na época 0, também loga para vermos o estado inicial.
        LOG_AUDIO_EVERY_N_EPOCHS = 10
        should_log_audio = (self.current_epoch % LOG_AUDIO_EVERY_N_EPOCHS == 0)

        if should_log_audio and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_audio"):
            sr = int(getattr(self.hparams, "sampling_rate", 16000))
            
            MAX_SAMPLES_TO_LOG = 8
            current_batch_size = noisy.shape[0]
            
            start_idx = batch_idx * current_batch_size
            
            if start_idx < MAX_SAMPLES_TO_LOG:
                count = min(current_batch_size, MAX_SAMPLES_TO_LOG - start_idx)
                
                for i in range(count):
                    global_idx = start_idx + i
                    
                    # Move para CPU apenas o necessário
                    noisy_audio = noisy[i].squeeze().cpu().float().clamp(-1, 1)
                    clean_audio = clean[i].squeeze().cpu().float().clamp(-1, 1)
                    enhanced_audio = enhanced[i].squeeze().cpu().float().clamp(-1, 1)
                    
                    self.logger.experiment.add_audio(
                        f"val_samples/sample_{global_idx}/noisy", noisy_audio, self.global_step, sample_rate=sr
                    )
                    self.logger.experiment.add_audio(
                        f"val_samples/sample_{global_idx}/clean", clean_audio, self.global_step, sample_rate=sr
                    )
                    self.logger.experiment.add_audio(
                        f"val_samples/sample_{global_idx}/enhanced", enhanced_audio, self.global_step, sample_rate=sr
                    )     