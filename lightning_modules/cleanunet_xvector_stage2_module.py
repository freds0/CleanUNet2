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

        self.model = CleanUNet2WithXVector(
            stage='stage2',
            use_xvector=True,  # Architecture uses it, but no extractor
            xvector_dim=model_config.get('xvector_dim', 512),
            conditioning_type=model_config.get('conditioning_type', 'addition'),
            cleanunet_params=model_config.get('cleanunet_params', {}),
            cleanspecnet_params=model_config.get('cleanspecnet_params', {}),
            xvector_local_path=model_config.get('xvector_local_path', None)
        )

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

        # ===== Load Stored Latents from Stage-1 (LAZY LOADING) =====
        self.latents_dir = Path(config.get('latents_dir', 'stored_latents_stage1'))

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {self.latents_dir}. "
                           "Please run Stage-1 training first.")

        # LAZY LOADING: Store apenas os caminhos dos arquivos, não carrega na RAM
        self.latent_file_map = self._build_latent_file_map()
        print(f"[Stage-2] Found {len(self.latent_file_map)} latent files from Stage-1 (lazy loading)")

        # Cache para manter latents recentemente usados (LRU cache)
        from collections import OrderedDict
        self.latent_cache = OrderedDict()
        self.latent_cache_size = 100  # Manter apenas 100 latents em cache (~1GB)

        # Validation batch counter
        self.global_val_batch_idx = 0

        # Audio samples for logging (6 samples: noisy, clean, denoised)
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _build_latent_file_map(self):
        """Build a map of batch_idx -> file_path (lazy loading, sem carregar na RAM)."""
        latent_files = sorted(self.latents_dir.glob('val_batch_*.pt'))

        if not latent_files:
            raise ValueError(f"No latent files found in {self.latents_dir}. "
                           "Please run Stage-1 training first.")

        file_map = {}

        for latent_file in latent_files:
            # Extrair batch_idx do nome do arquivo (val_batch_000123.pt)
            filename = latent_file.stem  # val_batch_000123
            idx = int(filename.split('_')[-1])
            file_map[idx] = latent_file

        return file_map

    def _get_latent(self, batch_idx):
        """
        Carrega um latent específico sob demanda (lazy loading com cache LRU).

        Args:
            batch_idx (int): Índice do batch

        Returns:
            dict: Dados do latent ou None se não encontrado
        """
        # Verificar se está no cache
        if batch_idx in self.latent_cache:
            # Mover para o final (MRU - most recently used)
            self.latent_cache.move_to_end(batch_idx)
            return self.latent_cache[batch_idx]

        # Verificar se o arquivo existe
        if batch_idx not in self.latent_file_map:
            return None

        # Carregar do disco
        try:
            latent_file = self.latent_file_map[batch_idx]
            data = torch.load(latent_file, map_location='cpu')

            # Adicionar ao cache
            self.latent_cache[batch_idx] = data

            # Limitar tamanho do cache (remover o mais antigo)
            if len(self.latent_cache) > self.latent_cache_size:
                self.latent_cache.popitem(last=False)  # Remove o primeiro (LRU)

            return data

        except Exception as e:
            print(f"[WARNING] Failed to load latent {batch_idx} from {latent_file}: {e}")
            return None

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

        # ===== Compute Latent Replication Loss (com lazy loading) =====
        stored_data = self._get_latent(self.global_val_batch_idx)

        if stored_data is not None:
            stored_latent = stored_data['fused_latent'].to(self.device)

            # Verificar se os shapes são compatíveis (batch size pode ser diferente)
            pred_batch = predicted_latent.shape[0]
            stored_batch = stored_latent.shape[0]

            if pred_batch != stored_batch:
                # Truncar para o menor batch size
                min_batch = min(pred_batch, stored_batch)
                predicted_latent_slice = predicted_latent[:min_batch]
                stored_latent_slice = stored_latent[:min_batch]

                if self.global_val_batch_idx == 0:  # Avisar apenas na primeira iteração
                    print(f"[WARNING] Batch size mismatch: predicted={pred_batch}, stored={stored_batch}")
                    print(f"[WARNING] Using first {min_batch} samples for latent loss")

                loss_latent = F.mse_loss(predicted_latent_slice, stored_latent_slice)
            else:
                # Shapes compatíveis, calcular loss normalmente
                loss_latent = F.mse_loss(predicted_latent, stored_latent)
        else:
            loss_latent = torch.tensor(0.0, device=self.device)
            if self.global_val_batch_idx < 10:  # Apenas avisa nas primeiras iterações
                print(f"[WARNING] No stored latent found for batch {self.global_val_batch_idx}")

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
                except Exception:
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

        # DESCONGELAR TODOS OS MÓDULOS (Stage-2 não usa X-Vector extractor)
        print("[Stage-2] Descongelando todos os módulos...")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"  - Descongelando: {name}")
            param.requires_grad = True

        # Todos os parâmetros são treináveis em Stage-2
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)

        # Estatísticas de parâmetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"[Stage-2] Optimizer: AdamW(lr={lr}, betas={betas})")
        print(f"[Stage-2] Total params: {total_params:,}")
        print(f"[Stage-2] Trainable params: {trainable_params_count:,}")

        # ===== Learning Rate Scheduler (Optional) =====
        scheduler_cfg = self.config.get('lr_scheduler', {})
        if scheduler_cfg:
            scheduler_type = scheduler_cfg.get('type', None)

            if scheduler_type == 'cosine_annealing_warm_restarts':
                params = scheduler_cfg.get('cosine_annealing_warm_restarts', {})
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=params.get('T_0', 50),
                    T_mult=params.get('T_mult', 2),
                    eta_min=params.get('eta_min', 1e-7)
                )
                print(f"[Stage-2] LR Scheduler: CosineAnnealingWarmRestarts(T_0={params.get('T_0', 50)}, T_mult={params.get('T_mult', 2)}, eta_min={params.get('eta_min', 1e-7)})")
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }

            elif scheduler_type == 'reduce_on_plateau':
                params = scheduler_cfg.get('reduce_on_plateau', {})
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=params.get('mode', 'min'),
                    factor=params.get('factor', 0.5),
                    patience=params.get('patience', 10),
                    min_lr=params.get('min_lr', 1e-7)
                )
                print(f"[Stage-2] LR Scheduler: ReduceLROnPlateau(mode={params.get('mode', 'min')}, factor={params.get('factor', 0.5)}, patience={params.get('patience', 10)})")
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': params.get('monitor', 'val_loss'),
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }

            elif scheduler_type == 'exponential':
                params = scheduler_cfg.get('exponential', {})
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=params.get('gamma', 0.995)
                )
                print(f"[Stage-2] LR Scheduler: ExponentialLR(gamma={params.get('gamma', 0.995)})")
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }

            elif scheduler_type == 'cosine_annealing':
                params = scheduler_cfg.get('cosine_annealing', {})
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=params.get('T_max', 1000),
                    eta_min=params.get('eta_min', 1e-7)
                )
                print(f"[Stage-2] LR Scheduler: CosineAnnealingLR(T_max={params.get('T_max', 1000)}, eta_min={params.get('eta_min', 1e-7)})")
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            else:
                print(f"[Stage-2] Warning: Unknown scheduler type '{scheduler_type}'. Using optimizer without scheduler.")

        return optimizer

