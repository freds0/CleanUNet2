"""
PyTorch Lightning module for CleanUNet2 Stage-2 training with SSL Embeddings (WavLM).

Stage-2: Training WITHOUT the WavLM extractor, replicating the Stage-1 fused latents
from the noisy audio only (a latent predictor learns to reproduce them). This gives
fast inference while keeping the benefit of the speaker/content information that the
Stage-1 weighted-layer WavLM embeddings provided.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import torchaudio

from cleanunet.cleanunet2_with_ssl_embeddings import CleanUNet2WithSSLEmbeddings
from losses import CleanUNet2Loss, MultiResolutionSTFTLoss, AntiWrappingPhaseLoss

# Import TorchMetrics
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class CleanUNet2SSLEmbeddingsStage2Module(pl.LightningModule):
    """
    Lightning module for Stage-2 training with SSL Embeddings (WavLM).

    Trains the model to replicate Stage-1's fused latent vectors using only the
    noisy audio (no WavLM extractor at inference time).
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        print("=" * 80)
        print("STAGE-2: Replicating Latents (SSL Embeddings - Inference Mode)")
        print("=" * 80)

        # ===== Model Initialization =====
        model_config = config.get('model', {})

        # Must match the Stage-1 fusion_type so the loaded fusion_block weights line up.
        self.fusion_type = model_config.get('fusion_type', 'cross_attention_film')

        self.model = CleanUNet2WithSSLEmbeddings(
            stage='stage2',
            conditioning_type=model_config.get('conditioning_type', 'addition'),
            cleanunet_params=model_config.get('cleanunet_params', {}),
            cleanspecnet_params=model_config.get('cleanspecnet_params', {}),
            wavlm_model=model_config.get('wavlm_model', 'microsoft/wavlm-large'),
            wavlm_layer=model_config.get('wavlm_layer', 12),
            wavlm_cache_dir=model_config.get('wavlm_cache_dir', None),
            use_preextracted_embeddings=model_config.get('use_preextracted_embeddings', False),
            wavlm_pooling_method=model_config.get('wavlm_pooling_method', 'self_attention'),
            wavlm_attention_heads=model_config.get('wavlm_attention_heads', 8),
            wavlm_use_weighted_layers=model_config.get('wavlm_use_weighted_layers', True),
            fusion_type=self.fusion_type,
            acoustic_layers=tuple(model_config.get('acoustic_layers', [1, 8])),
            semantic_layers=tuple(model_config.get('semantic_layers', [17, 24])),
        )

        # ===== Load Stage-1 Checkpoint =====
        stage1_ckpt = config.get('stage1_checkpoint') or config.get('pipeline', {}).get('stage1_checkpoint')
        if stage1_ckpt:
            print(f"[Stage-2] Loading Stage-1 checkpoint: {stage1_ckpt}")
            self.model.load_stage1_weights(stage1_ckpt)
        else:
            print("[WARNING] No Stage-1 checkpoint provided. Training from scratch.")

        # ===== Loss Initialization =====
        loss_cfg = config.get('losses', {})

        stft_cfg = loss_cfg.get('stft_config', {})
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg.get('fft_sizes', [512, 1024, 2048]),
            hop_sizes=stft_cfg.get('hop_sizes', [128, 256, 512]),
            win_lengths=stft_cfg.get('win_lengths', [512, 1024, 2048]),
            sc_lambda=loss_cfg.get('sc_lambda', 0.1),
            mag_lambda=loss_cfg.get('mag_lambda', 0.1)
        )

        self.criterion = CleanUNet2Loss(
            ell_p=loss_cfg.get('ell_p', 1),
            ell_p_lambda=loss_cfg.get('ell_p_lambda', 1.0),
            stft_lambda=loss_cfg.get('stft_lambda', 1.0),
            mrstftloss=mrstft_loss
        )

        self.phase_loss = AntiWrappingPhaseLoss(
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )

        self.weight_waveform = float(loss_cfg.get('weight_waveform', 10.0))
        self.weight_spec = float(loss_cfg.get('weight_spec', 1.0))
        self.weight_phase = float(loss_cfg.get('weight_phase', 1.0))
        self.gamma_latent = float(loss_cfg.get('gamma_latent', 0.05))

        print(f"[Stage-2] Loss weights: waveform={self.weight_waveform}, "
              f"spec={self.weight_spec}, phase={self.weight_phase}, gamma_latent={self.gamma_latent}")

        # ===== Metrics Initialization =====
        sr = config.get('audio', {}).get('sample_rate') or config.get('data', {}).get('sampling_rate', 16000)
        self.sample_rate = sr

        if sr < 8000:
            raise ValueError(
                f"[Stage-2] ERROR: Sample rate {sr} Hz is too low for PESQ metric. "
                f"PESQ requires at least 8000 Hz."
            )
        elif sr in [8000, 16000]:
            self.pesq_sample_rate = sr
        else:
            self.pesq_sample_rate = 16000

        self.val_pesq = PerceptualEvaluationSpeechQuality(fs=self.pesq_sample_rate, mode='wb')
        self.val_stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=False)
        self.val_sisdr = ScaleInvariantSignalNoiseRatio()

        self._pesq_resampler_config = {
            'needed': self.sample_rate != self.pesq_sample_rate,
            'orig_freq': self.sample_rate,
            'new_freq': self.pesq_sample_rate
        }
        self._pesq_resampler_cache = None

        # ===== (Optional) Stored Latents from Stage-1 =====
        # Stage-1 no longer saves latents: the val/loss_latent term never enters
        # training gradients (it is 0 in training_step), so the stored latents do
        # not affect the trained model. If a latents_dir from an older run is
        # present, the val/loss_latent diagnostic is still computed; otherwise it
        # is skipped (the guard in validation_step falls back to loss_latent=0).
        self.latents_dir = Path(config.get('latents_dir', 'stored_latents_stage1'))
        if self.latents_dir.exists() and any(self.latents_dir.glob('val_batch_*.pt')):
            self.stored_latents = self._load_stored_latents()
            print(f"[Stage-2] Loaded {len(self.stored_latents)} latent files from Stage-1")
        else:
            self.stored_latents = {}
            print("[Stage-2] No stored latents found; val/loss_latent diagnostic disabled.")

        self.global_val_batch_idx = 0
        self.val_audio_samples = []
        self.max_audio_samples = 6

    def _load_stored_latents(self):
        latent_files = sorted(self.latents_dir.glob('val_batch_*.pt'))
        if not latent_files:
            raise ValueError(f"No latent files found in {self.latents_dir}. "
                           "Please run Stage-1 training first.")
        stored_latents = {}
        for latent_file in latent_files:
            try:
                data = torch.load(latent_file, map_location='cpu')
                batch_idx = data.get('batch_idx', None)
                if batch_idx is not None:
                    stored_latents[batch_idx] = data
                else:
                    idx = int(latent_file.stem.split('_')[-1])
                    stored_latents[idx] = data
            except Exception as e:
                print(f"[WARNING] Failed to load {latent_file}: {e}")
        return stored_latents

    def _get_pesq_resampler(self):
        if not self._pesq_resampler_config['needed']:
            return None
        if self._pesq_resampler_cache is None:
            self._pesq_resampler_cache = torchaudio.transforms.Resample(
                orig_freq=self._pesq_resampler_config['orig_freq'],
                new_freq=self._pesq_resampler_config['new_freq']
            )
        return self._pesq_resampler_cache

    def load_state_dict(self, state_dict, strict=True):
        keys_to_ignore = [
            '_pesq_resampler_cache.kernel',
            '_pesq_resampler_cache.width',
            '_pesq_resampler_cache',
        ]
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not any(key.startswith(k) for k in keys_to_ignore):
                filtered_state_dict[key] = value
        return super().load_state_dict(filtered_state_dict, strict=strict)

    def forward(self, noisy_wav, noisy_spec):
        return self.model(noisy_wav, noisy_spec, clean_audio=None)

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, _ = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch

        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_audio=None, return_latents=True
        )

        loss_waveform = self.criterion(clean_wav, enhanced)
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )
        loss_phase = self.phase_loss(enhanced, clean_wav)

        loss_recon = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

        # Latent loss is computed in validation (where stored latents align by index).
        loss_latent = torch.tensor(0.0, device=self.device)
        total_loss = loss_recon + self.gamma_latent * loss_latent

        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_recon', loss_recon, on_step=False, on_epoch=True)
        self.log('train/loss_waveform', loss_waveform, on_step=False, on_epoch=True)
        self.log('train/loss_spec', loss_spec, on_step=False, on_epoch=True)
        self.log('train/loss_phase', loss_phase, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            noisy_wav, noisy_spec, clean_wav, clean_spec, _ = batch
        else:
            noisy_wav, noisy_spec, clean_wav, clean_spec = batch

        enhanced, enhanced_spec, latents = self.model(
            noisy_wav, noisy_spec, clean_audio=None, return_latents=True
        )
        predicted_latent = latents['predicted_latent']

        loss_waveform = self.criterion(clean_wav, enhanced)
        loss_spec = F.l1_loss(
            torch.log1p(F.relu(enhanced_spec) * 1000),
            torch.log1p(clean_spec * 1000)
        )
        loss_phase = self.phase_loss(enhanced, clean_wav)

        loss_recon = (self.weight_waveform * loss_waveform +
                     self.weight_spec * loss_spec +
                     self.weight_phase * loss_phase)

        if self.global_val_batch_idx in self.stored_latents:
            stored_latent = self.stored_latents[self.global_val_batch_idx]['fused_latent'].to(self.device)
            min_b = min(predicted_latent.size(0), stored_latent.size(0))
            loss_latent = F.mse_loss(predicted_latent[:min_b], stored_latent[:min_b])
        else:
            loss_latent = torch.tensor(0.0, device=self.device)

        total_loss = loss_recon + self.gamma_latent * loss_latent
        self.global_val_batch_idx += 1

        with torch.amp.autocast(device_type="cuda", enabled=False):
            preds = enhanced.squeeze(1).float()
            target = clean_wav.squeeze(1).float()
            is_silent_or_nan = (preds.abs().max() < 1e-5) or torch.isnan(preds).any()

            if is_silent_or_nan:
                val_pesq = torch.tensor(1.0, device=self.device)
                val_stoi = torch.tensor(1e-5, device=self.device)
                val_sisdr = torch.tensor(-50.0, device=self.device)
            else:
                try:
                    pesq_resampler = self._get_pesq_resampler()
                    if pesq_resampler is not None:
                        preds_pesq = pesq_resampler(preds)
                        target_pesq = pesq_resampler(target)
                    else:
                        preds_pesq = preds
                        target_pesq = target
                    val_pesq = self.val_pesq(target_pesq.cpu(), preds_pesq.cpu())
                except Exception as e:
                    print(f"[WARNING] PESQ computation failed: {e}")
                    val_pesq = torch.tensor(1.0, device=self.device)

                try:
                    val_stoi = self.val_stoi(target, preds)
                except Exception as e:
                    print(f"[WARNING] STOI computation failed: {e}")
                    val_stoi = torch.tensor(1e-5, device=self.device)

                try:
                    val_sisdr = self.val_sisdr(target, preds)
                except Exception as e:
                    print(f"[WARNING] SI-SDR computation failed: {e}")
                    val_sisdr = torch.tensor(-50.0, device=self.device)

        weighted_score = (val_stoi + (val_pesq / 4.5) + (val_sisdr / 30.0)) / 3.0

        if len(self.val_audio_samples) < self.max_audio_samples:
            self.val_audio_samples.append({
                'noisy': noisy_wav[0].detach().cpu(),
                'clean': clean_wav[0].detach().cpu(),
                'denoised': enhanced[0].detach().cpu()
            })

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
        if len(self.val_audio_samples) > 0:
            sr = self.sample_rate
            loggers = self.logger if isinstance(self.logger, list) else [self.logger] if self.logger else []
            for idx, sample in enumerate(self.val_audio_samples):
                for logger in loggers:
                    if logger is None:
                        continue
                    try:
                        if hasattr(logger.experiment, 'add_audio'):
                            logger.experiment.add_audio(f'audio/sample_{idx}_noisy', sample['noisy'], self.current_epoch, sample_rate=sr)
                            logger.experiment.add_audio(f'audio/sample_{idx}_clean', sample['clean'], self.current_epoch, sample_rate=sr)
                            logger.experiment.add_audio(f'audio/sample_{idx}_denoised', sample['denoised'], self.current_epoch, sample_rate=sr)
                        try:
                            import wandb
                            if isinstance(logger.experiment, wandb.sdk.wandb_run.Run):
                                logger.experiment.log({
                                    f'audio/sample_{idx}_noisy': wandb.Audio(sample['noisy'].numpy(), sample_rate=sr, caption=f'Noisy {idx}'),
                                    f'audio/sample_{idx}_clean': wandb.Audio(sample['clean'].numpy(), sample_rate=sr, caption=f'Clean {idx}'),
                                    f'audio/sample_{idx}_denoised': wandb.Audio(sample['denoised'].numpy(), sample_rate=sr, caption=f'Denoised {idx}')
                                })
                        except (ImportError, AttributeError):
                            pass
                    except Exception as e:
                        print(f"[Stage-2] Warning: Could not log audio sample {idx}: {e}")
            self.val_audio_samples = []
        self.global_val_batch_idx = 0

    def configure_optimizers(self):
        optimizer_cfg = self.config.get('optimizer', {})
        lr = float(optimizer_cfg.get('lr', 1e-4))
        betas = optimizer_cfg.get('betas', [0.9, 0.999])
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)
        print(f"[Stage-2] Optimizer: AdamW(lr={lr}, betas={betas})")
        return optimizer
