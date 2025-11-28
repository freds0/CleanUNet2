import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl

# Certifique-se que CleanUNet2, CleanUNet2Loss e MultiResolutionSTFTLoss estão acessíveis
from cleanunet.cleanunet2 import CleanUNet2
from losses import AntiWrappingPhaseLoss, MultiResolutionSTFTLoss, CleanUNet2Loss#, ComplexL1Loss
from metrics import ObjectiveMetricsPredictor


class CleanUNetLightningModule(pl.LightningModule):
    def __init__(self, hparams,
                    freeze_cleanunet: bool = False,
                    freeze_cleanspecnet: bool = True):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = CleanUNet2()

        # Carrega pesos (com try/except para evitar crash se arquivo não existir no teste)
        try:
            self.model.load_cleanunet_weights('checkpoints/cleanunet/last.ckpt')
            self.model.load_cleanspecnet_weights('checkpoints/cleanspecnet/last.ckpt')
        except Exception as e:
            print(f"Aviso: Não foi possível carregar checkpoints: {e}")

        # Configuração de Congelamento (Freezing)
        # Nota: O loop padrão do PyTorch seta requires_grad=True na inicialização.
        # Só precisamos setar False onde queremos congelar.
        
        if freeze_cleanunet:
            print("Congelando os pesos do CleanUNet.")
            for param in self.model.clean_unet.parameters():
                param.requires_grad = False
        else:
             for param in self.model.clean_unet.parameters():
                param.requires_grad = True

        if freeze_cleanspecnet:
            print("Congelando os pesos do CleanSpecNet.")
            for param in self.model.clean_spec_net.parameters():
                param.requires_grad = False
        else:
            for param in self.model.clean_spec_net.parameters():
                param.requires_grad = True

        # Sempre treinar o condicionador e o upsampler se estiver usando CleanUNet2 híbrido
        for param in self.model.WaveformConditioner.parameters():
            param.requires_grad = True
        for param in self.model.spec_upsampler.parameters():
            param.requires_grad = True

        # Loss Principal (Waveform)
        # Se você vai usar MRSTFT externa, pode zerar o stft_lambda aqui para evitar redundância
        self.criterion = CleanUNet2Loss(
            ell_p=1,
            ell_p_lambda=1.0,
            stft_lambda=1.0,
            mrstftloss=MultiResolutionSTFTLoss() # Passado mas não usado se stft_lambda=0
        )
        # 1Instancie a Phase Loss
        self.phase_loss = AntiWrappingPhaseLoss(
            n_fft=1024, 
            hop_length=256, 
            win_length=1024
        )    
        self.obj_metrics = ObjectiveMetricsPredictor()

    def forward(self, waveform, spectrogram):
        return self.model(waveform, spectrogram)

    def training_step(self, batch, batch_idx):
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)
        
        # 1. Loss no Domínio do Tempo (L1 no Waveform)
        # Como zeramos o stft_lambda no self.criterion, isso retorna basicamente L1
        loss_wav = self.criterion(clean, enhanced)

        # 2. Loss no Ramo de Espectrograma (Auxiliar)
        # Comparando Log-Magnitude Spectrograms (MSE)
        #loss_spec = F.mse_loss(enhanced_spec, clean_spec)
        eps = 1e-7
        loss_spec = F.l1_loss(
            torch.log(torch.clamp(enhanced_spec, min=eps)), 
            torch.log(torch.clamp(clean_spec, min=eps))
        )
        # 2. Calcule a Phase Loss
        # Comparando o waveform gerado (UNet) com o waveform limpo
        loss_phase = self.phase_loss(enhanced, clean)        

        # Soma Total
        # Sugestão: Pesos para equilibrar as magnitudes. 
        # MRSTFT costuma ser grande, L1 spec é pequeno.
        loss = (10.0*loss_wav) + loss_spec + loss_phase

        # Correção do nome da variável no log
        self.log("train_audio_loss", loss_wav, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_spec_loss", loss_spec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_phase_loss", loss_phase, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, noisy_spec, clean, clean_spec = batch
        enhanced, enhanced_spec = self(noisy, noisy_spec)
        
        # Cálculo das losses (mesma lógica do treino para consistência)
        loss_wav = self.criterion(clean, enhanced)

        eps = 1e-7
        #loss_spec = F.mse_loss(enhanced_spec, clean_spec)
        loss_spec = F.l1_loss(
            torch.log(torch.clamp(enhanced_spec, min=eps)), 
            torch.log(torch.clamp(clean_spec, min=eps))
        )

        loss_phase = self.phase_loss(enhanced, clean)  

        loss = (10.0*loss_wav) + loss_spec + loss_phase

        # --- Cálculo de Métricas ---
        # (Código de métricas mantido igual, apenas logs ajustados)
        batch_stoi = []
        batch_pesq = []
        batch_sisdr = []
        
        # Executa métricas apenas em parte do batch para acelerar validação se batch for grande
        # Ou mantenha o loop completo se precisão for crítica.
        batch_size = clean.shape[0]
        for i in range(batch_size):
            clean_waveform = clean[i]
            enhanced_waveform = enhanced[i]
            try:
                metrics = self.obj_metrics.predict_metrics(clean_waveform, enhanced_waveform)
                batch_stoi.append(metrics['stoi'])
                batch_pesq.append(metrics['pesq'])
                batch_sisdr.append(metrics['si_sdr'])
            except Exception:
                pass # Ignora falhas pontuais de métrica

        mean_stoi = np.mean(batch_stoi) if batch_stoi else 0
        mean_pesq = np.mean(batch_pesq) if batch_pesq else 0
        mean_sisdr = np.mean(batch_sisdr) if batch_sisdr else 0

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_stoi", mean_stoi, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_pesq", mean_pesq, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_sisdr", mean_sisdr, prog_bar=True, on_epoch=True, sync_dist=True)

        # --- TensorBoard ---
        if batch_idx == 0 and hasattr(self.logger, 'experiment'):
            tensorboard = self.logger.experiment
            num_samples = min(4, batch_size) # Reduzido para 4 para economizar espaço
            for i in range(num_samples):
                tensorboard.add_audio(f"Val_Sample_{i}/Noisy", noisy[i].squeeze().unsqueeze(0).cpu(), self.global_step, 16000)
                tensorboard.add_audio(f"Val_Sample_{i}/Enhanced", enhanced[i].squeeze().unsqueeze(0).cpu(), self.global_step, 16000)
                tensorboard.add_audio(f"Val_Sample_{i}/Clean", clean[i].squeeze().unsqueeze(0).cpu(), self.global_step, 16000)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)