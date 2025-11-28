from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet

#from cleanspecnet import CleanSpecNet
#from cleanunet import CleanUNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

class SpecUpsampler(nn.Module):
    """
    Upsample spectrogram by factor x256 in time using two ConvTranspose2d layers.
    Input: spec (B, F, T)  -- freq x time magnitude (no channel dim)
    Output: up_spec_as_wave (B, 1, L) -- single-channel time-domain-sized feature
    """
    def __init__(self,
                 in_channels=1,
                 hidden_channels=32,
                 freq_kernel=3,
                 time_kernel=16,
                 leaky_slope=0.4):
        super(SpecUpsampler, self).__init__()
        # We'll perform ConvTranspose2d with kernel_size=(freq_kernel, time_kernel)
        # stride=(1,16) so two stacked layers give factor 16*16 = 256 in time.
        # freq_kernel=3 with padding=1 preserves the freq dimension (approximately).
        # We use padding=(freq_kernel//2, 0) so freq dim preserved; time padding 0 for causal-like behavior.
        self.up1 = nn.ConvTranspose2d(
            in_channels, hidden_channels,
            kernel_size=(freq_kernel, time_kernel),
            stride=(1, time_kernel),
            padding=(freq_kernel // 2, 0),
            output_padding=(0, 0)
        )
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

        self.up2 = nn.ConvTranspose2d(
            hidden_channels, in_channels,
            kernel_size=(freq_kernel, time_kernel),
            stride=(1, time_kernel),
            padding=(freq_kernel // 2, 0),
            output_padding=(0, 0)
        )
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

    def forward(self, spec): 
        # spec: (B, F, T) -> convert to (B, C=1, F, T)
        x = spec.unsqueeze(1)
        x = self.up1(x)   # (B, hidden, F, T*16)
        x = self.act1(x)
        x = self.up2(x)   # (B, 1, F, T*256)  (since 16*16 = 256)
        x = self.act2(x)

        # Collapse frequency axis to produce a single-channel time signal
        # Strategy: average across frequency (simple, stable) -> (B, 1, T_high)
        time_feat = x.mean(dim=2, keepdim=True)  # (B, 1, 1, T_high)
        time_feat = time_feat.squeeze(2)         # (B, 1, T_high)
        return time_feat


class WaveformConditioner(nn.Module):
    def __init__(self, 
                 in_channels: int = 2, 
                 hidden_channels: int = 16, # 1. Adicionado canal intermediário
                 out_channels: int = 1, 
                 kernel_size: int = 7,
                 norm_type: str = 'batch' # 2. Opção para tipo de normalização
                ):
        super(WaveformConditioner, self).__init__()
        
        # Garante que o kernel seja ímpar para o cálculo do padding
        assert kernel_size % 2 == 1, "O tamanho do kernel deve ser ímpar."
        padding = (kernel_size - 1) // 2
        
        # Define o tipo de normalização
        if norm_type.lower() == 'batch':
            norm_layer = nn.BatchNorm1d(hidden_channels)
        elif norm_type.lower() == 'layer':
            # LayerNorm precisa do comprimento da sequência, o que é complicado aqui.
            # BatchNorm é mais comum para áudio em formato (B, C, T).
            raise NotImplementedError("LayerNorm não é ideal para este formato. Use 'batch'.")
        else:
            raise ValueError("norm_type deve ser 'batch'.")

        # 3. Construção de um bloco sequencial mais robusto
        self.conditioner_block = nn.Sequential(
            # Camada 1: Expande para canais intermediários
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=hidden_channels, 
                kernel_size=kernel_size,
                padding=padding
            ),
            norm_layer,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Camada 2: Projeta de volta para o canal de saída
            nn.Conv1d(
                in_channels=hidden_channels, 
                out_channels=out_channels, 
                kernel_size=1 # Uma convolução 1x1 para finalizar
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4. O forward pass agora é mais limpo
        return self.conditioner_block(x) 

class CleanUNet2(nn.Module):
    def __init__(self, 
            cleanunet_input_channels=1,
            cleanunet_output_channels=1,
            cleanunet_channels_H=64,
            cleanunet_max_H=768,
            cleanunet_encoder_n_layers=8,
            cleanunet_kernel_size=4,
            cleanunet_stride=2,
            cleanunet_tsfm_n_layers=5, 
            cleanunet_tsfm_n_head=8,
            cleanunet_tsfm_d_model=512, 
            cleanunet_tsfm_d_inner=2048,
            cleanspecnet_input_channels=513, 
            cleanspecnet_num_conv_layers=5, 
            cleanspecnet_kernel_size=4, 
            cleanspecnet_stride=1,
            cleanspecnet_num_attention_layers=5, 
            cleanspecnet_num_heads=8, 
            cleanspecnet_hidden_dim=512, 
            cleanspecnet_dropout=0.1):

        super(CleanUNet2, self).__init__()
        
        self.clean_unet = CleanUNet(
            channels_input=cleanunet_input_channels, 
            channels_output=cleanunet_output_channels,
            channels_H=cleanunet_channels_H, 
            max_H=cleanunet_max_H,
            encoder_n_layers=cleanunet_encoder_n_layers, 
            kernel_size=cleanunet_kernel_size, 
            stride=cleanunet_stride,
            tsfm_n_layers=cleanunet_tsfm_n_layers,
            tsfm_n_head=cleanunet_tsfm_n_head,
            tsfm_d_model=cleanunet_tsfm_d_model, 
            tsfm_d_inner=cleanunet_tsfm_d_inner
        )        

        self.clean_spec_net = CleanSpecNet(
            input_channels=cleanspecnet_input_channels, 
            num_conv_layers=cleanspecnet_num_conv_layers, 
            kernel_size=cleanspecnet_kernel_size, 
            stride=cleanspecnet_stride, 
            hidden_dim=cleanspecnet_hidden_dim, 
            num_attention_layers=cleanspecnet_num_attention_layers, 
            num_heads=cleanspecnet_num_heads, 
            dropout=cleanspecnet_dropout
        )
        #self.WaveformConditioner = WaveformConditioner()    

        # Instantiate the new upsampler (use defaults or change hidden_channels)
        self.spec_upsampler = SpecUpsampler(in_channels=1, hidden_channels=32,
                                            freq_kernel=3, time_kernel=16,
                                            leaky_slope=0.4)

        self.WaveformConditioner = WaveformConditioner()

    def _reconstruct_waveform(self, noisy_waveform, denoised_spectrogram, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
        stft_noisy = torch.stft(
            noisy_waveform.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn(n_fft).to(noisy_waveform.device),
            return_complex=True,
            center=False,
            normalized=True
        )
        phase_noisy = torch.angle(stft_noisy)
        denoised_complex_spectrogram = denoised_spectrogram * torch.exp(1j * phase_noisy)
        reconstructed_waveform = torch.istft(
            denoised_complex_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn(n_fft).to(noisy_waveform.device),
            length=noisy_waveform.shape[-1],
        )
        return reconstructed_waveform.unsqueeze(1)

    def forward(self, noisy_waveform, noisy_spectrogram, debug=False):
        # noisy_spectrogram: (B, F, T_spec)
        # noisy_waveform: (B, 1, L)  or (B, L)
        if debug:
            print(f"[DEBUG] Input noisy_waveform: {noisy_waveform.shape}")
            print(f"[DEBUG] Input noisy_spectrogram: {noisy_spectrogram.shape}")

        # 1) denoise spectrogram (CleanSpecNet)
        denoised_spectrogram = self.clean_spec_net(noisy_spectrogram)  # (B, F, T_spec)
        if debug:
            print(f"[DEBUG] Output denoised_spectogram: {denoised_spectrogram.shape}")

        # 2) upsample spectrogram to waveform resolution (x256 in time)
        up_spec_time = self.spec_upsampler(denoised_spectrogram)  # (B, 1, T_high)
        if debug:
            print(f"[DEBUG] Upsampled spec->time feature: {up_spec_time.shape}")

        # 3) make sure noisy_waveform has shape (B,1,L)
        if noisy_waveform.dim() == 2:
            noisy_waveform = noisy_waveform.unsqueeze(1)
        B, C, L = noisy_waveform.shape

        # 4) align lengths: if T_high != L, interpolate to match sample count
        T_high = up_spec_time.shape[-1]
        if T_high != L:
            # linear interpolation in time dimension
            up_spec_time = F.interpolate(up_spec_time, size=L, mode='linear', align_corners=False)
            if debug:
                print(f"[DEBUG] Interpolated up_spec_time to match waveform length: {up_spec_time.shape}")

        # 5) element-wise addition (conditioning by addition as paper suggests)
        # Note: you might want to scale the upsampled feature before addition (learnable scalar or fixed)
        combined = noisy_waveform + up_spec_time  # (B,1,L)
        if debug:
            print(f"[DEBUG] Combined (noisy + up_spec): {combined.shape}")

        # 6) pass through conditioner and CleanUNet
        conditioned_waveform = self.WaveformConditioner(torch.cat((noisy_waveform, up_spec_time), dim=1))
        if debug:
            print(f"[DEBUG] Conditioned waveform (after WaveformConditioner): {conditioned_waveform.shape}")

        denoised_waveform = self.clean_unet(conditioned_waveform)
        if debug:
            print(f"[DEBUG] Output denoised_waveform: {denoised_waveform.shape}")

        return denoised_waveform, denoised_spectrogram


    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        """Método auxiliar para carregar um checkpoint e extrair o state_dict."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Carrega para a CPU para evitar problemas de compatibilidade de dispositivo
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Procura por chaves comuns onde os pesos do modelo são armazenados
        if 'generator' in checkpoint:
            return checkpoint['generator']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            return checkpoint

    def load_cleanunet_weights(self, checkpoint_path):
        """Carrega pesos pré-treinados apenas para o submódulo CleanUNet."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        
        # Remove o 'print' e 'exit' de depuração
        
        clean_unet_state_dict = {}
        # O prefixo correto, com base na sua saída, é "model."
        prefix = "model." 
        
        for k, v in state_dict.items():
            # Apenas processa as chaves que começam com o prefixo esperado
            if k.startswith(prefix):
                # Remove o prefixo para que as chaves correspondam ao submódulo
                clean_unet_state_dict[k[len(prefix):]] = v
        
        if not clean_unet_state_dict:
            raise ValueError(f"Nenhuma chave compatível encontrada no checkpoint. Verifique o prefixo. Chaves disponíveis: {state_dict.keys()}")

        self.clean_unet.load_state_dict(clean_unet_state_dict)
        print("Weights loaded successfully into self.clean_unet.")


    def load_cleanspecnet_weights(self, checkpoint_path):
        """Carrega pesos pré-treinados apenas para o submódulo CleanSpecNet."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        
        clean_spec_net_state_dict = {}
        # O prefixo correto, com base no seu log de erro, também é "model."
        prefix = "model."
        
        for k, v in state_dict.items():
            # Apenas processa as chaves que começam com o prefixo esperado
            if k.startswith(prefix):
                # Remove o prefixo para que as chaves correspondam ao submódulo
                clean_spec_net_state_dict[k[len(prefix):]] = v
        
        if not clean_spec_net_state_dict:
            raise ValueError(f"Nenhuma chave compatível encontrada no checkpoint do CleanSpecNet. Verifique o prefixo. Chaves disponíveis: {state_dict.keys()}")

        self.clean_spec_net.load_state_dict(clean_spec_net_state_dict)
        print("Weights loaded successfully into self.clean_spec_net.")


    def load_cleanunet2_weights(self, checkpoint_path):
        """Carrega pesos pré-treinados para o modelo CleanUNet2 completo."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        self.load_state_dict(state_dict)
        print("Weights loaded successfully into the full CleanUNet2 model.")



# Example usage:
if __name__ == '__main__':
    '''
    noisy_waveform = torch.randn(4, 1, 80000).cuda()
    clean_waveform = torch.randn(4, 1, 80000).cuda()
    noisy_spectrogram = torch.randn(4, 513, 309).cuda()

    model = CleanUNet2().cuda()

    model.load_cleanunet_weights('checkpoints/cleanunet/last.ckpt')
    print(model)

    model.load_cleanspecnet_weights('checkpoints/cleanspecnet/last.ckpt')
    
    print(f"Noisy waveform shape: {noisy_waveform.shape}")
    print(f"Noisy spectrogram shape: {noisy_spectrogram.shape}")
    denoised_waveform = model(noisy_waveform, noisy_spectrogram, debug=False)

    print(f"Denoised waveform shape: {denoised_waveform.shape}")
    print(f"Clean waveform shape: {clean_waveform.shape}")

    loss = torch.nn.MSELoss()(clean_waveform, denoised_waveform)
    loss.backward()
    print(f"[DEBUG] Loss: {loss.item():.6f}")
    '''
    model = CleanUNet2().cuda()
    noisy_w = torch.randn(2, 1, 16000*4).cuda()       # 4s audio 16k
    noisy_spec = torch.randn(2, 513, 200).cuda()      # exemplo: 200 frames
    out = model(noisy_w, noisy_spec, debug=True)
    print(out.shape)  # deve ser (2, 1, 16000*4)

