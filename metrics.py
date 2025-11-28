import os
import argparse
from glob import glob
from os.path import join, basename, isdir
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi

class ObjectiveMetricsPredictor:
    def __init__(self, device=None):        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def si_snr(self, estimate, reference, epsilon=1e-8):
        # Garante tensores
        if isinstance(estimate, np.ndarray):
            estimate = torch.from_numpy(estimate)
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference)
            
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        reference_pow = reference.pow(2).mean(axis=-1, keepdim=True)
        mix_pow = (estimate * reference).mean(axis=-1, keepdim=True)
        scale = mix_pow / (reference_pow + epsilon)

        reference = scale * reference
        error = estimate - reference

        reference_pow = reference.pow(2)
        error_pow = error.pow(2)

        reference_pow = reference_pow.mean(axis=-1)
        error_pow = error_pow.mean(axis=-1)

        si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
        return si_snr.item()

    def _load_file(self, filepath: str):
        try:
            waveform, sr = torchaudio.load(filepath)
            # Reamostragem se necessário (PESQ requer 16k ou 8k)
            if sr != 16000:
                waveform = resample(waveform, sr, 16000)
                sr = 16000
            # Mix para mono se for estéreo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None, None
        return waveform, sr
    
    def __call__(self, input_str_clean, input_str_distorted):
        if isdir(input_str_clean):
            return self.predict_folder(input_str_clean, input_str_distorted)
        else:
            return self.predict_file(input_str_clean, input_str_distorted)

    def predict_metrics(self, waveform_clean, waveform_distorted) -> dict:
        """
        Calcula métricas a partir de tensores ou arrays numpy.
        Espera entradas 1D (time) ou (1, time).
        """
        # Converter para CPU e remover dimensões extras (garantir 1D para PESQ/STOI)
        if torch.is_tensor(waveform_clean):
            waveform_clean = waveform_clean.detach().cpu().squeeze()
        if torch.is_tensor(waveform_distorted):
            waveform_distorted = waveform_distorted.detach().cpu().squeeze()
            
        # Converter para Numpy se ainda não for
        if not isinstance(waveform_clean, np.ndarray):
            waveform_clean = waveform_clean.numpy()
        if not isinstance(waveform_distorted, np.ndarray):
            waveform_distorted = waveform_distorted.numpy()

        # Validação básica de shape
        if waveform_clean.ndim != 1 or waveform_distorted.ndim != 1:
            # Tenta forçar flatten se algo estranho passou
            waveform_clean = waveform_clean.flatten()
            waveform_distorted = waveform_distorted.flatten()

        try:
            pesq_score = pesq(16000, waveform_clean, waveform_distorted, mode="wb")
        except Exception:
            pesq_score = 0.0
            
        try:
            stoi_score = stoi(waveform_clean, waveform_distorted, 16000, extended=False)
        except Exception:
            stoi_score = 0.0
            
        # Para SI-SNR, usamos a versão torch (precisa de dimensões compatíveis)
        # Re-convertemos para tensor para aproveitar a função existente
        w_clean_t = torch.from_numpy(waveform_clean).unsqueeze(0) # (1, T)
        w_dist_t = torch.from_numpy(waveform_distorted).unsqueeze(0)
        
        si_sdr_score = self.si_snr(w_dist_t, w_clean_t)

        return {
            "stoi": stoi_score,
            "pesq": pesq_score,
            "si_sdr": si_sdr_score
        }

    def predict_file(self, filepath_clean: str, filepath_distorted: str) -> dict:
        waveform_clean, _ = self._load_file(filepath_clean)
        if waveform_clean is None:
            return None

        waveform_distorted, _ = self._load_file(filepath_distorted)
        if waveform_distorted is None:
            return None

        return self.predict_metrics(waveform_clean, waveform_distorted)

    def predict_folder(self, dirpath_clean: str, dirpath_distorted: str, batch_size: int = 1, search_str: str = '*.wav') -> dict:
        # Nota: batch_size é mantido como argumento para compatibilidade, 
        # mas o processamento é feito item a item por simplicidade e robustez com PESQ.
        scores = {}
        filelist_clean = sorted(glob(join(dirpath_clean, search_str)))
        filelist_distorted = sorted(glob(join(dirpath_distorted, search_str)))
        
        if len(filelist_clean) == 0:
            print("Nenhum arquivo encontrado.")
            return {}

        assert len(filelist_clean) == len(filelist_distorted), \
            f"Número de arquivos diferente: Clean({len(filelist_clean)}) vs Distorted({len(filelist_distorted)})"

        for f_clean, f_dist in tqdm(zip(filelist_clean, filelist_distorted), total=len(filelist_clean), desc="Calculating Metrics"):
            filename = os.path.basename(f_clean)
            metrics = self.predict_file(f_clean, f_dist)
            if metrics:
                scores[filename] = metrics
                     
        return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_clean',      '-i', type=str, default='samples/0', help='Input clean folder')
    parser.add_argument('--input_dir_dist',       '-r', type=str, default='samples/0', help='Input distorted folder')
    parser.add_argument('--output_file',          '-o', type=str, default='metrics_prediction.json', help='Output json filepath')
    # batch_size removido da lógica crítica, mas mantido no arg parser para não quebrar chamadas antigas
    parser.add_argument('--batch_size',           '-b', type=int, default=1, help='(Deprecated) Batch size')
    parser.add_argument('--search_pattern',       '-s', type=str, default='*.wav', help='Search pattern')
    parser.add_argument('--device',               '-d', type=str, default=None, help='Device: cpu | cuda')
    args = parser.parse_args()

    quality_predictor = ObjectiveMetricsPredictor(args.device)    
    scores = quality_predictor.predict_folder(args.input_dir_clean, args.input_dir_dist, search_str=args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(scores, ofile, indent=4)

if __name__ == "__main__":
    main()