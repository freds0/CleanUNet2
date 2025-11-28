import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
#from scipy.io.wavfile import read
from torchaudio import load
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel_fn
#from noise import NoiseAugmentation
#from augmentation import AudioAugmenter
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

#from speechbrain.inference.speaker import EncoderClassifier

MAX_WAV_VALUE = 32768.0

def load_wav(full_path, target_sr):
    #sampling_rate, data = read(full_path)
    data, sampling_rate = load(full_path, normalize=True)
    if sampling_rate != target_sr:
        #data = librosa.resample(data, sampling_rate, target_sr)
        data = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)(data)
        sampling_rate = target_sr
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def get_dataset_filelist(filelist):
    with open(filelist, 'r', encoding='utf-8') as ifile:
        lines = ifile.readlines()
        training_files = [(x.strip().split('|')[0], x.strip().split('|')[1]) for x in lines]
    return training_files


def custom_collate_fn(batch):
    """
    Agrupa os elementos do batch, lidando com tensores de diferentes comprimentos
    (ex: embeddings Wav2Vec com tamanho de tempo variável).
    """
    audios, specs, clean_audios, clean_specs = zip(*batch)

    return (
        torch.stack(audios),           # [B, T]
        torch.stack(specs),            # [B, F, T]
        torch.stack(clean_audios),     # [B, T]
        torch.stack(clean_specs),      # [B, F, T]
    )


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_files, segment_size=8192, n_fft=1025, num_mels=80,
                 hop_size=256, win_size=1024, sampling_rate=16000, fmin=0, fmax=8000, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, noise_addition=False, augmentations=None):
        self.data_dir = data_dir

        self.audio_files = get_dataset_filelist(data_files)

        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.noise_addition = noise_addition

        # Initialize the AudioAugmenter with the provided augmentations, if any        
        #self.audio_augmenter = AudioAugmenter(augmentations) if augmentations else None
        #if self.audio_augmenter:
        #    print("Using audio augmentations")

        self.cached_wav = None
        self.cached_wav_input = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.spectrogram_fn = T.Spectrogram(n_fft=1024, hop_length=256, win_length=1024, power=1.0, normalized=True, center=False)
        #self.xvector_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_xvector_model/spkrec-xvect-voxceleb", run_opts={"device": "cpu"}).eval()
        
    def __getitem__(self, index):
        #for attempt in range(3):  # até 3 tentativas
        clean_path, noisy_path = self.audio_files[index]

        clean_filepath = os.path.join(self.data_dir, clean_path)
        noisy_filepath = os.path.join(self.data_dir, noisy_path)

        try:
            clean_audio, clean_sr = load_wav(clean_filepath, self.sampling_rate)
            noisy_audio, noisy_sr = load_wav(noisy_filepath, self.sampling_rate)

            assert clean_sr == noisy_sr, "Sampling rates do not match"

            dir_path =  os.path.dirname(os.path.dirname(clean_filepath))
            filename = os.path.basename(clean_filepath)

            clean_audio = clean_audio / clean_audio.abs().max()
            input_audio = noisy_audio / noisy_audio.abs().max()

            self.cached_wav = clean_audio
            self.cached_wav_input = input_audio
            self._cache_ref_count = self.n_cache_reuse

            #    break  # sucesso, sai do loop

        except Exception as e:
            #print(f"[{attempt+1}/3] Erro ao processar {filename}: {e}")
            print(f"Erro ao processar {filename}: {e}")
            raise e

        if self._cache_ref_count > 0:
            clean_audio = self.cached_wav
            input_audio = self.cached_wav_input
            self._cache_ref_count -= 1

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                while audio_start % 2 != 0:
                    audio_start = random.randint(0, max_audio_start)
                audio_end = audio_start + self.segment_size
                clean_audio = clean_audio[:, audio_start:audio_end]
                input_audio = input_audio[:, audio_start:audio_end]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)), 'constant')

        input_spec = self.spectrogram_fn(input_audio).squeeze()
        clean_spec = self.spectrogram_fn(clean_audio).squeeze()

        input_audio = input_audio.squeeze().unsqueeze(0)
        clean_audio = clean_audio.squeeze().unsqueeze(0)
        input_spec = input_spec.squeeze()
        clean_spec = clean_spec.squeeze()

        return (input_audio, input_spec, clean_audio, clean_spec)


    def __len__(self):
        return len(self.audio_files)
