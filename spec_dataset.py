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
from augmentation import AudioAugmenter
import torchaudio

MAX_WAV_VALUE = 32768.0

def load_wav(full_path, target_sr):
    #sampling_rate, data = read(full_path)
    data, sampling_rate = load(full_path)
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


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, base_mels_path=None, use_interpolation=True, noise_addition=False, config_noise_aug=None, augmentations=None):
        self.audio_files = training_files
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
        #if noise_addition:
        #    self.noise_adder = NoiseAugmentation(sr=self.input_sampling_rate, **config_noise_aug)
        #else:
        #    self.noise_adder = None
        #self.noise_adder = None

        # Initialize the AudioAugmenter with the provided augmentations, if any
        print("Augmentations:", augmentations)
        self.audio_augmenter = AudioAugmenter(augmentations) if augmentations else None
        print("Using audio augmentations")

        if self.audio_augmenter:
            print("Using audio augmentations")        
        self.cached_wav = None
        self.cached_wav_input = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.base_mels_path = base_mels_path
        self.spectrogram_fn = T.Spectrogram(n_fft=1024, hop_length=256, win_length=1024, power=1.0, normalized=True, center=False)
        
    def __getitem__(self, index):
        filename = self.audio_files[index]

        if self._cache_ref_count == 0:
            try:
                #audio, sampling_rate = load_wav(filename)
                clean_audio, sampling_rate = load_wav(filename, self.sampling_rate)
                input_audio = clean_audio
            except Exception as e:
                # if file dont exist or is corrupted, select other sample
                print("WARNING: The file", filename, "Don't exist or is corrupted, please check this. Selecting other sample ...")
                print(e)
                return self.__getitem__(random.randint(0, self.__len__()))
            
            '''
            if self.noise_addition:
                noised_audio = self.noise_adder.augment(clean_audio.clone()).unsqueeze(0)
                input_audio = noised_audio

                if not noised_audio.size(1) == clean_audio.size(1):
                    torchaudio.save('noised_audio.wav', noised_audio, self.input_sampling_rate)
                    torchaudio.save('clean_audio.wav', clean_audio, self.input_sampling_rate)
                assert noised_audio.size(1) == clean_audio.size(1), "Noised audio and clean audio must have the same length"
            '''
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} Sampling_rate doesn't match target {} sampling_rate".format(
                    sampling_rate, self.sampling_rate))
            
            # Apply audio augmentations if any
            if self.audio_augmenter:
                # Convert the tensor to a NumPy array for augmentation
                audio_np = clean_audio.numpy()
                # Apply the augmentations
                audio_np = self.audio_augmenter.apply(audio_np, self.sampling_rate)
                # Convert back to a tensor
                noisy_audio = torch.from_numpy(audio_np)
            else:
                # If there are no augmentations, noisy audio is the same as clean audio.
                noisy_audio = clean_audio.clone()

            input_audio = noisy_audio

            '''
            if (self.result_sampling_rate != self.input_sampling_rate) and (not self.mel_spec_fine_tuning):
                audio_frames = input_audio.size(1) 
                # if odd turn even
                if audio_frames % 2 != 0:
                    input_audio = input_audio[:, :audio_frames-1]
                    clean_audio = clean_audio[:, :audio_frames-1]

                # resample input sample to TTS sampling_rate
                input_audio = self.audio_resample(input_audio)
            else:
                input_audio = None
            '''
            self.cached_clean_wav = clean_audio
            self.cached_input_wav = input_audio

            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            input_audio = self.cached_input_wav
            self._cache_ref_count -= 1


        # split audio into segments
        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                # generate a random even number
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

        if len(input_audio.shape) != 2:
            input_audio = input_audio.squeeze()

        if len(clean_audio.shape) != 2:
            clean_audio = clean_audio.squeeze()

        if len(input_spec.shape) != 3:
            input_spec = input_spec.squeeze()

        if len(clean_spec.shape) != 3:
            clean_spec = clean_spec.squeeze()
                    
        return (input_audio, input_spec, clean_audio, clean_spec)

    def __len__(self):
        return len(self.audio_files)
