#!/usr/bin/env python
"""
Optimized Inference Script for CleanUNet2.
Features:
- Batch processing (major speedup vs single file).
- Multi-worker data loading.
- Dynamic padding for variable length audio.
- Mixed Precision (AMP) support.
- Torch Compile support (optional for PyTorch 2.0+).

Usage:
    python inference.py --config configs/inference.yaml
"""

import os
import glob
import yaml
import argparse
import traceback
from argparse import Namespace
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# Ensure this import matches your project structure
from lightning_modules.cleanunet_module import CleanUNetLightningModule


class InferenceDataset(Dataset):
    """
    Efficient Dataset to load and pre-process audio files in parallel (CPU).
    """
    def __init__(self, file_paths: List[str], target_sr: int = 16000):
        self.file_paths = file_paths
        self.target_sr = target_sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # 1. Load audio
            waveform, sr = torchaudio.load(path)
            
            # 2. Mix to Mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Resample if sampling rate differs from target
            if sr != self.target_sr:
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Return: waveform tensor, filename string, original sample rate (int), original length (int)
            return waveform, os.path.basename(path), sr, waveform.shape[-1]
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return None

def inference_collate_fn(batch):
    """
    Collates a list of audio tensors with different lengths by padding dynamically.
    """
    # Filter out failed loads
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    waveforms, filenames, original_srs, original_lens = zip(*batch)

    # Find max length in current batch
    max_len = max([w.shape[-1] for w in waveforms])

    # Pad all waveforms to match the max length
    padded_wavs = []
    for w in waveforms:
        pad_amount = max_len - w.shape[-1]
        if pad_amount > 0:
            # Pad at the end (last dimension)
            w = F.pad(w, (0, pad_amount))
        padded_wavs.append(w)

    # Stack into a single batch tensor (Batch, Channels, Time)
    batch_tensor = torch.stack(padded_wavs)

    return batch_tensor, filenames, original_srs, original_lens


def run_inference(config: dict):
    # ---------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------
    inf_cfg = config.get('inference', {})
    model_hparams = config.get('model', {})
    audio_cfg = config.get('audio', {})

    input_dir = inf_cfg.get('input_dir')
    output_dir = inf_cfg.get('output_dir', 'denoised_results')
    checkpoint_path = inf_cfg.get('checkpoint_path')
    
    # Performance parameters
    batch_size = inf_cfg.get('batch_size', 1)
    num_workers = inf_cfg.get('num_workers', 4)
    use_amp = inf_cfg.get('use_amp', True)     # Automatic Mixed Precision
    
    force_cpu = inf_cfg.get('force_cpu', False)
    use_cuda = torch.cuda.is_available() and not force_cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    print(f"[INFO] Running on: {device} | Batch Size: {batch_size} | AMP: {use_amp}")

    # ---------------------------------------------------
    # 2. Load Model
    # ---------------------------------------------------
    hparams = Namespace(**model_hparams)
    try:
        print(f"[INFO] Loading model from checkpoint: {checkpoint_path}")
        model = CleanUNetLightningModule(hparams)
        
        # Load weights directly to the target device to save CPU RAM
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # strict=False allows loading even if some keys (like loss params) are missing
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        # Optional optimization: Torch Compile (PyTorch 2.0+)
        # Note: If inference is slow or hangs on first batch, try disabling this block.
        if hasattr(torch, 'compile') and use_cuda:
            try:
                print("[INFO] Compiling model with torch.compile (this may take a minute)...")
                # mode='reduce-overhead' is good for small batches, 'max-autotune' for speed
                model = torch.compile(model) 
            except Exception as e:
                print(f"[WARNING] torch.compile failed: {e}. Falling back to eager mode.")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        return

    # ---------------------------------------------------
    # 3. Data Preparation (Dataset & DataLoader)
    # ---------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    input_pattern = inf_cfg.get('input_pattern', '*.wav')
    search_path = os.path.join(input_dir, input_pattern)
    
    # Recursive search if pattern contains "**"
    recursive_search = "**" in input_pattern
    audio_files = glob.glob(search_path, recursive=recursive_search)

    if not audio_files:
        print(f"[WARNING] No files found at {search_path}")
        return

    print(f"[INFO] Found {len(audio_files)} files to process.")

    target_sr = audio_cfg.get('target_sample_rate', 16000)
    dataset = InferenceDataset(audio_files, target_sr=target_sr)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=inference_collate_fn,
        pin_memory=use_cuda
    )

    # ---------------------------------------------------
    # 4. Inference Loop
    # ---------------------------------------------------
    # STFT parameters must match training configuration of CleanSpecNet
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length).to(device)

    # AMP Context (no-op if on CPU or disabled)
    amp_context = torch.autocast(device_type="cuda") if use_amp and use_cuda else torch.no_grad()

    print("[INFO] Starting inference loop...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Denoising"):
            if batch is None: continue
            
            # Unpack batch
            waveforms, filenames, original_srs, original_lens = batch
            waveforms = waveforms.to(device, non_blocking=True) # (B, 1, T_padded)

            # CleanUNet2 requires (Audio, Spectrogram) inputs.
            # We compute the spectrogram on-the-fly for the batch.
            with amp_context:
                # Remove channel dim for STFT: (B, 1, T) -> (B, T)
                spec_batch = torch.stft(
                    waveforms.squeeze(1),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=True,
                    return_complex=True
                ).abs()

                # Forward Pass
                output = model(waveforms, spec_batch)
                
                # Handle return type: might be (wav, spec) or just wav
                if isinstance(output, (tuple, list)):
                    enhanced_batch = output[0]
                else:
                    enhanced_batch = output

            # Post-processing and Saving
            # Move entire batch to CPU to unblock GPU
            enhanced_batch = enhanced_batch.float().cpu()

            for i, filename in enumerate(filenames):
                # Crop padding to restore original length
                length = original_lens[i]
                audio = enhanced_batch[i, :, :length] # (1, Length)

                # Get original sample rate (FIX: removed .item() as it's a python int tuple)
                orig_sr = original_srs[i]
                
                # Optional: Resample back to original SR if needed
                if orig_sr != target_sr:
                     resampler_back = T.Resample(target_sr, orig_sr)
                     audio = resampler_back(audio)

                # Save to disk
                save_path = os.path.join(output_dir, filename)
                torchaudio.save(save_path, audio, orig_sr)

    print(f"[INFO] Done! Results saved to '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Enable cuDNN benchmark for optimized kernel selection
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
        # Set higher precision for matrix multiplication on Ampere+ GPUs
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')

    run_inference(config)
