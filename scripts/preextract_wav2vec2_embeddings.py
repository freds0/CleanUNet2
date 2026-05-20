#!/usr/bin/env python3
"""
Script to pre-extract Wav2Vec2 embeddings and save them to disk.
This significantly speeds up training by avoiding repeated embedding extraction.

Usage:
    python scripts/preextract_wav2vec2_embeddings.py \
        --data_dir /path/to/audio/files \
        --file_list filelists/train.csv \
        --output_dir wav2vec2_embeddings_cache \
        --model facebook/wav2vec2-xls-r-2b \
        --pooling_method self_attention \
        --batch_size 8 \
        --num_workers 4
"""

import argparse
import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import hashlib


class AudioFileDataset(Dataset):
    """Simple dataset for loading audio files."""

    def __init__(self, file_list, data_dir, target_sr=16000):
        self.data_dir = Path(data_dir) if data_dir else Path('.')
        self.target_sr = target_sr

        # Load file list
        if file_list.endswith('.csv'):
            # Try to read CSV - it may not have headers and uses | as separator
            try:
                df = pd.read_csv(file_list, sep='|', header=None, names=['clean', 'noisy'])
                self.audio_files = df['clean'].tolist()
            except Exception as e:
                print(f"Error reading CSV with | separator: {e}")
                # Try with comma separator and header
                try:
                    df = pd.read_csv(file_list)
                    self.audio_files = df['clean'].tolist()
                except:
                    # Fallback: read as text file
                    with open(file_list, 'r') as f:
                        self.audio_files = [line.strip().split('|')[0] for line in f]
        else:
            # Assume text file with one path per line
            with open(file_list, 'r') as f:
                self.audio_files = [line.strip() for line in f]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        try:
            # Handle both absolute and relative paths
            if Path(audio_file).is_absolute():
                audio_path = Path(audio_file)
            else:
                audio_path = self.data_dir / audio_file

            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Squeeze to 1D
            waveform = waveform.squeeze(0)

            return {
                'waveform': waveform,
                'path': str(audio_path),
                'filename': self.audio_files[idx]
            }
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return empty waveform to skip this file
            return {
                'waveform': torch.zeros(self.target_sr),  # 1 second of silence
                'path': str(audio_path) if 'audio_path' in locals() else audio_file,
                'filename': audio_file,
                'error': str(e)
            }


def collate_fn(batch):
    """Collate function to handle variable-length audio."""
    # Find max length
    max_len = max(item['waveform'].shape[0] for item in batch)

    # Pad all waveforms to max length
    waveforms = []
    for item in batch:
        waveform = item['waveform']
        if waveform.shape[0] < max_len:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
        waveforms.append(waveform)

    return {
        'waveform': torch.stack(waveforms),
        'path': [item['path'] for item in batch],
        'filename': [item['filename'] for item in batch]
    }


def extract_wav2vec2_embeddings(
    model_name,
    data_dir,
    file_list,
    output_dir,
    batch_size=8,
    num_workers=4,
    device='cuda',
    layer_index=24
):
    """Extract and save raw Wav2Vec2 embeddings from intermediate layer without pooling."""

    print("=" * 80)
    print("Wav2Vec2 Embedding Extraction (Raw Sequence - Middle Layer)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"File list: {file_list}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Layer index: {layer_index}")
    print(f"Mode: Raw sequence (no pooling)")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Wav2Vec2 model
    print("\nLoading Wav2Vec2 model...")
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded: {model_name}")
    print(f"  Hidden size: {model.config.hidden_size}")

    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = AudioFileDataset(file_list, data_dir, target_sr=16000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"✓ Dataset loaded: {len(dataset)} audio files")

    # Extract embeddings
    print("\nExtracting raw sequence embeddings (no pooling)...")
    total_extracted = 0
    total_skipped = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            waveforms = batch['waveform'].to(device)
            filenames = batch['filename']

            # Extract Wav2Vec2 features from intermediate layer
            outputs = model(waveforms, output_hidden_states=True)
            # Get hidden states from middle layer (layer 24 for 48-layer model)
            hidden_states = outputs.hidden_states[layer_index]  # (B, T, D)

            # NO POOLING - save raw sequence embeddings from middle layer

            # Save embeddings to disk
            for i, filename in enumerate(filenames):
                # Skip files with errors (check if waveform is all zeros)
                if batch['waveform'][i].abs().max() < 1e-6:
                    print(f"Skipping file with error: {filename}")
                    total_skipped += 1
                    continue

                # Create hash of filename for unique ID
                file_hash = hashlib.md5(filename.encode()).hexdigest()
                save_path = output_dir / f"{file_hash}.pt"

                # Skip if already exists
                if save_path.exists():
                    total_skipped += 1
                    continue

                # Save raw sequence embedding (T, D) from middle layer
                torch.save({
                    'embedding': hidden_states[i].cpu(),  # Shape: (T, D)
                    'filename': filename,
                    'model': model_name,
                    'layer_index': layer_index,
                    'pooling_method': 'none'  # No pooling applied
                }, save_path)

                total_extracted += 1

    print("\n" + "=" * 80)
    print("Extraction complete!")
    print(f"✓ Extracted: {total_extracted} embeddings")
    print(f"⊘ Skipped (already exist): {total_skipped} embeddings")
    print(f"✓ Total files processed: {len(dataset)}")
    print(f"✓ Embeddings saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Pre-extract raw Wav2Vec2 embeddings from middle layer (no pooling)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--file_list', type=str, required=True,
                        help='CSV file with audio file paths (column: clean)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-xls-r-2b',
                        help='Wav2Vec2 model name')
    parser.add_argument('--layer_index', type=int, default=24,
                        help='Layer index to extract from (default: 24 for 48-layer model)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'

    extract_wav2vec2_embeddings(
        model_name=args.model,
        data_dir=args.data_dir,
        file_list=args.file_list,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        layer_index=args.layer_index
    )


if __name__ == '__main__':
    main()
