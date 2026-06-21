#!/usr/bin/env python3
"""
Pre-extract speaker embeddings for all audio files in a filelist.

Saves embeddings to disk so Stage-1 training can load them directly
instead of extracting on-the-fly (much faster training).

Usage:
    python extract_embeddings.py \
        --config configs/stage1_xvector.yaml \
        --output_dir experiments/exp_xvector_stage1/embedding_cache

    # Or specify model and paths directly:
    python extract_embeddings.py \
        --speaker_model titanet \
        --data_dir /path/to/VoiceBank-DEMAND-16k \
        --filelist filelists/train.csv \
        --output_dir embeddings/titanet

The embeddings are stored as .pt files keyed by MD5 hash of the
relative audio path (compatible with the XVectorCache used during training).
"""

import argparse
import hashlib
import os
import sys
import torch
import torchaudio
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleanunet.speaker_extractor import SpeakerExtractor, SPEAKER_MODELS


def get_cache_key(audio_path: str) -> str:
    """Generate MD5 cache key from audio path (same as XVectorCache)."""
    path_bytes = str(audio_path).encode('utf-8')
    return hashlib.md5(path_bytes).hexdigest()


def load_filelist(filelist_path: str) -> list:
    """Load filelist and return list of clean audio relative paths."""
    paths = []
    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: clean_path|noisy_path
            parts = line.split('|')
            clean_rel = parts[0].strip()
            paths.append(clean_rel)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract speaker embeddings for CleanUNet2 training."
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='YAML config file (extracts speaker_model, data_dir, filelist from it)'
    )
    parser.add_argument(
        '--speaker_model', type=str, default=None,
        help=f'Speaker model to use. Options: {list(SPEAKER_MODELS.keys())}'
    )
    parser.add_argument(
        '--speaker_model_local_path', type=str, default=None,
        help='Local path for clova checkpoint dir or redimnet variant string'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Base directory for audio files'
    )
    parser.add_argument(
        '--filelist', type=str, default=None,
        help='Path to filelist (train.csv or test.csv)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save extracted embeddings'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for extraction (default: 8)'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=16000,
        help='Audio sample rate (default: 16000)'
    )
    parser.add_argument(
        '--max_duration', type=float, default=None,
        help='Max audio duration in seconds (truncate longer files)'
    )
    args = parser.parse_args()

    # Load from config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})

        if args.speaker_model is None:
            args.speaker_model = model_cfg.get('speaker_model', 'xvector')
        if args.speaker_model_local_path is None:
            args.speaker_model_local_path = model_cfg.get('speaker_model_local_path')
        if args.data_dir is None:
            args.data_dir = data_cfg.get('data_dir', '.')
        if args.filelist is None:
            args.filelist = data_cfg.get('train_list_path', 'filelists/train.csv')
        if args.sample_rate == 16000:
            args.sample_rate = data_cfg.get('sampling_rate', 16000)

    # Validate
    if args.speaker_model is None:
        parser.error("--speaker_model is required (or use --config)")
    if args.data_dir is None:
        parser.error("--data_dir is required (or use --config)")
    if args.filelist is None:
        parser.error("--filelist is required (or use --config)")

    if args.speaker_model not in SPEAKER_MODELS:
        parser.error(f"Unknown speaker_model: {args.speaker_model}. Options: {list(SPEAKER_MODELS.keys())}")

    model_info = SPEAKER_MODELS[args.speaker_model]
    print("=" * 70)
    print(f"Speaker Embedding Extraction")
    print("=" * 70)
    print(f"  Model: {model_info['display_name']} ({args.speaker_model})")
    print(f"  Embedding dim: {model_info['embedding_dim']}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Filelist: {args.filelist}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample rate: {args.sample_rate}")
    print("=" * 70)

    # Load extractor
    print("\nLoading speaker model...")
    extractor = SpeakerExtractor(
        model_name=args.speaker_model,
        device='cpu',
        local_path=args.speaker_model_local_path
    )

    # Load filelist
    audio_paths = load_filelist(args.filelist)
    print(f"\nFound {len(audio_paths)} audio files in filelist")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which files already have embeddings (skip if already extracted)
    already_done = 0
    to_process = []
    for rel_path in audio_paths:
        cache_key = get_cache_key(rel_path)
        cache_file = output_dir / f"{cache_key}.pt"
        if cache_file.exists():
            already_done += 1
        else:
            to_process.append(rel_path)

    if already_done > 0:
        print(f"Skipping {already_done} files (already extracted)")
    print(f"Processing {len(to_process)} files...")

    if not to_process:
        print("All embeddings already extracted. Done!")
        return

    # Process in batches
    max_samples = int(args.max_duration * args.sample_rate) if args.max_duration else None
    extracted = 0
    failed = 0

    for i in tqdm(range(0, len(to_process), args.batch_size), desc="Extracting"):
        batch_paths = to_process[i:i + args.batch_size]

        # Load audio batch
        waveforms = []
        valid_paths = []
        for rel_path in batch_paths:
            full_path = os.path.join(args.data_dir, rel_path)
            if not os.path.exists(full_path):
                print(f"\n  WARNING: File not found: {full_path}")
                failed += 1
                continue
            try:
                wav, sr = torchaudio.load(full_path)
                if sr != args.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
                wav = wav.squeeze(0)  # (samples,)
                if max_samples and wav.shape[0] > max_samples:
                    wav = wav[:max_samples]
                waveforms.append(wav)
                valid_paths.append(rel_path)
            except Exception as e:
                print(f"\n  ERROR loading {full_path}: {e}")
                failed += 1
                continue

        if not waveforms:
            continue

        # Pad to same length for batch processing
        max_len = max(w.shape[0] for w in waveforms)
        batch_tensor = torch.zeros(len(waveforms), max_len)
        for j, wav in enumerate(waveforms):
            batch_tensor[j, :wav.shape[0]] = wav

        # Extract embeddings
        with torch.no_grad():
            embeddings = extractor.extract_embeddings(batch_tensor, sample_rate=args.sample_rate)

        # Handle shape: may be (batch, 1, dim) or (batch, dim)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        # Save each embedding
        for j, rel_path in enumerate(valid_paths):
            cache_key = get_cache_key(rel_path)
            cache_file = output_dir / f"{cache_key}.pt"
            torch.save(embeddings[j].cpu(), cache_file)
            extracted += 1

    print(f"\nDone!")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (already done): {already_done}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {extracted + already_done + failed}")
    print(f"  Output dir: {output_dir}")
    print(f"\nTo use during training, add to your config:")
    print(f"  model:")
    print(f"    use_preextracted_embeddings: true")
    print(f"    embedding_cache_dir: {output_dir}")


if __name__ == "__main__":
    main()
