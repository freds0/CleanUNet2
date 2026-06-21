#!/usr/bin/env python3
"""
Pre-extract speaker embeddings for all audio files in a filelist.

Saves embeddings to disk so Stage-1 training can load them directly
instead of extracting on-the-fly (much faster training).

Uses GPU when available for fast batch extraction.
Skips files that already have cached embeddings.

Usage:
    # From config (reads speaker_model, data_dir, filelist, output_dir)
    python extract_embeddings.py --config configs/stage1_titanet.yaml

    # With explicit arguments
    python extract_embeddings.py \
        --speaker_model titanet \
        --data_dir /path/to/VoiceBank-DEMAND-16k \
        --filelist filelists/train.csv \
        --output_dir embeddings/titanet

    # Extract both train and test
    python extract_embeddings.py --config configs/stage1_titanet.yaml --filelist filelists/train.csv
    python extract_embeddings.py --config configs/stage1_titanet.yaml --filelist filelists/test.csv
"""

import argparse
import hashlib
import json
import os
import sys
import torch
import torchaudio
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleanunet.speaker_extractor import SPEAKER_MODELS


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
            parts = line.split('|')
            clean_rel = parts[0].strip()
            paths.append(clean_rel)
    return paths


def load_extractor_on_device(speaker_model, device, local_path=None):
    """
    Load the speaker model directly on the specified device (GPU or CPU).
    Unlike SpeakerExtractor (which forces CPU for training safety),
    this loads on GPU for fast offline extraction.
    """
    model_info = SPEAKER_MODELS[speaker_model]
    backend = model_info['backend']

    if backend == 'speechbrain':
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        source = model_info['source']
        savedir = model_info['savedir']

        if local_path and os.path.exists(local_path):
            classifier = EncoderClassifier.from_hparams(
                source=local_path, savedir=local_path,
                run_opts={"device": str(device)}
            )
        else:
            classifier = EncoderClassifier.from_hparams(
                source=source, savedir=savedir,
                run_opts={"device": str(device)}
            )

        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False

        def extract_fn(waveform_batch):
            return classifier.encode_batch(waveform_batch.to(device))

        return extract_fn

    elif backend == 'clova':
        from cleanunet.speaker_extractor import ClovaResNetEncoder

        if not local_path or not os.path.exists(local_path):
            raise ValueError(f"Clova requires speaker_model_local_path. Got: {local_path}")

        model_path = os.path.join(local_path, 'model_se.pth')
        config_path = os.path.join(local_path, 'config_se.json')
        if not os.path.exists(model_path):
            alt = os.path.join(local_path, 'model_se.pth.tar')
            if os.path.exists(alt):
                model_path = alt

        with open(config_path, 'r') as f:
            config = json.load(f)

        encoder = ClovaResNetEncoder(
            input_dim=config.get('model_params', {}).get('input_dim', 64),
            proj_dim=config.get('model_params', {}).get('proj_dim', 512),
            audio_config=config.get('audio', {}),
        )
        state = torch.load(model_path, map_location='cpu')
        encoder.load_state_dict(state['model'] if 'model' in state else state)
        encoder.eval()
        encoder = encoder.to(device)
        for p in encoder.parameters():
            p.requires_grad = False

        def extract_fn(waveform_batch):
            waveform_batch = waveform_batch.to(device)
            batch_embs = []
            for i in range(waveform_batch.shape[0]):
                emb = encoder.compute_embedding(waveform_batch[i:i+1])
                batch_embs.append(emb)
            return torch.cat(batch_embs, dim=0)

        return extract_fn

    elif backend == 'redimnet':
        hub_repo = model_info['hub_repo']
        model_name = model_info.get('default_model_name', 'b2')
        train_type = model_info.get('default_train_type', 'ft_lm')
        dataset = model_info.get('default_dataset', 'vox2')

        if local_path and ':' in str(local_path):
            parts = str(local_path).split(':')
            if len(parts) >= 1:
                model_name = parts[0]
            if len(parts) >= 2:
                train_type = parts[1]
            if len(parts) >= 3:
                dataset = parts[2]

        redimnet = torch.hub.load(
            hub_repo, 'ReDimNet',
            model_name=model_name, train_type=train_type, dataset=dataset,
        )
        redimnet.eval()
        redimnet = redimnet.to(device)
        for p in redimnet.parameters():
            p.requires_grad = False

        def extract_fn(waveform_batch):
            return redimnet(waveform_batch.to(device))

        return extract_fn

    elif backend == 'nemo':
        import nemo.collections.asr as nemo_asr

        nemo_model_name = model_info['nemo_model_name']

        if local_path and os.path.isfile(local_path) and local_path.endswith('.nemo'):
            model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(local_path)
        else:
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(nemo_model_name)

        model.eval()
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        def extract_fn(waveform_batch):
            waveform_batch = waveform_batch.to(device)
            lengths = torch.tensor([waveform_batch.shape[1]] * waveform_batch.shape[0], device=device)
            _, embs = model(input_signal=waveform_batch, input_signal_length=lengths)
            return embs

        return extract_fn

    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract speaker embeddings for CleanUNet2 training."
    )
    parser.add_argument('--config', type=str, default=None,
        help='YAML config file (reads speaker_model, data_dir, filelist, output_dir)')
    parser.add_argument('--speaker_model', type=str, default=None,
        help=f'Speaker model. Options: {list(SPEAKER_MODELS.keys())}')
    parser.add_argument('--speaker_model_local_path', type=str, default=None,
        help='Local path for clova checkpoint or redimnet variant string')
    parser.add_argument('--data_dir', type=str, default=None,
        help='Base directory for audio files')
    parser.add_argument('--filelist', type=str, default=None,
        help='Path to filelist (train.csv or test.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
        help='Directory to save embeddings (default: from config embedding_cache_dir)')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size for extraction (default: 32)')
    parser.add_argument('--sample_rate', type=int, default=16000,
        help='Audio sample rate (default: 16000)')
    parser.add_argument('--max_duration', type=float, default=None,
        help='Max audio duration in seconds (truncate longer files)')
    parser.add_argument('--device', type=str, default='auto',
        help='Device: auto, cuda, cpu (default: auto)')
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
        if args.output_dir is None:
            args.output_dir = model_cfg.get('embedding_cache_dir')
        if args.data_dir is None:
            args.data_dir = data_cfg.get('data_dir', '.')
        if args.filelist is None:
            # Use both train and val filelists when loading from config
            train_list = data_cfg.get('train_list_path', 'filelists/train.csv')
            val_list = data_cfg.get('val_list_path', 'filelists/test.csv')
            args.filelist = [train_list, val_list]
        if args.sample_rate == 16000:
            args.sample_rate = data_cfg.get('sampling_rate', 16000)

    # Wrap single filelist string into list for uniform handling
    if args.filelist and isinstance(args.filelist, str):
        args.filelist = [args.filelist]

    # Validate
    if args.speaker_model is None:
        parser.error("--speaker_model is required (or use --config)")
    if args.data_dir is None:
        parser.error("--data_dir is required (or use --config)")
    if args.filelist is None:
        parser.error("--filelist is required (or use --config)")
    if args.output_dir is None:
        parser.error("--output_dir is required (or use --config with embedding_cache_dir)")
    if args.speaker_model not in SPEAKER_MODELS:
        parser.error(f"Unknown speaker_model: {args.speaker_model}. Options: {list(SPEAKER_MODELS.keys())}")

    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model_info = SPEAKER_MODELS[args.speaker_model]
    print("=" * 70)
    print("Speaker Embedding Extraction")
    print("=" * 70)
    print(f"  Model: {model_info['display_name']} ({args.speaker_model})")
    print(f"  Embedding dim: {model_info['embedding_dim']}")
    print(f"  Device: {device}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Filelists: {', '.join(args.filelist)}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample rate: {args.sample_rate}")
    print("=" * 70)

    # Load extractor on device (GPU if available)
    print("\nLoading speaker model...")
    extract_fn = load_extractor_on_device(
        args.speaker_model, device, local_path=args.speaker_model_local_path
    )
    print("Model loaded!")

    # Load filelist(s)
    audio_paths = []
    for fl in args.filelist:
        paths = load_filelist(fl)
        print(f"  Loaded {len(paths)} files from {fl}")
        audio_paths.extend(paths)
    # Deduplicate preserving order
    audio_paths = list(dict.fromkeys(audio_paths))
    print(f"\nTotal unique audio files: {len(audio_paths)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip files that already have embeddings
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
    print(f"Processing {len(to_process)} files...\n")

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
                tqdm.write(f"  WARNING: File not found: {full_path}")
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
                tqdm.write(f"  ERROR loading {full_path}: {e}")
                failed += 1
                continue

        if not waveforms:
            continue

        # Pad to same length for batch processing
        max_len = max(w.shape[0] for w in waveforms)
        batch_tensor = torch.zeros(len(waveforms), max_len)
        for j, wav in enumerate(waveforms):
            batch_tensor[j, :wav.shape[0]] = wav

        # Extract embeddings on device
        with torch.no_grad():
            embeddings = extract_fn(batch_tensor)

        # Handle shape: may be (batch, 1, dim) or (batch, dim)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        # Save each embedding to disk
        embeddings_cpu = embeddings.cpu()
        for j, rel_path in enumerate(valid_paths):
            cache_key = get_cache_key(rel_path)
            cache_file = output_dir / f"{cache_key}.pt"
            torch.save(embeddings_cpu[j], cache_file)
            extracted += 1

    print(f"\nDone!")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (already done): {already_done}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {extracted + already_done + failed}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
