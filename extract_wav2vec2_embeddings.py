"""
Pre-extract Wav2Vec2 embeddings for all audio files before training.

This script extracts embeddings from all clean audio files in the training/validation
datasets and saves them to disk. During training, the model will load pre-extracted
embeddings instead of extracting them on-the-fly.

Usage:
    python extract_wav2vec2_embeddings.py --config configs/train_xvector_vanilla_stage1.yaml
"""

import torch
import torchaudio
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import hashlib
import os
from cleanunet.wav2vec2_extractor import Wav2Vec2Extractor


def get_cache_key(audio_path):
    """Generate a unique cache key from audio file path using MD5 hash."""
    path_bytes = str(audio_path).encode('utf-8')
    return hashlib.md5(path_bytes).hexdigest()


def load_audio_files_from_config(config):
    """
    Load list of audio files from config.

    Returns:
        list: List of clean audio file paths
    """
    data_dir = config['data']['data_dir']
    train_list = config['data']['train_list_path']
    val_list = config['data']['val_list_path']

    audio_files = []

    # Load training files
    print(f"[Extract] Loading training files from: {train_list}")
    train_path = Path(train_list)
    if train_path.exists():
        with open(train_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Handle multiple separator formats: clean,noisy or clean|noisy
                separator = ',' if ',' in line else '|' if '|' in line else None

                if separator:
                    clean_rel, noisy_rel = line.split(separator, 1)
                    clean_rel = clean_rel.strip()
                    noisy_rel = noisy_rel.strip()

                    # Add clean path (absolute path if it starts with /, else relative to data_dir)
                    clean_path = clean_rel if clean_rel.startswith('/') else os.path.join(data_dir, clean_rel)
                    audio_files.append(clean_path)
                else:
                    # Single file format
                    clean_rel = line.strip()
                    clean_path = clean_rel if clean_rel.startswith('/') else os.path.join(data_dir, clean_rel)
                    audio_files.append(clean_path)
    else:
        print(f"[Extract] Warning: Training list not found: {train_list}")

    # Load validation files
    print(f"[Extract] Loading validation files from: {val_list}")

    # Check if it's a two_folders format BEFORE trying to create Path
    if isinstance(val_list, str) and val_list.startswith("two_folders:"):
        print("[Extract] Detected two_folders format for validation")
        print("[Extract] Extracting embeddings from both clean and noisy folders...")

        # Parse two_folders format
        folders_spec = val_list[len("two_folders:"):]
        try:
            clean_dir, noisy_dir = folders_spec.split(",", 1)
            clean_dir = clean_dir.strip()
            noisy_dir = noisy_dir.strip()

            # Get all audio files from clean folder
            from glob import glob
            clean_files = []
            for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
                clean_files.extend(glob(os.path.join(clean_dir, ext)))
                clean_files.extend(glob(os.path.join(clean_dir, '**', ext), recursive=True))

            audio_files.extend(clean_files)
            print(f"[Extract] Found {len(clean_files)} files in clean folder: {clean_dir}")

        except Exception as e:
            print(f"[Extract] Warning: Failed to parse two_folders format: {e}")
    else:
        # Standard file list format
        val_path = Path(val_list)
        if val_path.exists():
            with open(val_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Handle multiple separator formats: clean,noisy or clean|noisy
                    separator = ',' if ',' in line else '|' if '|' in line else None

                    if separator:
                        clean_rel, noisy_rel = line.split(separator, 1)
                        clean_rel = clean_rel.strip()
                        noisy_rel = noisy_rel.strip()

                        # Add clean path (absolute path if it starts with /, else relative to data_dir)
                        clean_path = clean_rel if clean_rel.startswith('/') else os.path.join(data_dir, clean_rel)
                        audio_files.append(clean_path)
                    else:
                        # Single file format
                        clean_rel = line.strip()
                        clean_path = clean_rel if clean_rel.startswith('/') else os.path.join(data_dir, clean_rel)
                        audio_files.append(clean_path)
        else:
            print(f"[Extract] Warning: Validation list not found: {val_list}")

    # Remove duplicates while preserving order
    audio_files = list(dict.fromkeys(audio_files))

    print(f"[Extract] Found {len(audio_files)} unique audio files")
    return audio_files


def extract_and_save_embeddings(config, device='cuda', force_reextract=False):
    """
    Extract wav2vec2 embeddings for all audio files and save as RAW sequences.

    Saves full temporal sequence (batch, time_steps, embedding_dim) instead of pooled embeddings.
    Self-attention pooling will be applied during model training.

    Args:
        config (dict): Configuration dictionary
        device (str): Device to use for extraction ('cuda' or 'cpu')
        force_reextract (bool): If True, re-extract even if embeddings exist
    """
    # Get cache directory from config
    cache_dir = config['model'].get('wav2vec2_cache_dir', 'wav2vec2_embeddings_raw')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Wav2Vec2 Raw Embedding Pre-Extraction (No Pooling)")
    print("=" * 80)
    print("Saving full temporal sequences for self-attention pooling during training")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {device}")
    print(f"Force re-extract: {force_reextract}")
    print("=" * 80 + "\n")

    # Load audio files
    audio_files = load_audio_files_from_config(config)

    if len(audio_files) == 0:
        print("[Extract] Error: No audio files found!")
        return

    # Load Wav2Vec2 configuration (try multiple keys for compatibility)
    expected_model = 'facebook/wav2vec2-xls-r-2b'

    # Support multiple config key names for model
    model_name = (config['model'].get('wav2vec2_model') or
                  config['model'].get('wav2vec2', {}).get('model_name'))

    # Helpful error message if config is missing Wav2Vec2 model
    if model_name is None:
        print("[Extract] ERROR: Missing Wav2Vec2 model in config!")
        print()
        print("[Extract] Add the following to your config under 'model' section:")
        print()
        print("  model:")
        print("    wav2vec2:")
        print(f"      model_name: \"{expected_model}\"")
        print()
        raise ValueError("Wav2Vec2 model not specified in config")

    # Validate model is correct
    if model_name != expected_model:
        print(f"[Extract] ERROR: Wrong Wav2Vec2 model in config!")
        print(f"[Extract] Expected: {expected_model}")
        print(f"[Extract] Got:      {model_name}")
        print()
        print("[Extract] Fix your config - update under 'model' section:")
        print(f"  wav2vec2_model: \"{expected_model}\"")
        print()
        raise ValueError(f"Model must be {expected_model}, got {model_name}")

    # Initialize Wav2Vec2 extractor (ALL layers)
    print("[Extract] Initializing Wav2Vec2 extractor (ALL layers, raw sequences, no pooling)...")
    print(f"[Extract] Model: {model_name}")

    extractor = Wav2Vec2Extractor(
        model_name=model_name,
        device=device,
        pooling_method='mean'  # Initializer param (not used for all-layer extraction)
    )

    # Number of layers saved per file: embedding output + every transformer layer
    num_layers = extractor.get_num_layers()
    print(f"[Extract] Saving ALL {num_layers} layers per file "
          f"(stack shape: ({num_layers}, time, {extractor.get_embedding_dim()}))")

    # Get target sample rate
    target_sr = config['data'].get('sampling_rate', 16000)

    # Get embedding dimension
    embedding_dim = extractor.get_embedding_dim()

    # Statistics
    extracted = 0
    skipped = 0
    failed = 0

    # Track shapes for logging
    time_steps_list = []

    # Extract embeddings
    print(f"\n[Extract] Extracting RAW embeddings for {len(audio_files)} files...")
    print(f"[Extract] Target sample rate: {target_sr} Hz")
    print(f"[Extract] Embedding dimension: {embedding_dim}\n")

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"\n[Extract] Warning: File not found: {audio_path}")
                failed += 1
                continue

            # Generate cache filename
            cache_key = get_cache_key(audio_path)
            cache_file = cache_dir / f"{cache_key}.pt"

            # Skip if already extracted (unless force_reextract)
            if cache_file.exists() and not force_reextract:
                skipped += 1
                continue

            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample if needed
                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=target_sr
                    )
                    waveform = resampler(waveform)

                # Extract ALL layers (stacked, NO POOLING)
                # waveform shape: (1, samples) -> need (batch, samples)
                embedding = extractor.extract_all_layers(
                    waveform,
                    sample_rate=target_sr,
                )

                # embedding shape: (batch, num_layers, time_steps, embedding_dim)
                # Squeeze batch dimension: (num_layers, time_steps, embedding_dim)
                embedding = embedding.squeeze(0).cpu()

                # Store metadata about this sequence (time is dim 1 now)
                time_steps_list.append(embedding.shape[1])

                # Save to disk
                torch.save(embedding, cache_file)
                extracted += 1

            except Exception as e:
                print(f"\n[Extract] Error processing {audio_path}: {e}")
                failed += 1
                continue

    # Print statistics
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print("=" * 80)
    print(f"Total files: {len(audio_files)}")
    print(f"  - Extracted: {extracted}")
    print(f"  - Skipped (already exists): {skipped}")
    print(f"  - Failed: {failed}")
    print(f"\nEmbeddings saved to: {cache_dir}")
    print(f"Cache size: {sum(f.stat().st_size for f in cache_dir.glob('*.pt')) / (1024**2):.2f} MB")

    if time_steps_list:
        print(f"\nTemporal Sequence Statistics:")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Min time steps: {min(time_steps_list)}")
        print(f"  - Max time steps: {max(time_steps_list)}")
        print(f"  - Mean time steps: {sum(time_steps_list) / len(time_steps_list):.1f}")
        print(f"  - Output shape per file: ({num_layers}, time_steps, {embedding_dim})")

    print("=" * 80 + "\n")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'sample_rate': target_sr,
        'total_files': len(audio_files),
        'extracted': extracted,
        'skipped': skipped,
        'failed': failed,
        'format': 'all_layers_raw_sequences',
        'pooling': 'none (learnable softmax over all layers applied during training)',
        'output_shape_per_file': f'({num_layers}, time_steps, {embedding_dim})',
        'time_steps_min': min(time_steps_list) if time_steps_list else 0,
        'time_steps_max': max(time_steps_list) if time_steps_list else 0,
        'time_steps_mean': sum(time_steps_list) / len(time_steps_list) if time_steps_list else 0,
    }

    metadata_file = cache_dir / 'metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"[Extract] Metadata saved to: {metadata_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Pre-extract Wav2Vec2 embeddings')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction even if embeddings exist')

    args = parser.parse_args()

    # Load config
    print(f"[Extract] Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Extract] Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # Extract embeddings
    extract_and_save_embeddings(config, device=args.device, force_reextract=args.force)

    print("[Extract] Done! You can now start training with pre-extracted raw embeddings.")
    print("")
    print("IMPORTANT: These embeddings are RAW sequences (time_steps, embedding_dim)")
    print("Self-attention pooling will be applied during model training.")
    print("")
    print("[Extract] Make sure to set the following in your config:")
    print("  - use_preextracted_embeddings: true")
    print("  - wav2vec2_cache_dir: 'wav2vec2_embeddings_raw'  (or your custom directory)")
    print("")
    print("[Extract] The model will automatically apply self-attention pooling")
    print("         to each raw sequence during training.\n")


if __name__ == '__main__':
    main()
