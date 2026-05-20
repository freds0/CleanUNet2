#!/usr/bin/env python3
"""
Script to test and validate the embedding cache system.

Usage:
    python scripts/test_embedding_cache.py \
        --cache_dir wav2vec2_embeddings_stage1_cache \
        --file_list filelists/train.csv
"""

import argparse
import torch
from pathlib import Path
import hashlib
import pandas as pd
from collections import Counter


def get_file_hash(filename):
    """Get MD5 hash of filename (same as used in cache)."""
    return hashlib.md5(filename.encode()).hexdigest()


def test_cache(cache_dir, file_list):
    """Test and validate embedding cache."""

    print("=" * 80)
    print("Embedding Cache Validation")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print(f"File list: {file_list}")
    print()

    cache_dir = Path(cache_dir)

    # Check if cache exists
    if not cache_dir.exists():
        print(f"❌ ERROR: Cache directory does not exist: {cache_dir}")
        return False

    # Load file list
    if file_list.endswith('.csv'):
        df = pd.read_csv(file_list)
        expected_files = df['clean'].tolist()
    else:
        with open(file_list, 'r') as f:
            expected_files = [line.strip() for line in f]

    print(f"✓ Expected files in list: {len(expected_files)}")

    # Find cached embeddings
    cached_files = list(cache_dir.glob('*.pt'))
    print(f"✓ Cached embeddings found: {len(cached_files)}")
    print()

    if len(cached_files) == 0:
        print("❌ ERROR: No cached embeddings found!")
        print("   Run: bash scripts/preextract_all_embeddings.sh")
        return False

    # Load and validate embeddings
    print("Validating embeddings...")

    valid_count = 0
    invalid_count = 0
    missing_files = []
    extra_files = []

    # Check each expected file
    expected_hashes = {}
    for filename in expected_files:
        file_hash = get_file_hash(filename)
        expected_hashes[file_hash] = filename

        cache_path = cache_dir / f"{file_hash}.pt"

        if not cache_path.exists():
            missing_files.append(filename)
            continue

        try:
            # Load embedding
            data = torch.load(cache_path, map_location='cpu')

            # Validate structure
            if 'embedding' not in data:
                print(f"  ⚠️  WARNING: Missing 'embedding' key in {cache_path.name}")
                invalid_count += 1
                continue

            if 'filename' not in data:
                print(f"  ⚠️  WARNING: Missing 'filename' key in {cache_path.name}")
                invalid_count += 1
                continue

            # Validate embedding shape
            embedding = data['embedding']
            if not isinstance(embedding, torch.Tensor):
                print(f"  ⚠️  WARNING: Embedding is not a tensor in {cache_path.name}")
                invalid_count += 1
                continue

            if embedding.dim() != 1:
                print(f"  ⚠️  WARNING: Embedding has wrong dimensions {embedding.shape} in {cache_path.name}")
                invalid_count += 1
                continue

            valid_count += 1

        except Exception as e:
            print(f"  ❌ ERROR loading {cache_path.name}: {e}")
            invalid_count += 1

    # Check for extra files (not in file list)
    cached_hashes = {f.stem for f in cached_files}
    extra_hashes = cached_hashes - set(expected_hashes.keys())

    if extra_hashes:
        # Try to load and get filenames
        for file_hash in list(extra_hashes)[:5]:  # Show first 5
            cache_path = cache_dir / f"{file_hash}.pt"
            try:
                data = torch.load(cache_path, map_location='cpu')
                if 'filename' in data:
                    extra_files.append(data['filename'])
            except:
                pass

    # Print statistics
    print()
    print("=" * 80)
    print("Validation Results")
    print("=" * 80)
    print(f"✓ Valid embeddings: {valid_count}")
    print(f"✗ Invalid embeddings: {invalid_count}")
    print(f"⊘ Missing embeddings: {len(missing_files)}")
    print(f"+ Extra embeddings: {len(extra_hashes)}")
    print()

    # Coverage
    coverage = (valid_count / len(expected_files)) * 100 if expected_files else 0
    print(f"Coverage: {coverage:.1f}% ({valid_count}/{len(expected_files)})")
    print()

    # Show embedding details (from first valid one)
    if valid_count > 0:
        first_valid = None
        for filename in expected_files[:10]:
            file_hash = get_file_hash(filename)
            cache_path = cache_dir / f"{file_hash}.pt"
            if cache_path.exists():
                try:
                    first_valid = torch.load(cache_path, map_location='cpu')
                    break
                except:
                    pass

        if first_valid:
            print("Sample Embedding Details:")
            print(f"  Filename: {first_valid.get('filename', 'N/A')}")
            print(f"  Model: {first_valid.get('model', 'N/A')}")
            print(f"  Pooling: {first_valid.get('pooling_method', 'N/A')}")
            print(f"  Shape: {first_valid['embedding'].shape}")
            print(f"  Dtype: {first_valid['embedding'].dtype}")
            print(f"  Min/Max: [{first_valid['embedding'].min():.4f}, {first_valid['embedding'].max():.4f}]")
            print()

    # Show missing files (first 10)
    if missing_files:
        print("Missing Embeddings (first 10):")
        for filename in missing_files[:10]:
            print(f"  - {filename}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        print()

    # Show extra files (first 10)
    if extra_files:
        print("Extra Embeddings (not in file list, first 10):")
        for filename in extra_files[:10]:
            print(f"  + {filename}")
        if len(extra_hashes) > 10:
            print(f"  ... and {len(extra_hashes) - 10} more")
        print()

    # Disk usage
    total_size = sum(f.stat().st_size for f in cached_files)
    avg_size = total_size / len(cached_files) if cached_files else 0

    print("Disk Usage:")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Average per file: {avg_size / 1024:.1f} KB")
    print()

    # Final verdict
    print("=" * 80)
    if coverage >= 99.0 and invalid_count == 0:
        print("✓ CACHE IS VALID AND READY TO USE")
        print("=" * 80)
        return True
    elif coverage >= 95.0:
        print("⚠️  CACHE IS MOSTLY VALID")
        print("   Some embeddings are missing or invalid.")
        print("   Consider re-extracting: bash scripts/preextract_all_embeddings.sh")
        print("=" * 80)
        return True
    else:
        print("❌ CACHE HAS ISSUES")
        print("   Please re-extract embeddings:")
        print("   1. rm -rf {cache_dir}")
        print("   2. bash scripts/preextract_all_embeddings.sh")
        print("=" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(description='Test and validate embedding cache')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Directory containing cached embeddings')
    parser.add_argument('--file_list', type=str, required=True,
                        help='CSV file with expected audio files')

    args = parser.parse_args()

    success = test_cache(args.cache_dir, args.file_list)

    exit(0 if success else 1)


if __name__ == '__main__':
    main()
