"""
Test script for Wav2Vec2 embedding system.

This script validates the wav2vec2 extraction, caching, and loading functionality.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from cleanunet.wav2vec2_extractor import Wav2Vec2Extractor
from cleanunet.wav2vec2_cache import Wav2Vec2Cache
import tempfile


def test_wav2vec2_extractor():
    """Test 1: Wav2Vec2 extractor basic functionality"""
    print("=" * 80)
    print("TEST 1: Wav2Vec2 Extractor")
    print("=" * 80)

    print("\n1. Initializing Wav2Vec2 extractor...")
    try:
        extractor = Wav2Vec2Extractor(
            model_name='facebook/wav2vec2-xls-r-300m',
            device='cpu',
            layer=-1
        )
        print(f"   ✓ Extractor initialized")
        print(f"   ✓ Embedding dimension: {extractor.get_embedding_dim()}")
    except Exception as e:
        print(f"   ✗ Failed to initialize extractor: {e}")
        return False

    print("\n2. Creating dummy audio...")
    # Create dummy audio (1 second at 16kHz)
    dummy_audio = torch.randn(1, 16000)
    print(f"   ✓ Dummy audio shape: {dummy_audio.shape}")

    print("\n3. Extracting embeddings...")
    try:
        embedding = extractor.extract_embeddings(
            dummy_audio,
            sample_rate=16000,
            return_mean=True
        )
        print(f"   ✓ Embedding extracted")
        print(f"   ✓ Embedding shape: {embedding.shape}")

        # Verify shape
        if embedding.shape == (1, 1024):
            print(f"   ✓ Embedding shape is correct!")
        else:
            print(f"   ✗ Unexpected embedding shape: {embedding.shape}")
            return False

    except Exception as e:
        print(f"   ✗ Failed to extract embeddings: {e}")
        return False

    print("\n4. Testing interpolation...")
    try:
        embedding_interp = extractor.extract_and_interpolate(
            dummy_audio,
            target_length=100,
            sample_rate=16000
        )
        print(f"   ✓ Interpolated embedding shape: {embedding_interp.shape}")

        if embedding_interp.shape == (1, 1024, 100):
            print(f"   ✓ Interpolation works correctly!")
        else:
            print(f"   ✗ Unexpected interpolated shape: {embedding_interp.shape}")
            return False

    except Exception as e:
        print(f"   ✗ Failed interpolation: {e}")
        return False

    print("\n✅ TEST 1 PASSED!\n")
    return True


def test_wav2vec2_cache():
    """Test 2: Wav2Vec2 cache functionality"""
    print("=" * 80)
    print("TEST 2: Wav2Vec2 Cache")
    print("=" * 80)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        print(f"\n1. Creating cache in: {cache_dir}")

        # Create some dummy embeddings
        dummy_paths = [
            "/path/to/audio1.wav",
            "/path/to/audio2.wav",
            "/path/to/audio3.wav"
        ]

        print(f"\n2. Saving {len(dummy_paths)} dummy embeddings...")
        for i, path in enumerate(dummy_paths):
            embedding = torch.randn(1024)  # 1024-dim embedding
            # Save directly to cache directory
            import hashlib
            cache_key = hashlib.md5(path.encode('utf-8')).hexdigest()
            cache_file = cache_dir / f"{cache_key}.pt"
            torch.save(embedding, cache_file)

        print(f"   ✓ Saved {len(dummy_paths)} embeddings")

        print(f"\n3. Initializing cache...")
        cache = Wav2Vec2Cache(cache_dir=str(cache_dir), enabled=True)
        print(f"   ✓ Cache initialized")

        print(f"\n4. Testing cache loading...")
        hits = 0
        misses = 0

        for path in dummy_paths:
            embedding = cache.get(path, device='cpu')
            if embedding is not None:
                hits += 1
                print(f"   ✓ Loaded: {Path(path).name} - Shape: {embedding.shape}")
            else:
                misses += 1
                print(f"   ✗ Miss: {Path(path).name}")

        print(f"\n5. Cache statistics:")
        print(f"   - Hits: {hits}")
        print(f"   - Misses: {misses}")

        if hits == len(dummy_paths) and misses == 0:
            print(f"   ✓ All embeddings loaded successfully!")
        else:
            print(f"   ✗ Some embeddings missing!")
            return False

        print(f"\n6. Testing cache miss (non-existent file)...")
        missing_embedding = cache.get("/path/to/nonexistent.wav", device='cpu')
        if missing_embedding is None:
            print(f"   ✓ Cache miss handled correctly")
        else:
            print(f"   ✗ Unexpected cache hit for non-existent file")
            return False

        print(f"\n7. Cache statistics:")
        cache.print_stats()

    print("\n✅ TEST 2 PASSED!\n")
    return True


def test_integration():
    """Test 3: Integration test with model"""
    print("=" * 80)
    print("TEST 3: Model Integration")
    print("=" * 80)

    print("\n1. Importing model...")
    try:
        from cleanunet.cleanunet2_with_xvector import CleanUNet2WithXVector
        print("   ✓ Model imported")
    except Exception as e:
        print(f"   ✗ Failed to import model: {e}")
        return False

    print("\n2. Creating model with wav2vec2 (on-the-fly extraction)...")
    print("   Note: This will download wav2vec2 model if not cached")
    try:
        model = CleanUNet2WithXVector(
            stage='stage1',
            use_xvector=False,
            use_wav2vec2=True,
            wav2vec2_model='facebook/wav2vec2-xls-r-300m',
            use_preextracted_embeddings=False,  # Extract on-the-fly for testing
            cleanunet_params={
                'channels_input': 1,
                'channels_output': 1,
                'channels_H': 64,
                'max_H': 768,
                'encoder_n_layers': 8
            },
            cleanspecnet_params={
                'input_channels': 513,
                'num_conv_layers': 5
            }
        )
        print("   ✓ Model created with wav2vec2 support")
        print(f"   ✓ Embedding type: {model.embedding_type}")
        print(f"   ✓ Embedding dim: {model.embedding_dim}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        print(f"   Note: This may fail if wav2vec2 model cannot be downloaded")
        return False

    print("\n3. Testing forward pass (dummy data)...")
    try:
        # Create dummy inputs
        batch_size = 2
        samples = 16000
        freq_bins = 513
        time_steps = 64

        noisy_waveform = torch.randn(batch_size, 1, samples)
        noisy_spec = torch.randn(batch_size, freq_bins, time_steps)
        clean_audio = torch.randn(batch_size, 1, samples)

        # Forward pass
        enhanced_waveform, enhanced_spec = model(
            noisy_waveform,
            noisy_spec,
            clean_audio=clean_audio
        )

        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Enhanced waveform shape: {enhanced_waveform.shape}")
        print(f"   ✓ Enhanced spec shape: {enhanced_spec.shape}")

    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ TEST 3 PASSED!\n")
    return True


def print_summary(results):
    """Print test summary"""
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    test_names = [
        "Wav2Vec2 Extractor",
        "Wav2Vec2 Cache",
        "Model Integration"
    ]

    all_passed = True
    for name, result in zip(test_names, results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:30s} {status}")
        if not result:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!\n")
        print("The Wav2Vec2 embedding system is working correctly.")
        print("\nNext steps:")
        print("1. Run: python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml")
        print("2. Then: python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1")
    else:
        print("\n⚠️  SOME TESTS FAILED\n")
        print("Please check the error messages above.")

    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("WAV2VEC2 EMBEDDING SYSTEM TEST SUITE")
    print("=" * 80)
    print("\nThis script tests the wav2vec2 embedding extraction and caching system.\n")

    results = []

    try:
        # Test 1: Extractor
        results.append(test_wav2vec2_extractor())

        # Test 2: Cache
        results.append(test_wav2vec2_cache())

        # Test 3: Integration
        results.append(test_integration())

        # Print summary
        print_summary(results)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
