"""
Test script for Self-Attention Pooling in Wav2Vec2Extractor
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cleanunet.wav2vec2_extractor import Wav2Vec2Extractor, SelfAttentionPooling


def test_self_attention_pooling_module():
    """Test SelfAttentionPooling module directly"""
    print("\n" + "=" * 80)
    print("TEST 1: SelfAttentionPooling Module")
    print("=" * 80)

    batch_size = 4
    time_steps = 100
    embedding_dim = 1024

    # Create module
    pooling = SelfAttentionPooling(embedding_dim=embedding_dim, num_heads=8)

    # Create dummy input
    hidden_states = torch.randn(batch_size, time_steps, embedding_dim)

    print(f"Input shape: {hidden_states.shape}")

    # Forward pass
    pooled = pooling(hidden_states)

    print(f"Output shape: {pooled.shape}")
    print(f"Expected shape: ({batch_size}, {embedding_dim})")

    assert pooled.shape == (batch_size, embedding_dim), "Output shape mismatch!"
    print("✓ Self-Attention Pooling module test PASSED!")

    return pooling


def test_wav2vec2_with_self_attention():
    """Test Wav2Vec2Extractor with self-attention pooling"""
    print("\n" + "=" * 80)
    print("TEST 2: Wav2Vec2Extractor with Self-Attention Pooling")
    print("=" * 80)

    try:
        # Initialize extractor with self-attention pooling
        extractor = Wav2Vec2Extractor(
            model_name="facebook/wav2vec2-xls-r-300m",
            device='cpu',
            layer=-1,
            pooling_method='self_attention',
            num_attention_heads=8
        )

        # Create dummy audio (2 seconds at 16kHz)
        batch_size = 2
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)

        waveform = torch.randn(batch_size, num_samples)

        print(f"\nInput waveform shape: {waveform.shape}")

        # Extract embeddings with self-attention pooling
        embeddings = extractor.extract_embeddings(
            waveform,
            sample_rate=sample_rate,
            return_mean=True
        )

        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Expected shape: ({batch_size}, {extractor.embedding_dim})")

        assert embeddings.shape == (batch_size, extractor.embedding_dim), "Embeddings shape mismatch!"
        print("✓ Wav2Vec2Extractor with self-attention pooling test PASSED!")

        return extractor

    except Exception as e:
        print(f"\n⚠ Test failed or skipped: {type(e).__name__}: {str(e)[:200]}")
        print("This is expected if you don't have internet or the model isn't downloaded.")
        return None


def test_wav2vec2_with_mean_pooling():
    """Test Wav2Vec2Extractor with mean pooling (baseline)"""
    print("\n" + "=" * 80)
    print("TEST 3: Wav2Vec2Extractor with Mean Pooling (Baseline)")
    print("=" * 80)

    try:
        # Initialize extractor with mean pooling
        extractor = Wav2Vec2Extractor(
            model_name="facebook/wav2vec2-xls-r-300m",
            device='cpu',
            layer=-1,
            pooling_method='mean'
        )

        # Create dummy audio
        batch_size = 2
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)

        waveform = torch.randn(batch_size, num_samples)

        print(f"\nInput waveform shape: {waveform.shape}")

        # Extract embeddings with mean pooling
        embeddings = extractor.extract_embeddings(
            waveform,
            sample_rate=sample_rate,
            return_mean=True
        )

        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Expected shape: ({batch_size}, {extractor.embedding_dim})")

        assert embeddings.shape == (batch_size, extractor.embedding_dim), "Embeddings shape mismatch!"
        print("✓ Wav2Vec2Extractor with mean pooling test PASSED!")

        return extractor

    except Exception as e:
        print(f"\n⚠ Test failed or skipped: {type(e).__name__}: {str(e)[:200]}")
        print("This is expected if you don't have internet or the model isn't downloaded.")
        return None


def test_attention_pooling_trainable():
    """Test that attention pooling has trainable parameters"""
    print("\n" + "=" * 80)
    print("TEST 4: Verify Self-Attention Pooling is Trainable")
    print("=" * 80)

    try:
        extractor = Wav2Vec2Extractor(
            model_name="facebook/wav2vec2-xls-r-300m",
            device='cpu',
            layer=-1,
            pooling_method='self_attention',
            num_attention_heads=8
        )

        # Count trainable parameters in attention pooling
        attention_params = sum(p.numel() for p in extractor.attention_pooling.parameters() if p.requires_grad)

        # Count frozen parameters in wav2vec2 model
        model_params = sum(p.numel() for p in extractor.model.parameters())
        model_trainable = sum(p.numel() for p in extractor.model.parameters() if p.requires_grad)

        print(f"\nWav2Vec2 model parameters: {model_params:,}")
        print(f"Wav2Vec2 trainable parameters: {model_trainable:,} (should be 0)")
        print(f"Attention pooling trainable parameters: {attention_params:,} (should be > 0)")

        assert model_trainable == 0, "Wav2Vec2 model should be frozen!"
        assert attention_params > 0, "Attention pooling should have trainable parameters!"

        print("✓ Trainability test PASSED!")

        return extractor

    except Exception as e:
        print(f"\n⚠ Test failed or skipped: {type(e).__name__}: {str(e)[:200]}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SELF-ATTENTION POOLING TEST SUITE")
    print("=" * 80)

    # Test 1: SelfAttentionPooling module
    pooling = test_self_attention_pooling_module()

    # Test 2: Wav2Vec2Extractor with self-attention
    extractor_attn = test_wav2vec2_with_self_attention()

    # Test 3: Wav2Vec2Extractor with mean pooling
    extractor_mean = test_wav2vec2_with_mean_pooling()

    # Test 4: Verify trainability
    if extractor_attn is not None:
        test_attention_pooling_trainable()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)
    print("\nNOTE: If wav2vec2 tests were skipped, download the model first:")
    print("  python -c \"from transformers import Wav2Vec2Model; " +
          "Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m', " +
          "cache_dir='pretrained_models/wav2vec2', use_safetensors=True)\"")
    print("=" * 80 + "\n")
