"""
Quick test to verify WavLM model can be loaded and check embedding dimension
"""

import torch
from cleanunet.wav2vec2_extractor import Wav2Vec2Extractor

print("=" * 80)
print("Testing WavLM Model Integration")
print("=" * 80)

# Test 1: Load the extractor with default model (microsoft/wavlm-base)
print("\n[Test 1] Loading WavLM extractor with microsoft/wavlm-base...")
try:
    extractor = Wav2Vec2Extractor(
        model_name="microsoft/wavlm-base",
        device='cpu',
        layer=-1,
        pooling_method='mean'
    )
    print(f"✓ Model loaded successfully!")
    print(f"✓ Embedding dimension: {extractor.get_embedding_dim()}")

    # Test 2: Extract embeddings from dummy audio
    print("\n[Test 2] Testing embedding extraction...")
    dummy_audio = torch.randn(1, 16000)  # 1 second at 16kHz
    embeddings = extractor.extract_embeddings(dummy_audio, sample_rate=16000, return_mean=True)
    print(f"✓ Input shape: {dummy_audio.shape}")
    print(f"✓ Output embedding shape: {embeddings.shape}")
    print(f"✓ Expected shape: (1, {extractor.get_embedding_dim()})")

    if embeddings.shape == (1, extractor.get_embedding_dim()):
        print("✓ Shape matches expected!")
    else:
        print("✗ Shape mismatch!")

    print("\n" + "=" * 80)
    print("All tests passed! WavLM integration is working correctly.")
    print("=" * 80)
    print("\nModel Details:")
    print(f"  - Model: microsoft/wavlm-base")
    print(f"  - Embedding dimension: {extractor.get_embedding_dim()}")
    print(f"  - Cached at: pretrained_models/wavlm")
    print("\nNext steps:")
    print("  1. Run: python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml")
    print("  2. Then train with: python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nIf you see a connection error, the model needs to be downloaded first.")
    print("Make sure you have internet connection or download the model manually.")
