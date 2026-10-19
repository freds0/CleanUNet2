"""Shared pytest setup: puts the repository root on sys.path and defines paths and skip markers.

Tests that need the official CleanUMamba checkpoint (../cleanumamba), the dataset or CUDA are skipped when those
are missing.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLEANUMAMBA_DIR = os.path.join(os.path.dirname(ROOT), "cleanumamba")
E8_CKPT = os.path.join(CLEANUMAMBA_DIR, "checkpoints", "models", "CleanUMamba-3N-E8-full.pkl")
DATA_DIR = "/home/fred/DATASETS/VoiceBank-DEMAND-16k"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def has_mamba_ssm():
    """True if the mamba_ssm CUDA kernels can be used (package installed and a GPU present)."""
    from cleanunet.sequence import HAS_MAMBA_SSM
    return HAS_MAMBA_SSM and torch.cuda.is_available()


requires_mamba_ssm = pytest.mark.skipif(not has_mamba_ssm(), reason="mamba_ssm CUDA kernels not available")
requires_e8 = pytest.mark.skipif(not os.path.exists(E8_CKPT), reason="CleanUMamba E8 checkpoint not found")
requires_data = pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason="VoiceBank-DEMAND not found")
