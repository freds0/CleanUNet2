"""Compatibility with CleanUMamba: checkpoint loading, numerical equivalence and streaming."""

import functools
import os
import sys

import pytest
import torch

from cleanunet.cleanunet2 import CleanUNet2
from config_utils import resolve_architecture
from conftest import CLEANUMAMBA_DIR, DATA_DIR, E8_CKPT, ROOT, requires_data, requires_e8, requires_mamba_ssm


def e8_model():
    arch = resolve_architecture(os.path.join(ROOT, "configs/model/cleanunet_mamba_e8.yaml"))
    model = CleanUNet2(**arch)
    model.load_cleanumamba_weights(E8_CKPT)
    return model


def test_no_missing_or_unexpected_keys_for_e8():
    if not os.path.exists(E8_CKPT):
        pytest.skip("CleanUMamba E8 checkpoint not found")
    arch = resolve_architecture(os.path.join(ROOT, "configs/model/cleanunet_mamba_e8.yaml"))
    arch["cleanunet_params"]["bottleneck"]["backend"] = "torch"   # works without CUDA
    model = CleanUNet2(**arch)
    missing, unexpected = model.clean_unet.load_state_dict(
        {k: v.float() for k, v in torch.load(E8_CKPT, weights_only=True)["model_state_dict"].items()}, strict=True)
    assert not missing and not unexpected


@requires_e8
@requires_mamba_ssm
def test_matches_original_cleanumamba_implementation():
    """Same weights + same input -> same output as the CleanUMamba repository code.

    The original code targets mamba-ssm 1.2.2; with mamba-ssm 2.x `create_block` needs
    `d_intermediate=0`, which is patched here only for the test.
    """
    pytest.importorskip("torchinfo")
    sys.path.insert(0, CLEANUMAMBA_DIR)
    try:
        import mamba_ssm.models.mixer_seq_simple as mss
        import src.network.CleanUMamba as cm
        cm.create_block = functools.partial(mss.create_block, d_intermediate=0)
        ckpt = torch.load(E8_CKPT, weights_only=True)
        original = cm.CleanUMamba(**ckpt["network_config"], device="cuda")
        original.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})
    finally:
        sys.path.remove(CLEANUMAMBA_DIR)
    ours = e8_model().cuda().eval()
    original.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 1, 48000, device="cuda") * 0.1
    with torch.no_grad():
        ref = original(x.clone())          # the original divides its input in place
        out = ours(x)["wav"]
    assert (ref - out).abs().max() < 1e-4 * ref.abs().max()


@requires_e8
@pytest.mark.parametrize("chunk_size", [256, 1000, 4096])
def test_streaming_is_exact_in_float64(chunk_size):
    """Frame-by-frame streaming reproduces offline inference exactly (float64, pure-PyTorch Mamba).

    The last `frame_length` samples are excluded: offline inference truncates the signal at the
    padded length, while streaming zero-extends it by one frame (both are valid, they differ only
    within the lookahead window at the very end).
    """
    from cleanunet.streaming import stream_denoise
    arch = resolve_architecture(os.path.join(ROOT, "configs/model/cleanunet_mamba_e8.yaml"))
    arch["cleanunet_params"]["bottleneck"]["backend"] = "torch"
    model = CleanUNet2(**arch)
    model.load_cleanumamba_weights(E8_CKPT)
    net = model.clean_unet.eval().double()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 6000, dtype=torch.float64) * 0.1
    with torch.no_grad():
        offline = net(x).view(1, -1)
    streamed = stream_denoise(net, x.view(1, -1), chunk_size=chunk_size, normalize_input=False)
    keep = x.shape[-1] - net.valid_length(1)
    assert streamed.shape == offline.shape
    assert (streamed - offline)[:, :keep].abs().max() < 1e-9


@requires_e8
@requires_mamba_ssm
@requires_data
def test_streaming_cuda_fp32_on_speech():
    """With the CUDA kernels in float32 the streaming output matches offline inference on real speech."""
    from cleanunet.streaming import stream_denoise
    from dataset import PairDataset
    net = e8_model().clean_unet.cuda().eval()
    noisy, _, _ = PairDataset(DATA_DIR, os.path.join(ROOT, "filelists/test.csv"), max_files=1)[0]
    x = noisy.cuda().view(1, 1, -1)
    x = x / (x.std() + 1e-3)                        # CleanUNet2 normalizes outside CleanUNet
    with torch.no_grad():
        offline = net(x).view(1, -1)
    streamed = stream_denoise(net, x.view(1, -1), chunk_size=1024, normalize_input=False)
    keep = x.shape[-1] - net.valid_length(1)
    err = (streamed - offline)[:, :keep]
    snr_db = 10 * torch.log10(offline[:, :keep].pow(2).sum() / err.pow(2).sum())
    assert snr_db > 40, f"streaming/offline SNR {snr_db:.1f} dB"


@requires_e8
@requires_mamba_ssm
@requires_data
def test_zero_shot_e8_improves_voicebank_test_files():
    """The DNS-trained CleanUMamba-E8 must improve PESQ over the noisy input on VoiceBank test files."""
    from dataset import PairDataset
    from evaluation.metrics import evaluate_many, summarize
    model = e8_model().cuda().eval()
    ds = PairDataset(DATA_DIR, os.path.join(ROOT, "filelists/test.csv"), max_files=40)
    noisy_pairs, enhanced_pairs = [], []
    for noisy, clean, _ in ds:
        with torch.no_grad():
            enhanced = model(noisy.cuda().unsqueeze(0))["wav"].view(-1).cpu()
        noisy_pairs.append((clean.view(-1).numpy(), noisy.view(-1).numpy()))
        enhanced_pairs.append((clean.view(-1).numpy(), enhanced.numpy()))
    noisy_m = summarize(evaluate_many(noisy_pairs, full=False, num_workers=8))
    enh_m = summarize(evaluate_many(enhanced_pairs, full=False, num_workers=8))
    assert enh_m["pesq_wb"] > noisy_m["pesq_wb"] + 0.5
    assert enh_m["stoi"] > noisy_m["stoi"]
