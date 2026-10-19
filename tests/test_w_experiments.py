"""Tests for the experiments of section 14 of IMPROVEMENT_PLAN.md: batched blend, compressed-magnitude loss, ensemble
distillation teachers, metric discriminator (manual optimization), test-time augmentation, weight soups, output
ensembles, robust PESQ and the low-SNR unseen-noise set."""

import argparse
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from conftest import DATA_DIR, ROOT, requires_data
from test_v_experiments import hybrid_arch, make_module, save_lightning, wave_arch

PY = sys.executable


def test_batched_blend_matches_the_single_file_blend(tmp_path):
    from cleanunet.blend import blend
    from cleanunet.cleanunet2 import CleanUNet2
    spec_model, _ = save_lightning(tmp_path, hybrid_arch(), seed=2)
    spec_model.eval()
    torch.manual_seed(0)
    x = torch.randn(3, 1, 2048) * torch.tensor([0.05, 0.2, 1.0]).view(3, 1, 1)
    with torch.no_grad():
        y = CleanUNet2(**wave_arch()).eval()(x)["wav"]
        batched = blend(y, x, spec_model, 0.3)
        single = torch.cat([blend(y[i:i + 1], x[i:i + 1], spec_model, 0.3) for i in range(3)])
    assert torch.allclose(batched, single, atol=1e-5)


def test_compressed_magnitude_loss_is_added_on_normalized_signals():
    from losses import CompressedMagnitudeLoss
    loss = CompressedMagnitudeLoss(n_fft=64, hop_length=16)
    x = torch.randn(2, 1, 2048)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-7)
    assert loss(x, 0.5 * x).item() > 0

    plain = make_module(wave_arch(), mode="waveform")
    cm = _module_with_loss(wave_arch(), {"mode": "waveform", "cmag": {"weight": 2.0, "n_fft": 64, "hop_length": 16}})
    noisy, clean, _ = _batch()
    plain.train()
    cm.train()
    l0, l1 = plain.training_step((noisy, clean, None), 0), cm.training_step((noisy, clean, None), 0)
    with torch.no_grad():
        out = cm.model(noisy)
        s = out["scale"]
        expected = 2.0 * CompressedMagnitudeLoss(64, 16)(out["wav"] / s, clean / s)
    assert torch.allclose(l1 - l0, expected, atol=1e-5)
    assert "model.cmag" not in "".join(cm.state_dict())             # no weights: checkpoints are unchanged


def _batch():
    torch.manual_seed(0)
    return torch.randn(2, 1, 2048) * 0.1, torch.randn(2, 1, 2048) * 0.1, ["a", "b"]


def _module_with_loss(arch, loss, **cfg):
    from lightning_modules.cleanunet_module import CleanUNetLightningModule
    torch.manual_seed(0)
    return CleanUNetLightningModule({"model": {"architecture": arch}, "loss": loss,
                                     "optim": {"schedule": "constant"}, **cfg})


def test_ensemble_teacher_is_the_weighted_mean_of_its_members(tmp_path):
    from cleanunet.blend import blend
    t1, p1 = save_lightning(tmp_path, wave_arch(), seed=1)
    t2, p2 = save_lightning(tmp_path, wave_arch(), seed=3)
    spec, ps = save_lightning(tmp_path, hybrid_arch(), seed=4)
    teachers = [{"checkpoint": p1}, {"checkpoint": p2, "blend_spec": ps, "blend_w": 0.5, "weight": 3.0}]
    kd = _module_with_loss(wave_arch(), {"mode": "waveform"}, distill={"weight": 1.0, "teachers": teachers})
    assert len(kd.teacher) == 3 and "teacher" not in "".join(kd.state_dict())
    x = _batch()[0]
    with torch.no_grad():
        expected = 0.25 * t1.eval()(x)["wav"] + 0.75 * blend(t2.eval()(x)["wav"], x, spec.eval(), 0.5)
    assert torch.allclose(kd._teacher_output(x), expected, atol=1e-5)
    with pytest.raises(ValueError):
        _module_with_loss(wave_arch(), {"mode": "waveform"}, distill={"weight": 1.0, "teachers": [{"blend_w": 1}]})


class _Pairs(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        clean = torch.randn(1, 2048, generator=g) * 0.1
        return clean + torch.randn(1, 2048, generator=g) * 0.05, clean, str(i)


def test_metric_gan_trains_the_generator_and_the_discriminator():
    import pytorch_lightning as pl
    module = _module_with_loss(wave_arch(), {"mode": "waveform",
                                             "metric_gan": {"weight": 0.5, "warmup_steps": 1, "pesq_workers": 1,
                                                            "disc": {"channels": 4, "n_fft": 64, "hop_length": 16}}},
                               trainer={"gradient_clip_val": 10.0})
    assert module.automatic_optimization is False
    calls = []

    def fake_pesq(ref, est, pool=None):          # PESQ cannot score 0.13 s of noise: a fixed score and a failure
        calls.append(ref.shape[0])
        return torch.tensor([0.6, float("nan")])

    module.pesq_fn = fake_pesq
    g0 = [p.detach().clone() for p in module.model.parameters()]
    d0 = [p.detach().clone() for p in module.metric_disc.parameters()]
    # with manual optimization every optimizer.step() counts as a trainer step: 3 batches = 6 steps (the runs of the
    # queue stop on max_epochs and the LR schedule is stepped once per batch, so this does not matter there)
    trainer = pl.Trainer(accelerator="cpu", max_steps=6, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                         num_sanity_val_steps=0)
    trainer.fit(module, torch.utils.data.DataLoader(_Pairs(), batch_size=2))
    assert int(module.mgan_step) == 3 and calls == [2, 2, 2]
    assert any(not torch.equal(a, b) for a, b in zip(g0, module.model.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(d0, module.metric_disc.parameters()))
    assert "mgan_step" in module.state_dict()
    assert not any(k.startswith("model.") and "metric_disc" in k for k in module.state_dict())


def test_discriminator_loss_skips_failed_scores():
    from cleanunet.metric_disc import MetricDiscriminator, discriminator_loss
    torch.manual_seed(0)
    d = MetricDiscriminator(channels=4, n_fft=64, hop_length=16).eval()       # eval: no dropout
    ref, est = torch.randn(2, 1, 2048), torch.randn(2, 1, 2048)
    mse = torch.nn.functional.mse_loss
    with torch.no_grad():
        clean_term = mse(d(ref, ref), torch.ones(2))
        nan = float("nan")
        assert torch.allclose(discriminator_loss(d, ref, est, torch.tensor([nan, nan])), clean_term)
        one = discriminator_loss(d, ref, est, torch.tensor([nan, 0.4]))
        assert torch.allclose(one, clean_term + mse(d(ref[1:], est[1:]), torch.tensor([0.4])))


@requires_data
def test_batch_pesq_scores_speech_and_flags_silence():
    from cleanunet.metric_disc import batch_pesq
    clean, _ = sf.read(os.path.join(DATA_DIR, "test/clean/p232_001.flac"), dtype="float32")
    noisy, _ = sf.read(os.path.join(DATA_DIR, "test/noisy/p232_001.flac"), dtype="float32")
    L = min(len(clean), 32768)
    ref = torch.from_numpy(np.stack([clean[:L], np.zeros(L, np.float32)]))
    est = torch.from_numpy(np.stack([noisy[:L], np.zeros(L, np.float32)]))
    s = batch_pesq(ref, est)
    assert 0.0 < s[0] < 1.05 and torch.isnan(s[1])


def _eval_args(**kw):
    base = dict(device="cpu", streaming=False, condition="normal", bf16=False, blend_model=None, blend_w=0.5,
                tta="none")
    base.update(kw)
    return argparse.Namespace(**base)


def test_tta_averages_the_realigned_outputs():
    from cleanunet.cleanunet2 import CleanUNet2
    from evaluate import TTA_SHIFTS, enhance
    torch.manual_seed(0)
    model = CleanUNet2(**wave_arch()).eval()
    x = torch.randn(3000) * 0.1
    f = lambda z: model(z.view(1, 1, -1))["wav"].view(-1)  # noqa: E731
    with torch.no_grad():
        assert torch.allclose(enhance(model, x, _eval_args()), f(x), atol=1e-6)
        flip = 0.5 * (f(x) - f(-x))
        assert torch.allclose(enhance(model, x, _eval_args(tta="flip")), flip, atol=1e-6)
        shifted = torch.stack([f(torch.nn.functional.pad(x, (s, 0)))[s:] for s in TTA_SHIFTS]).mean(0)
        assert torch.allclose(enhance(model, x, _eval_args(tta="shift")), shifted, atol=1e-6)
    with pytest.raises(ValueError):
        enhance(model, x, _eval_args(tta="flip", condition="shuffle"), cond=(x, x))


def test_soup_is_the_mean_of_the_weights_and_loads_in_evaluate(tmp_path):
    from evaluate import build_model
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_soup import soup
    m1, p1 = save_lightning(tmp_path, wave_arch(), seed=1)
    m2, p2 = save_lightning(tmp_path, wave_arch(), seed=2)
    out = str(tmp_path / "soup.ckpt")
    torch.save(soup([p1, p2]), out)
    args = argparse.Namespace(config=None, architecture=None, checkpoint=out, cleanumamba=None)
    model = build_model(args)
    for (k, a), b, c in zip(model.state_dict().items(), m1.state_dict().values(), m2.state_dict().values()):
        if a.is_floating_point():
            assert torch.allclose(a, (b + c) / 2, atol=1e-6), k
    _, p3 = save_lightning(tmp_path, hybrid_arch(), seed=3)
    with pytest.raises(ValueError):
        soup([p1, p3])


@requires_data
def test_ensemble_outputs_averages_the_audio_and_writes_evaluate_results(tmp_path):
    pairs = ["test/clean/p232_001.flac|test/noisy/p232_001.flac", "test/clean/p232_002.flac|test/noisy/p232_002.flac"]
    filelist = tmp_path / "list.csv"
    filelist.write_text("\n".join(pairs) + "\n")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for line in pairs:
        clean_rel, noisy_rel = line.split("|")
        name = os.path.basename(clean_rel)
        sf.write(a / name, sf.read(os.path.join(DATA_DIR, noisy_rel))[0], 16000)
        sf.write(b / name, sf.read(os.path.join(DATA_DIR, clean_rel))[0], 16000)
    out = tmp_path / "ens"
    subprocess.run([PY, os.path.join(ROOT, "scripts/ensemble_outputs.py"), "--inputs", str(a), str(b), "--weights",
                    "1", "3", "--filelist", str(filelist), "--out_dir", str(out), "--fast", "--robust_pesq",
                    "--save_audio", "--workers", "1"], check=True, cwd=ROOT, capture_output=True)
    for name in ("p232_001.flac", "p232_002.flac"):
        avg = sf.read(out / name)[0]
        expected = 0.25 * sf.read(a / name)[0] + 0.75 * sf.read(b / name)[0]
        assert np.abs(avg - expected).max() < 1e-4
    import json
    s = json.load(open(out / "summary.json"))["summary"]
    assert s["n_files"] == 2 and s["pesq_wb"] > 3.0 and "pesq_wb_r" in s


@requires_data
def test_robust_pesq_with_a_single_pad_is_pesq():
    from evaluation.metrics import evaluate_pair, robust_pesq_wb
    clean, _ = sf.read(os.path.join(DATA_DIR, "test/clean/p232_001.flac"))
    noisy, _ = sf.read(os.path.join(DATA_DIR, "test/noisy/p232_001.flac"))
    m = evaluate_pair(clean, noisy, full=False, robust=True)
    assert robust_pesq_wb(clean, noisy, pads=(0,)) == pytest.approx(m["pesq_wb"])
    assert abs(m["pesq_wb_r"] - m["pesq_wb"]) < 0.5


def test_low_snr_mixtures_are_rescaled_with_their_reference(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_val_unseen as mvu
    from noise_bank import active_speech_power
    data, demand = tmp_path / "data", tmp_path / "demand"
    (data / "train/clean").mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    speech = 0.9 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    sf.write(data / "train/clean/x.flac", speech, 16000, subtype="PCM_16")
    for env in mvu.DEMAND_UNUSED:
        (demand / env).mkdir(parents=True)
        sf.write(demand / env / "ch01.wav", 0.3 * rng.standard_normal(64000), 16000)
    filelist = tmp_path / "pairs.csv"
    filelist.write_text("train/clean/x.flac|train/noisy/x.flac\n")
    args = argparse.Namespace(data_dir=str(data), demand_dir=str(demand), filelist=str(filelist), name="low",
                              out_dir=str(tmp_path), snrs=[-5.0], sample_rate=16000, seed=0, rescale=False)
    with pytest.raises(SystemExit):
        mvu.make(args)
    args.rescale = True
    mvu.make(args)
    clean_rel, noisy_rel = (tmp_path / "low.csv").read_text().strip().split("|")
    assert clean_rel == "low/clean/x.flac"
    clean = sf.read(data / clean_rel)[0]
    noisy = sf.read(data / noisy_rel)[0]
    assert np.abs(noisy).max() < 1.0 and np.abs(clean).max() < 0.9
    snr = 10 * np.log10(active_speech_power(clean, 16000) / np.mean((noisy - clean) ** 2))
    assert abs(snr + 5.0) < 0.2


def test_random_sign_flips_talkers_and_excerpts_and_is_off_by_default(tmp_path):
    import random
    from noise_bank import NoiseBank
    from test_noise_bank import fake_musan
    root = fake_musan(tmp_path)
    excerpt = lambda sub, length: np.linspace(0.1, 1.0, length, dtype=np.float32)   # noqa: E731  (asymmetric)
    for kind, weights in (("noise", {"noise": 1}), ("babble", {"babble": 1})):
        plain = NoiseBank(root, weights=weights, babble_talkers=(1, 1))
        signed = NoiseBank(root, weights=weights, babble_talkers=(1, 1), random_sign=True)
        plain._excerpt = signed._excerpt = excerpt
        random.seed(0)
        assert all(plain.sample(100)[0].sum() > 0 for _ in range(20))               # default: polarity kept
        signs = {float(np.sign(signed.sample(100)[0].sum())) for _ in range(40)}
        assert signs == {-1.0, 1.0}, kind


def test_polarity_augmentation_flips_the_pair_and_the_teacher_target(tmp_path):
    teacher, path = save_lightning(tmp_path, wave_arch(), seed=1)
    kw = dict(distill={"teacher_checkpoint": path, "weight": 0.5})
    flipped = _module_with_loss(wave_arch(), {"mode": "waveform"}, augment_polarity=1.0, **kw)
    plain = _module_with_loss(wave_arch(), {"mode": "waveform"}, **kw)
    noisy, clean, _ = _batch()
    flipped.train()
    plain.train()
    loss = flipped.training_step((noisy, clean, None), 0)
    with torch.no_grad():
        student = flipped.model(-noisy)["wav"]
        target = -teacher.eval()(noisy)["wav"]               # the teacher sees the original polarity
        expected = flipped.criterion(-clean, student) + 0.5 * flipped.criterion(target, student)
    assert torch.allclose(loss, expected, atol=1e-5)
    assert torch.allclose(plain.training_step((noisy, clean, None), 0),
                          plain._compute_losses(noisy, clean)[0], atol=1e-6)
    flipped.eval()
    assert flipped._flip_polarity(noisy, clean)[2] is None                         # never at evaluation
    with pytest.raises(ValueError):
        _module_with_loss(wave_arch(), {"mode": "waveform"}, augment_polarity=1.5)


def test_invert_evaluates_the_polarity_inverted_input():
    from cleanunet.cleanunet2 import CleanUNet2
    from evaluate import enhance
    torch.manual_seed(0)
    model = CleanUNet2(**wave_arch()).eval()
    x = torch.randn(3000) * 0.1
    with torch.no_grad():
        expected = -model(-x.view(1, 1, -1))["wav"].view(-1)
        assert torch.allclose(enhance(model, x, _eval_args(invert=True)), expected, atol=1e-6)
