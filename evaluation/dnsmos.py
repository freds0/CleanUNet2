#!/usr/bin/env python
"""
DNSMOS (Reddy et al., ICASSP 2021/2022): non-intrusive MOS prediction with the ONNX models of the Microsoft
DNS-Challenge repository (DNSMOS/DNSMOS/sig_bak_ovr.onnx: P.835 SIG / BAK / OVRL; DNSMOS/DNSMOS/model_v8.onnx: P.808).
A re-implementation of DNSMOS/dnsmos_local.py (non-personalized), so that no downloaded code runs: 9.01 s windows
(shorter clips are repeated), 1 s hop, the P.835 polynomial calibration, the P.808 log-mel input.

A second opinion next to PESQ, which a metric discriminator can exploit (IMPROVEMENT_PLAN.md, 14.7.2 / 14.8).
The model files are downloads, so this script is standalone (numpy, soundfile, librosa, onnxruntime) and is meant to be
run with `python -I`, with the model directory passed as an argument.

Usage:
    python -I evaluation/dnsmos.py --models /home/fred/DATASETS/DNSMOS_dl_20261016 \
        --audio_dir results/abl/X_audio/S30_soup --names filelists/test.csv --out results/abl/X_dnsmos/S30_soup.csv
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SR = 16000
INPUT_LENGTH = 9.01
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

_SESS = {}


def _sessions(models_dir):
    if "p835" not in _SESS:
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        _SESS["p835"] = ort.InferenceSession(os.path.join(models_dir, "sig_bak_ovr.onnx"), opt,
                                             providers=["CPUExecutionProvider"])
        _SESS["p808"] = ort.InferenceSession(os.path.join(models_dir, "model_v8.onnx"), opt,
                                             providers=["CPUExecutionProvider"])
    return _SESS["p835"], _SESS["p808"]


def _melspec(audio):
    import librosa
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


def score(audio: np.ndarray, models_dir: str) -> dict:
    """DNSMOS of one 16 kHz clip: SIG, BAK, OVRL (P.835, calibrated) and P808 (means over the 9.01 s windows)."""
    p835, p808 = _sessions(models_dir)
    audio = np.asarray(audio, dtype=np.float64)
    n = int(INPUT_LENGTH * SR)
    while len(audio) < n:
        audio = np.append(audio, audio)
    hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
    rows = []
    for i in range(hops):
        seg = audio[int(i * SR):int((i + INPUT_LENGTH) * SR)]
        if len(seg) < n:
            continue
        sig, bak, ovr = p835.run(None, {"input_1": seg.astype("float32")[np.newaxis, :]})[0][0]
        mel = _melspec(seg[:-160]).astype("float32")[np.newaxis, :, :]
        rows.append((P_SIG(sig), P_BAK(bak), P_OVR(ovr), p808.run(None, {"input_1": mel})[0][0][0]))
    sig, bak, ovr, mos = np.mean(np.array(rows, dtype=np.float64), axis=0)
    return {"sig": float(sig), "bak": float(bak), "ovrl": float(ovr), "p808": float(mos)}


def _score_file(args):
    path, models_dir = args
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float64")
    if sr != SR or audio.ndim != 1:
        raise ValueError(f"{path}: expected mono {SR} Hz audio")
    return score(audio, models_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="directory with sig_bak_ovr.onnx and model_v8.onnx")
    p.add_argument("--audio_dir", required=True, help="directory with <name>.flac / <name>.wav files")
    p.add_argument("--names", required=True, help="filelist (clean|noisy per line): the file names to score")
    p.add_argument("--column", type=int, default=0,
                   help="with --audio_dir = the dataset root: score this column of the filelist (0 clean, 1 noisy)")
    p.add_argument("--out", required=True, help="per-file CSV; a .json summary is written next to it")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    paths, names = [], []
    with open(args.names) as fh:
        for line in fh:
            if not line.strip():
                continue
            rel = line.strip().split("|")[args.column]
            name = os.path.splitext(os.path.basename(rel))[0]
            cands = [os.path.join(args.audio_dir, rel)] + [os.path.join(args.audio_dir, name + e)
                                                           for e in (".flac", ".wav")]
            path = next((c for c in cands if os.path.isfile(c)), None)
            if path is None:
                raise FileNotFoundError(f"no audio for {name} in {args.audio_dir}")
            paths.append(path)
            names.append(name)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_score_file, [(q, args.models) for q in paths], chunksize=8))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = ("sig", "bak", "ovrl", "p808")
    with open(args.out, "w") as fh:
        fh.write("file," + ",".join(keys) + "\n")
        for n, r in zip(names, res):
            fh.write(n + "," + ",".join(f"{r[k]:.6f}" for k in keys) + "\n")
    summary = {k: float(np.mean([r[k] for r in res])) for k in keys}
    summary["n_files"] = len(res)
    summary["audio_dir"] = args.audio_dir
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
