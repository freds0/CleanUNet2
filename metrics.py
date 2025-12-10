"""
Objective Metrics Predictor

This module computes objective speech/audio quality metrics (PESQ, STOI, SI-SDR)
for single file pairs or for folders of paired clean/distorted audio.

Usage (CLI):
    python metrics.py --input_dir_clean path/to/clean --input_dir_dist path/to/distorted --output_file results.json

Dependencies:
    - torch
    - torchaudio
    - numpy
    - pesq (pip install pesq)
    - pystoi (pip install pystoi)
    - tqdm
"""

import os
import argparse
import json
from glob import glob
from os.path import join, basename, isdir
from pathlib import Path
from typing import Optional, Dict, Tuple

import logging
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ObjectiveMetricsPredictor:
    """
    Compute objective metrics (PESQ, STOI, SI-SDR) for audio signals.

    The class can be called with file paths or used programmatically by passing
    waveform tensors / numpy arrays to `predict_metrics`.

    Notes:
      - PESQ expects 16 kHz sampling rate (this class will resample if needed).
      - Inputs are converted to mono automatically (averaging channels).
      - SI-SDR is implemented with PyTorch tensors and returns a scalar float.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            device: "cpu" or "cuda" (optional). If None, auto-detects CUDA if available.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logging.info(f"ObjectiveMetricsPredictor initialized on device: {self.device}")

    @staticmethod
    def _to_mono_and_resample(waveform: torch.Tensor, sr: int, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Convert waveform to mono and resample to target_sr if necessary.
        Returns (waveform, sr) where waveform is a torch.Tensor (1, T) and sr == target_sr.
        """
        # waveform: (channels, T)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mix to mono

        if sr != target_sr:
            waveform = resample(waveform, orig_freq=sr, new_freq=target_sr)
            sr = target_sr

        return waveform, sr

    def _load_file(self, filepath: str, target_sr: int = 16000) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """
        Load an audio file and return (waveform, sr). On error returns (None, None).
        Waveform is a torch.Tensor with shape (1, T) in float32.
        """
        try:
            waveform, sr = torchaudio.load(filepath)  # waveform shape: (channels, T)
            waveform, sr = self._to_mono_and_resample(waveform, sr, target_sr)
            # ensure float32
            waveform = waveform.to(dtype=torch.float32)
            return waveform, sr
        except Exception as e:
            logging.error(f"Error loading file '{filepath}': {e}")
            return None, None

    def __call__(self, input_str_clean: str, input_str_distorted: str) -> Dict[str, dict]:
        """
        If input_str_clean is a directory, compute metrics for folder pairs.
        Otherwise compute metrics for the two file paths.
        """
        if isdir(input_str_clean):
            return self.predict_folder(input_str_clean, input_str_distorted)
        else:
            metrics = self.predict_file(input_str_clean, input_str_distorted)
            return {basename(input_str_clean): metrics} if metrics else {}

    def si_snr(self, estimate: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-8) -> float:
        """
        Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
        Expects tensors shaped (1, T) or (T,) — this implementation will operate on last axis.

        Returns:
            scalar float (dB)
        """
        # Convert to 1D floats
        if isinstance(estimate, np.ndarray):
            estimate = torch.from_numpy(estimate)
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference)

        # Ensure float tensors
        estimate = estimate.to(dtype=torch.float32)
        reference = reference.to(dtype=torch.float32)

        # Flatten to (T,)
        if estimate.dim() > 1:
            estimate = estimate.view(-1)
        if reference.dim() > 1:
            reference = reference.view(-1)

        # Zero-mean normalization
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()

        # Projection of estimate onto reference
        reference_energy = torch.sum(reference ** 2)
        if reference_energy.item() == 0:
            logging.warning("Reference signal has zero energy; returning -inf for SI-SNR.")
            return float("-inf")

        scale = torch.dot(estimate, reference) / (reference_energy + epsilon)
        scaled_ref = scale * reference
        e_noise = estimate - scaled_ref

        ratio = torch.sum(scaled_ref ** 2) / (torch.sum(e_noise ** 2) + epsilon)
        si_snr_value = 10.0 * torch.log10(ratio + epsilon)
        return float(si_snr_value.item())

    def predict_metrics(self, waveform_clean, waveform_distorted) -> Optional[dict]:
        """
        Compute PESQ, STOI and SI-SDR for given waveforms.
        Accepts either torch.Tensor or numpy.ndarray. Expects 1D (T,) or (1, T) shapes.

        Returns:
            dict with keys: "pesq", "stoi", "si_sdr" or None if computation failed.
        """
        # Convert torch tensors to numpy after ensuring correct shape and device
        # Ensure CPU numpy arrays for pesq/stoi
        try:
            # If input is torch tensor, convert to cpu numpy
            if torch.is_tensor(waveform_clean):
                w_clean = waveform_clean.detach().cpu().squeeze().numpy()
            else:
                w_clean = np.asarray(waveform_clean)

            if torch.is_tensor(waveform_distorted):
                w_dist = waveform_distorted.detach().cpu().squeeze().numpy()
            else:
                w_dist = np.asarray(waveform_distorted)

            # Flatten if necessary
            if w_clean.ndim != 1:
                w_clean = w_clean.flatten()
            if w_dist.ndim != 1:
                w_dist = w_dist.flatten()

            # PESQ (wideband) - may raise exceptions for invalid input lengths / content
            try:
                pesq_score = pesq(16000, w_clean, w_dist, mode="wb")
            except Exception as e:
                logging.debug(f"PESQ computation failed: {e}")
                pesq_score = 0.0

            # STOI - may raise exceptions, fall back to 0.0
            try:
                stoi_score = stoi(w_clean, w_dist, 16000, extended=False)
            except Exception as e:
                logging.debug(f"STOI computation failed: {e}")
                stoi_score = 0.0

            # SI-SDR / SI-SNR computed with torch for numerical stability
            try:
                w_clean_t = torch.from_numpy(w_clean).to(device=self.device)
                w_dist_t = torch.from_numpy(w_dist).to(device=self.device)
                si_sdr_score = self.si_snr(w_dist_t, w_clean_t)
            except Exception as e:
                logging.debug(f"SI-SDR computation failed: {e}")
                si_sdr_score = 0.0

            return {"pesq": float(pesq_score), "stoi": float(stoi_score), "si_sdr": float(si_sdr_score)}
        except Exception as e:
            logging.error(f"Failed to compute metrics: {e}")
            return None

    def predict_file(self, filepath_clean: str, filepath_distorted: str) -> Optional[dict]:
        """
        Load two audio files, preprocess them (mono + resample to 16kHz) and compute metrics.
        Returns a metrics dict or None on error.
        """
        waveform_clean, sr_clean = self._load_file(filepath_clean)
        if waveform_clean is None:
            logging.error(f"Failed to load clean file: {filepath_clean}")
            return None

        waveform_distorted, sr_dist = self._load_file(filepath_distorted)
        if waveform_distorted is None:
            logging.error(f"Failed to load distorted file: {filepath_distorted}")
            return None

        # Both waveforms are (1, T) torch tensors at 16 kHz
        return self.predict_metrics(waveform_clean.squeeze(0), waveform_distorted.squeeze(0))

    def predict_folder(self, dirpath_clean: str, dirpath_distorted: str, batch_size: int = 1, search_str: str = "*.wav") -> Dict[str, dict]:
        """
        Compute metrics for every paired file in two folders. Files are matched by sorted order.
        Returns a dictionary mapping filename -> metrics dict.
        """
        logging.info(f"Scanning clean folder: {dirpath_clean}, distorted folder: {dirpath_distorted}")
        filelist_clean = sorted(glob(join(dirpath_clean, search_str)))
        filelist_distorted = sorted(glob(join(dirpath_distorted, search_str)))

        if len(filelist_clean) == 0:
            logging.warning("No files found in the clean folder.")
            return {}

        if len(filelist_clean) != len(filelist_distorted):
            logging.error(
                f"Mismatch in file counts: clean={len(filelist_clean)} distorted={len(filelist_distorted)}. "
                "Make sure folders contain matching files in the same order."
            )
            raise AssertionError("Unequal number of files in clean and distorted folders.")

        scores = {}
        # Use tqdm for a progress bar; compute files one-by-one (PESQ/STOI are single-sample routines)
        for f_clean, f_dist in tqdm(zip(filelist_clean, filelist_distorted), total=len(filelist_clean), desc="Calculating Metrics"):
            filename = basename(f_clean)
            metrics = self.predict_file(f_clean, f_dist)
            if metrics:
                scores[filename] = metrics
            else:
                logging.warning(f"Metrics computation failed for pair: {f_clean} , {f_dist}")
        return scores


def main():
    parser = argparse.ArgumentParser(description="Compute PESQ/STOI/SI-SDR for clean/distorted audio pairs.")
    parser.add_argument("--input_dir_clean", "-i", type=str, default="samples/0", help="Input clean folder (or single file)")
    parser.add_argument("--input_dir_dist", "-r", type=str, default="samples/0", help="Input distorted folder (or single file)")
    parser.add_argument("--output_file", "-o", type=str, default="metrics_prediction.json", help="Output JSON filepath")
    # batch_size kept for backwards compatibility; not used to parallelize PESQ/STOI
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="(Deprecated) Batch size")
    parser.add_argument("--search_pattern", "-s", type=str, default="*.wav", help="Search pattern for folder mode")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device (cpu or cuda). If omitted auto-detects.")
    args = parser.parse_args()

    predictor = ObjectiveMetricsPredictor(device=args.device)
    if isdir(args.input_dir_clean):
        results = predictor.predict_folder(args.input_dir_clean, args.input_dir_dist, batch_size=args.batch_size, search_str=args.search_pattern)
    else:
        results = predictor(args.input_dir_clean, args.input_dir_dist)

    # Write JSON results
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Saved metrics to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

