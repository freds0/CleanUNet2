"""
Objective speech-enhancement metrics on whole utterances (16 kHz):
PESQ (WB/NB), STOI, SI-SDR, CSIG, CBAK, COVL, SSNR, WSS and LLR.

The composite measures follow CMGAN / CleanUMamba (see evaluation/composite.py). Unlike the
CleanUMamba `evaluate_dns` script, every file is evaluated (it only scored the last one).
"""

from typing import Dict

import numpy as np
from pesq import pesq
from pystoi import stoi

from .composite import llr, snr, wss

METRIC_NAMES = ("pesq_wb", "pesq_nb", "stoi", "si_sdr", "csig", "cbak", "covl", "ssnr", "wss", "llr", "pesq_wb_r",
                "pesq_wb_hp")
# Leading-silence pads (samples) of the robust PESQ-WB: the same zeros are prepended to the reference and to the
# estimate. PESQ's time alignment makes the score of a single file jump by more than 1 point under such neutral
# changes (IMPROVEMENT_PLAN.md, 12.11); their mean is a steadier per-file score.
ROBUST_PADS = (0, 40, 100, 160, 240)
# Band-limited PESQ-WB: reference and estimate high-passed at this frequency. PESQ-WB's input filter still passes
# 50-100 Hz, where VoiceBank-DEMAND's test noise has ~88% of its energy and its speech very little; half of the gain of
# the metric discriminator sits there (IMPROVEMENT_PLAN.md, 14.7.2).
BAND_LIMIT_HZ = 100.0


def highpass(x: np.ndarray, cutoff_hz: float = BAND_LIMIT_HZ, sample_rate: int = 16000) -> np.ndarray:
    """Zero-phase high-pass: 4th-order Butterworth applied forward and backward (scipy sosfiltfilt)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff_hz, "highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB (both signals mean-removed)."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def composite_metrics(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000,
                      pesq_wb: float = None) -> Dict[str, float]:
    """CSIG / CBAK / COVL (Hu & Loizou), plus SSNR, WSS and LLR. Inputs are float arrays."""
    alpha = 0.95
    wss_dist_vec = np.sort(wss(clean, enhanced, sample_rate))
    wss_dist = float(np.mean(wss_dist_vec[0: round(np.size(wss_dist_vec) * alpha)]))

    llr_dist = np.sort(llr(clean, enhanced, sample_rate))
    llr_top = llr_dist[0: round(np.size(llr_dist) * alpha)]
    llr_mean = float(np.mean(llr_top[~np.isnan(llr_top)]))

    _, segsnr = snr(clean, enhanced, sample_rate)
    seg_snr = float(np.mean(segsnr))

    if pesq_wb is None:
        pesq_wb = pesq(sample_rate, clean, enhanced, "wb")

    csig = float(np.clip(3.093 - 1.029 * llr_mean + 0.603 * pesq_wb - 0.009 * wss_dist, 1, 5))
    cbak = float(np.clip(1.634 + 0.478 * pesq_wb - 0.007 * wss_dist + 0.063 * seg_snr, 1, 5))
    covl = float(np.clip(1.594 + 0.805 * pesq_wb - 0.512 * llr_mean - 0.007 * wss_dist, 1, 5))
    return {"csig": csig, "cbak": cbak, "covl": covl, "ssnr": seg_snr, "wss": wss_dist, "llr": llr_mean}


def robust_pesq_wb(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000, first: float = None,
                   pads=ROBUST_PADS) -> float:
    """Mean PESQ-WB over the leading-silence pads of ROBUST_PADS (`first`: the already computed score of pad 0)."""
    scores = []
    for p in pads:
        if p == 0 and first is not None:
            scores.append(first)
            continue
        z = np.zeros(p)
        scores.append(float(pesq(sample_rate, np.concatenate([z, clean]), np.concatenate([z, enhanced]), "wb")))
    return float(np.mean(scores))


def evaluate_pair(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000,
                  full: bool = True, robust: bool = False, band_limited: bool = False) -> Dict[str, float]:
    """All metrics for one utterance. With full=False only PESQ-WB, STOI and SI-SDR are computed; robust=True adds
    pesq_wb_r (robust_pesq_wb), band_limited=True adds pesq_wb_hp (both signals high-passed at BAND_LIMIT_HZ)."""
    if sample_rate not in (8000, 16000):
        raise ValueError("PESQ requires 8 or 16 kHz audio")
    clean = np.asarray(clean, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)
    L = min(len(clean), len(enhanced))
    clean, enhanced = clean[:L], enhanced[:L]

    out = {"pesq_wb": float(pesq(sample_rate, clean, enhanced, "wb")),
           "stoi": float(stoi(clean, enhanced, sample_rate, extended=False)),
           "si_sdr": si_sdr(clean, enhanced)}
    if full:
        out["pesq_nb"] = float(pesq(sample_rate, clean, enhanced, "nb"))
        out.update(composite_metrics(clean, enhanced, sample_rate, pesq_wb=out["pesq_wb"]))
    if robust:
        out["pesq_wb_r"] = robust_pesq_wb(clean, enhanced, sample_rate, first=out["pesq_wb"])
    if band_limited:
        out["pesq_wb_hp"] = float(pesq(sample_rate, highpass(clean, sample_rate=sample_rate),
                                       highpass(enhanced, sample_rate=sample_rate), "wb"))
    return out


def _evaluate_pair_star(args):
    clean, enhanced, sample_rate, full, robust, band_limited = args
    try:
        return evaluate_pair(clean, enhanced, sample_rate, full, robust, band_limited)
    except Exception as err:  # e.g. pesq.NoUtterancesError on silent outputs
        return {"error": f"{type(err).__name__}: {err}"}


def evaluate_many(pairs, sample_rate: int = 16000, full: bool = True, num_workers: int = 8, robust: bool = False,
                  band_limited: bool = False):
    """Evaluate a list of (clean, enhanced) arrays in parallel. Returns a list of metric dicts."""
    jobs = [(c, e, sample_rate, full, robust, band_limited) for c, e in pairs]
    if num_workers <= 1:
        return [_evaluate_pair_star(j) for j in jobs]
    # "fork" works for interactive/stdin callers too; workers never touch CUDA.
    from multiprocessing import get_context
    with get_context("fork").Pool(num_workers, initializer=_limit_threads) as pool:
        return pool.map(_evaluate_pair_star, jobs, chunksize=4)


def _limit_threads():
    """One BLAS thread per worker process (avoids oversubscription)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def summarize(results) -> Dict[str, float]:
    """Mean of each metric over the files that were scored without error."""
    ok = [r for r in results if "error" not in r]
    keys = [k for k in METRIC_NAMES if ok and k in ok[0]]
    summary = {k: float(np.mean([r[k] for r in ok])) for k in keys}
    summary["n_files"] = len(ok)
    summary["n_errors"] = len(results) - len(ok)
    return summary
