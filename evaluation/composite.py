"""
Composite speech-quality measures (WSS, LLR, segmental SNR) used to compute CSIG, CBAK and COVL.

Copied from CleanUMamba (src/util/python_eval.py, MIT License), which adapted them from CMGAN
(https://github.com/ruizhecao96/CMGAN, src/tools/compute_metrics.py, MIT License, (c) 2022 Ruizhe Cao),
itself a port of the MATLAB composite measures of Hu & Loizou (2008).
"""

import math

import numpy as np
from scipy.fftpack import fft
from scipy.linalg import toeplitz


def wss(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech, which must be the same.
    """Weighted spectral slope distance (Klatt), per frame; returns the frame values used by CSIG/CBAK/COVL."""
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Files must have same length.")

    # Global variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(
        int
    )  # window length in samples
    skiprate = (np.floor(np.divide(winlength, 4))).astype(int)  # window skip in samples
    max_freq = (np.divide(sample_rate, 2)).astype(int)  # maximum bandwidth
    num_crit = 25  # number of critical bands

    USE_FFT_SPECTRUM = 1  # defaults to 10th order LP spectrum
    n_fft = (np.power(2, np.ceil(np.log2(2 * winlength)))).astype(int)
    n_fftby2 = (np.multiply(0.5, n_fft)).astype(int)  # FFT size/2
    Kmax = 20.0  # value suggested by Klatt, pg 1280
    Klocmax = 1.0  # value suggested by Klatt, pg 1280

    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = np.array(
        [
            50.0000,
            120.000,
            190.000,
            260.000,
            330.000,
            400.000,
            470.000,
            540.000,
            617.372,
            703.378,
            798.717,
            904.128,
            1020.38,
            1148.30,
            1288.72,
            1442.54,
            1610.70,
            1794.16,
            1993.93,
            2211.08,
            2446.71,
            2701.97,
            2978.04,
            3276.17,
            3597.63,
        ]
    )
    bandwidth = np.array(
        [
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            77.3724,
            86.0056,
            95.3398,
            105.411,
            116.256,
            127.914,
            140.423,
            153.823,
            168.154,
            183.457,
            199.776,
            217.153,
            235.631,
            255.255,
            276.072,
            298.126,
            321.465,
            346.136,
        ]
    )

    bw_min = bandwidth[0]  # minimum critical bandwidth

    # Set up the critical band filters.
    # Note here that Gaussianly shaped filters are used.
    # Also, the sum of the filter weights are equivalent for each critical band filter.
    # Filter less than -30 dB and set to zero.
    min_factor = math.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter
    crit_filter = np.empty((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = np.arange(n_fftby2)
        crit_filter[i, :] = np.exp(
            -11 * np.square(np.divide(j - np.floor(f0), bw)) + norm_factor
        )
        cond = np.greater(crit_filter[i, :], min_factor)
        crit_filter[i, :] = np.where(cond, crit_filter[i, :], 0)
    # For each frame of input speech, calculate the Weighted Spectral Slope Measure
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength] / 32768
        processed_frame = processed_speech[start : start + winlength] / 32768
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)
        # (2) Compute the Power Spectrum of Clean and Processed
        # if USE_FFT_SPECTRUM:
        clean_spec = np.square(np.abs(fft(clean_frame, n_fft)))
        processed_spec = np.square(np.abs(fft(processed_frame, n_fft)))

        # (3) Compute Filterbank Output Energies (in dB scale)
        clean_energy = np.matmul(crit_filter, clean_spec[0:n_fftby2])
        processed_energy = np.matmul(crit_filter, processed_spec[0:n_fftby2])

        clean_energy = 10 * np.log10(np.maximum(clean_energy, 1e-10))
        processed_energy = 10 * np.log10(np.maximum(processed_energy, 1e-10))

        # (4) Compute Spectral Slope (dB[i+1]-dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[0 : num_crit - 1]
        processed_slope = (
            processed_energy[1:num_crit] - processed_energy[0 : num_crit - 1]
        )

        # (5) Find the nearest peak locations in the spectra to each critical band.
        #     If the slope is negative, we search to the left. If positive, we search to the right.
        clean_loc_peak = np.empty(num_crit - 1)
        processed_loc_peak = np.empty(num_crit - 1)

        for i in range(num_crit - 1):
            # find the peaks in the clean speech signal
            if clean_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (clean_slope[n] > 0):
                    n = n + 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (clean_slope[n] <= 0):
                    n = n - 1
                clean_loc_peak[i] = clean_energy[n + 1]

            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (processed_slope[n] > 0):
                    n = n + 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (processed_slope[n] <= 0):
                    n = n - 1
                processed_loc_peak[i] = processed_energy[n + 1]

        # (6) Compute the WSS Measure for this frame. This includes determination of the weighting function.
        dBMax_clean = np.max(clean_energy)
        dBMax_processed = np.max(processed_energy)
        """
        The weights are calculated by averaging individual weighting factors from the clean and processed frame.
        These weights W_clean and W_processed should range from 0 to 1 and place more emphasis on spectral peaks
        and less emphasis on slope differences in spectral valleys.
        This procedure is described on page 1280 of Klatt's 1982 ICASSP paper.
        """
        Wmax_clean = np.divide(
            Kmax, Kmax + dBMax_clean - clean_energy[0 : num_crit - 1]
        )
        Wlocmax_clean = np.divide(
            Klocmax, Klocmax + clean_loc_peak - clean_energy[0 : num_crit - 1]
        )
        W_clean = np.multiply(Wmax_clean, Wlocmax_clean)

        Wmax_processed = np.divide(
            Kmax, Kmax + dBMax_processed - processed_energy[0 : num_crit - 1]
        )
        Wlocmax_processed = np.divide(
            Klocmax, Klocmax + processed_loc_peak - processed_energy[0 : num_crit - 1]
        )
        W_processed = np.multiply(Wmax_processed, Wlocmax_processed)

        W = np.divide(np.add(W_clean, W_processed), 2.0)
        slope_diff = np.subtract(clean_slope, processed_slope)[0 : num_crit - 1]
        distortion[frame_count] = np.dot(W, np.square(slope_diff)) / np.sum(W)
        # this normalization is not part of Klatt's paper, but helps to normalize the measure.
        # Here we scale the measure by the sum of the weights.
        start = start + skiprate
    return distortion


def llr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech.  Must be the same.
    """Log-likelihood ratio between the LPC models of clean and processed speech, per frame."""
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    # Global Variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(
        int
    )  # window length in samples
    skiprate = (np.floor(winlength / 4)).astype(int)  # window skip in samples
    if sample_rate < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int((clean_length - winlength) / skiprate)  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Get the autocorrelation lags and LPC parameters used to compute the LLR measure.
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

        # (3) Compute the LLR measure
        numerator = np.dot(np.matmul(A_processed, toeplitz(R_clean)), A_processed)
        denominator = np.dot(np.matmul(A_clean, toeplitz(R_clean)), A_clean)
        distortion[frame_count] = np.log(numerator / denominator)
        start = start + skiprate
    return distortion


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocorrelation Lags
    """Autocorrelation and LPC coefficients of one frame (Levinson-Durbin). Returns (acorr, refcoeff, lpparams)."""
    winlength = np.size(speech_frame)
    R = np.empty(model_order + 1)
    E = np.empty(model_order + 1)
    for k in range(model_order + 1):
        R[k] = np.dot(speech_frame[0 : winlength - k], speech_frame[k:winlength])

    # (2) Levinson-Durbin
    a = np.ones(model_order)
    a_past = np.empty(model_order)
    rcoeff = np.empty(model_order)
    E[0] = R[0]
    for i in range(model_order):
        a_past[0:i] = a[0:i]
        sum_term = np.dot(a_past[0:i], R[i:0:-1])
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i == 0:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 : -1 : -1], rcoeff[i])
        else:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 :: -1], rcoeff[i])
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = R
    refcoeff = rcoeff
    lpparams = np.concatenate((np.array([1]), -a))
    return acorr, refcoeff, lpparams


def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    """Overall SNR and per-frame segmental SNR (clipped to [-10, 35] dB), as in the MATLAB composite measures."""
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    overall_snr = 10 * np.log10(
        np.sum(np.square(clean_speech))
        / np.sum(np.square(clean_speech - processed_speech))
    )

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(
            signal_energy / (noise_energy + EPS) + EPS
        )
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr
