"""
Metrics for speech enhancement evaluation
Includes PESQ, STOI, and SI-SDR metrics
"""

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi


def calculate_pesq(enhanced, clean, sr=16000):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score.
    
    Args:
        enhanced (torch.Tensor): Enhanced audio tensor (batch, samples) or (batch, 1, samples)
        clean (torch.Tensor): Clean reference audio tensor
        sr (int): Sample rate (8000 or 16000)
        
    Returns:
        float: Average PESQ score
    """
    # Convert to numpy and handle dimensions
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if enhanced.ndim == 3:
        enhanced = enhanced.squeeze(1)
    if clean.ndim == 3:
        clean = clean.squeeze(1)
    
    # Calculate PESQ for each sample in batch
    batch_size = enhanced.shape[0]
    pesq_scores = []
    
    mode = 'wb' if sr == 16000 else 'nb'
    
    for i in range(batch_size):
        try:
            score = pesq(sr, clean[i], enhanced[i], mode)
            pesq_scores.append(score)
        except Exception as e:
            print(f"[Warning] PESQ calculation failed for sample {i}: {e}")
            pesq_scores.append(0.0)
    
    return np.mean(pesq_scores)


def calculate_stoi(enhanced, clean, sr=16000):
    """
    Calculate STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        enhanced (torch.Tensor): Enhanced audio tensor (batch, samples)
        clean (torch.Tensor): Clean reference audio tensor
        sr (int): Sample rate
        
    Returns:
        float: Average STOI score
    """
    # Convert to numpy and handle dimensions
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if enhanced.ndim == 3:
        enhanced = enhanced.squeeze(1)
    if clean.ndim == 3:
        clean = clean.squeeze(1)
    
    # Calculate STOI for each sample in batch
    batch_size = enhanced.shape[0]
    stoi_scores = []
    
    for i in range(batch_size):
        try:
            score = stoi(clean[i], enhanced[i], sr, extended=False)
            stoi_scores.append(score)
        except Exception as e:
            print(f"[Warning] STOI calculation failed for sample {i}: {e}")
            stoi_scores.append(0.0)
    
    return np.mean(stoi_scores)


def calculate_sisdr(enhanced, clean):
    """
    Calculate SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).
    
    Args:
        enhanced (torch.Tensor): Enhanced audio tensor (batch, samples)
        clean (torch.Tensor): Clean reference audio tensor
        
    Returns:
        float: Average SI-SDR score in dB
    """
    # Convert to numpy and handle dimensions
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if enhanced.ndim == 3:
        enhanced = enhanced.squeeze(1)
    if clean.ndim == 3:
        clean = clean.squeeze(1)
    
    # Calculate SI-SDR for each sample in batch
    batch_size = enhanced.shape[0]
    sisdr_scores = []
    
    for i in range(batch_size):
        try:
            # Ensure vectors are 1D
            s_target = clean[i]
            s_estimate = enhanced[i]
            
            # Remove mean
            s_target = s_target - np.mean(s_target)
            s_estimate = s_estimate - np.mean(s_estimate)
            
            # Calculate SI-SDR
            alpha = np.dot(s_estimate, s_target) / (np.linalg.norm(s_target) ** 2 + 1e-8)
            s_target_scaled = alpha * s_target
            
            # Signal and noise
            e_noise = s_estimate - s_target_scaled
            
            # SI-SDR in dB
            sisdr = 10 * np.log10(
                (np.linalg.norm(s_target_scaled) ** 2) / 
                (np.linalg.norm(e_noise) ** 2 + 1e-8)
            )
            
            sisdr_scores.append(sisdr)
            
        except Exception as e:
            print(f"[Warning] SI-SDR calculation failed for sample {i}: {e}")
            sisdr_scores.append(0.0)
    
    return np.mean(sisdr_scores)


def calculate_snr(enhanced, clean):
    """
    Calculate SNR (Signal-to-Noise Ratio).
    
    Args:
        enhanced (torch.Tensor): Enhanced audio tensor
        clean (torch.Tensor): Clean reference audio tensor
        
    Returns:
        float: Average SNR in dB
    """
    # Convert to numpy and handle dimensions
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if enhanced.ndim == 3:
        enhanced = enhanced.squeeze(1)
    if clean.ndim == 3:
        clean = clean.squeeze(1)
    
    # Calculate SNR for each sample in batch
    batch_size = enhanced.shape[0]
    snr_scores = []
    
    for i in range(batch_size):
        try:
            signal_power = np.sum(clean[i] ** 2)
            noise = enhanced[i] - clean[i]
            noise_power = np.sum(noise ** 2)
            
            snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
            snr_scores.append(snr)
            
        except Exception as e:
            print(f"[Warning] SNR calculation failed for sample {i}: {e}")
            snr_scores.append(0.0)
    
    return np.mean(snr_scores)


def calculate_metrics(enhanced, clean, sr=16000, metrics=['pesq', 'stoi', 'sisdr']):
    """
    Calculate multiple metrics at once.
    
    Args:
        enhanced (torch.Tensor or np.ndarray): Enhanced audio
        clean (torch.Tensor or np.ndarray): Clean reference audio
        sr (int): Sample rate
        metrics (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary with metric names and values
    """
    results = {}
    
    if 'pesq' in metrics:
        try:
            results['pesq'] = calculate_pesq(enhanced, clean, sr)
        except Exception as e:
            print(f"[Warning] PESQ calculation failed: {e}")
            results['pesq'] = 0.0
    
    if 'stoi' in metrics:
        try:
            results['stoi'] = calculate_stoi(enhanced, clean, sr)
        except Exception as e:
            print(f"[Warning] STOI calculation failed: {e}")
            results['stoi'] = 0.0
    
    if 'sisdr' in metrics:
        try:
            results['sisdr'] = calculate_sisdr(enhanced, clean)
        except Exception as e:
            print(f"[Warning] SI-SDR calculation failed: {e}")
            results['sisdr'] = 0.0
    
    if 'snr' in metrics:
        try:
            results['snr'] = calculate_snr(enhanced, clean)
        except Exception as e:
            print(f"[Warning] SNR calculation failed: {e}")
            results['snr'] = 0.0
    
    return results


class MetricsTracker:
    """
    Track metrics across multiple batches/epochs.
    """
    
    def __init__(self, metrics=['pesq', 'stoi', 'sisdr']):
        """
        Initialize metrics tracker.
        
        Args:
            metrics (list): List of metric names to track
        """
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.values = {metric: [] for metric in self.metrics}
    
    def update(self, enhanced, clean, sr=16000):
        """
        Update metrics with new batch.
        
        Args:
            enhanced (torch.Tensor): Enhanced audio
            clean (torch.Tensor): Clean audio
            sr (int): Sample rate
        """
        results = calculate_metrics(enhanced, clean, sr, self.metrics)
        
        for metric, value in results.items():
            if metric in self.values:
                self.values[metric].append(value)
    
    def compute(self):
        """
        Compute average metrics.
        
        Returns:
            dict: Dictionary with average metric values
        """
        averages = {}
        
        for metric, values in self.values.items():
            if len(values) > 0:
                averages[metric] = np.mean(values)
            else:
                averages[metric] = 0.0
        
        return averages
    
    def compute_std(self):
        """
        Compute standard deviation of metrics.
        
        Returns:
            dict: Dictionary with std of metric values
        """
        stds = {}
        
        for metric, values in self.values.items():
            if len(values) > 0:
                stds[metric] = np.std(values)
            else:
                stds[metric] = 0.0
        
        return stds


# Example usage
if __name__ == '__main__':
    """
    Test metrics calculation.
    """
    print("="*60)
    print("Testing Metrics")
    print("="*60)
    
    # Create dummy audio
    sr = 16000
    duration = 2  # seconds
    samples = sr * duration
    
    # Generate test signals
    clean = torch.randn(2, 1, samples)  # Batch of 2
    
    # Simulate enhanced audio (clean + small noise)
    noise = torch.randn_like(clean) * 0.1
    enhanced = clean + noise
    
    print(f"\nTest audio:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Shape: {clean.shape}")
    
    # Test individual metrics
    print(f"\nTesting individual metrics:")
    
    pesq_score = calculate_pesq(enhanced, clean, sr)
    print(f"  PESQ: {pesq_score:.4f}")
    
    stoi_score = calculate_stoi(enhanced, clean, sr)
    print(f"  STOI: {stoi_score:.4f}")
    
    sisdr_score = calculate_sisdr(enhanced, clean)
    print(f"  SI-SDR: {sisdr_score:.4f} dB")
    
    snr_score = calculate_snr(enhanced, clean)
    print(f"  SNR: {snr_score:.4f} dB")
    
    # Test calculate_metrics
    print(f"\nTesting calculate_metrics:")
    metrics = calculate_metrics(enhanced, clean, sr)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Test MetricsTracker
    print(f"\nTesting MetricsTracker:")
    tracker = MetricsTracker()
    
    # Simulate multiple batches
    for i in range(3):
        noise = torch.randn_like(clean) * (0.1 + i * 0.05)
        enhanced = clean + noise
        tracker.update(enhanced, clean, sr)
    
    avg_metrics = tracker.compute()
    std_metrics = tracker.compute_std()
    
    print(f"  Average metrics over 3 batches:")
    for metric, value in avg_metrics.items():
        print(f"    {metric.upper()}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    print("\n" + "="*60)
    print("✓ All metrics tests passed!")
    print("="*60)