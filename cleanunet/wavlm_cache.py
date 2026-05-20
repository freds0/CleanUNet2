"""
WavLM Embedding Cache System

This module provides a caching mechanism for pre-extracted wavlm embeddings.
Unlike the x-vector cache that extracts on-the-fly, this cache ONLY loads
pre-extracted embeddings from disk.

The embeddings must be pre-extracted using extract_wavlm_embeddings.py before training.
"""

import torch
import hashlib
from pathlib import Path
from typing import Optional, Dict
import os


class WavLMCache:
    """
    Cache system for pre-extracted WavLM embeddings.

    This cache ONLY loads embeddings that were pre-extracted and saved to disk.
    It does NOT extract embeddings on-the-fly.

    Cache Structure:
        cache_dir/
        ├── <hash1>.pt
        ├── <hash2>.pt
        ├── metadata.yaml
        └── ...
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        """
        Initialize WavLM cache.

        Args:
            cache_dir (str): Directory containing pre-extracted embeddings
            enabled (bool): Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)

        if self.enabled:
            if not self.cache_dir.exists():
                print(f"[WavLMCache] Warning: Cache directory does not exist: {self.cache_dir}")
                print(f"[WavLMCache] Please run extract_wavlm_embeddings.py first!")
                self.enabled = False
            else:
                print(f"[WavLMCache] Cache enabled: {self.cache_dir}")
                self._stats = {
                    'hits': 0,
                    'misses': 0
                }

                # Load metadata if available
                metadata_file = self.cache_dir / 'metadata.yaml'
                if metadata_file.exists():
                    import yaml
                    with open(metadata_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                    print(f"[WavLMCache] Loaded metadata:")
                    print(f"  - Model: {metadata.get('model_name', 'unknown')}")
                    print(f"  - Embedding dim: {metadata.get('embedding_dim', 'unknown')}")
                    print(f"  - Sample rate: {metadata.get('sample_rate', 'unknown')} Hz")
                    print(f"  - Total embeddings: {metadata.get('extracted', 'unknown')}")
        else:
            print("[WavLMCache] Cache disabled")
            self._stats = None

    def _get_cache_key(self, audio_path: str) -> str:
        """
        Generate a unique cache key from audio file path.

        Uses MD5 hash of the full path to create a unique identifier.

        Args:
            audio_path (str): Full path to audio file

        Returns:
            str: MD5 hash of the path (used as cache filename)
        """
        path_bytes = str(audio_path).encode('utf-8')
        return hashlib.md5(path_bytes).hexdigest()

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get the full path to the cache file for this audio."""
        cache_key = self._get_cache_key(audio_path)
        return self.cache_dir / f"{cache_key}.pt"

    def get(self, audio_path: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        Retrieve wavlm embedding from cache.

        Args:
            audio_path (str): Path to the audio file
            device (str): Device to load tensor to

        Returns:
            torch.Tensor or None: Cached embedding if found, None otherwise
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(audio_path)

        if cache_path.exists():
            try:
                embedding = torch.load(cache_path, map_location=device)
                self._stats['hits'] += 1
                return embedding
            except Exception as e:
                print(f"[WavLMCache] Warning: Failed to load cache for {audio_path}: {e}")
                self._stats['misses'] += 1
                return None
        else:
            self._stats['misses'] += 1
            return None

    def get_batch(self, audio_paths: list, device: str = 'cpu') -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve multiple wavlm embeddings from cache.

        Args:
            audio_paths (list): List of audio file paths
            device (str): Device to load tensors to

        Returns:
            dict: Mapping from audio_path to cached embedding (or None if not cached)
        """
        results = {}
        for path in audio_paths:
            results[path] = self.get(path, device=device)
        return results

    def exists(self, audio_path: str) -> bool:
        """
        Check if embedding exists in cache without loading it.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            bool: True if embedding exists, False otherwise
        """
        if not self.enabled:
            return False

        cache_path = self._get_cache_path(audio_path)
        return cache_path.exists()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including hits, misses, and hit rate
        """
        if not self.enabled or self._stats is None:
            return {'enabled': False}

        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'enabled': True,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'total_requests': total_requests,
            'hit_rate': f"{hit_rate:.2f}%",
            'cache_size_mb': self._get_cache_size_mb()
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate total size of cache directory in MB."""
        if not self.cache_dir.exists():
            return 0.0

        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pt") if f.is_file()
        )
        return total_size / (1024 * 1024)  # Convert to MB

    def _reset_stats(self):
        """Reset cache statistics."""
        if self._stats is not None:
            self._stats = {
                'hits': 0,
                'misses': 0
            }

    def print_stats(self):
        """Print cache statistics to console."""
        stats = self.get_stats()

        if not stats['enabled']:
            print("[WavLMCache] Cache is disabled")
            return

        print("\n" + "=" * 60)
        print("WavLM Embedding Cache Statistics")
        print("=" * 60)
        print(f"Cache Directory: {self.cache_dir}")
        print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"  - Hits: {stats['hits']}")
        print(f"  - Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']}")
        print("=" * 60 + "\n")

    def __repr__(self):
        return f"WavLMCache(cache_dir={self.cache_dir}, enabled={self.enabled})"
