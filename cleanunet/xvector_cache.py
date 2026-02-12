"""
X-Vector Cache System for CleanUNet2-xvector

This module provides a caching mechanism for x-vector embeddings to avoid
re-extracting them on every training step. Embeddings are stored on disk
and indexed by audio file hash.
"""

import torch
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict
import os


class XVectorCache:
    """
    Cache system for X-Vector embeddings.

    Stores extracted x-vectors on disk to avoid redundant computation.
    Uses MD5 hash of audio file path as the cache key.

    Cache Structure:
        cache_dir/
        ├── <hash1>.pt
        ├── <hash2>.pt
        └── ...
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        """
        Initialize X-Vector cache.

        Args:
            cache_dir (str): Directory to store cached x-vectors
            enabled (bool): Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[XVectorCache] Cache enabled: {self.cache_dir}")
            self._stats = {
                'hits': 0,
                'misses': 0,
                'saves': 0
            }
        else:
            print("[XVectorCache] Cache disabled")
            self._stats = None

    def _get_cache_key(self, audio_path: str) -> str:
        """
        Generate a unique cache key from audio file path.

        Uses MD5 hash of the full path to create a unique identifier.
        This ensures that different files always get different cache entries.

        Args:
            audio_path (str): Full path to audio file

        Returns:
            str: MD5 hash of the path (used as cache filename)
        """
        # Use full path to ensure uniqueness across different directories
        path_bytes = str(audio_path).encode('utf-8')
        return hashlib.md5(path_bytes).hexdigest()

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get the full path to the cache file for this audio."""
        cache_key = self._get_cache_key(audio_path)
        return self.cache_dir / f"{cache_key}.pt"

    def get(self, audio_path: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        Retrieve x-vector from cache if available.

        Args:
            audio_path (str): Path to the audio file
            device (str): Device to load tensor to

        Returns:
            torch.Tensor or None: Cached x-vector if found, None otherwise
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(audio_path)

        if cache_path.exists():
            try:
                xvector = torch.load(cache_path, map_location=device)
                self._stats['hits'] += 1
                return xvector
            except Exception as e:
                print(f"[XVectorCache] Warning: Failed to load cache for {audio_path}: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
                self._stats['misses'] += 1
                return None
        else:
            self._stats['misses'] += 1
            return None

    def set(self, audio_path: str, xvector: torch.Tensor):
        """
        Save x-vector to cache.

        Args:
            audio_path (str): Path to the audio file
            xvector (torch.Tensor): X-vector embedding to cache
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(audio_path)

        try:
            # Save to temporary file first to avoid corruption
            temp_path = cache_path.with_suffix('.tmp')
            torch.save(xvector.cpu(), temp_path)
            # Atomic rename (safer than direct write)
            temp_path.rename(cache_path)
            self._stats['saves'] += 1
        except Exception as e:
            print(f"[XVectorCache] Warning: Failed to save cache for {audio_path}: {e}")
            # Clean up temporary file if it exists
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def get_batch(self, audio_paths: list, device: str = 'cpu') -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve multiple x-vectors from cache.

        Args:
            audio_paths (list): List of audio file paths
            device (str): Device to load tensors to

        Returns:
            dict: Mapping from audio_path to cached x-vector (or None if not cached)
        """
        results = {}
        for path in audio_paths:
            results[path] = self.get(path, device=device)
        return results

    def set_batch(self, xvector_dict: Dict[str, torch.Tensor]):
        """
        Save multiple x-vectors to cache.

        Args:
            xvector_dict (dict): Mapping from audio_path to x-vector
        """
        for path, xvector in xvector_dict.items():
            self.set(path, xvector)

    def clear(self):
        """Clear all cached x-vectors."""
        if not self.enabled:
            return

        count = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                print(f"[XVectorCache] Warning: Failed to delete {cache_file}: {e}")

        print(f"[XVectorCache] Cleared {count} cache files")
        self._reset_stats()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including hits, misses, saves, and hit rate
        """
        if not self.enabled or self._stats is None:
            return {'enabled': False}

        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'enabled': True,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'saves': self._stats['saves'],
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
                'misses': 0,
                'saves': 0
            }

    def print_stats(self):
        """Print cache statistics to console."""
        stats = self.get_stats()

        if not stats['enabled']:
            print("[XVectorCache] Cache is disabled")
            return

        print("\n" + "=" * 60)
        print("X-Vector Cache Statistics")
        print("=" * 60)
        print(f"Cache Directory: {self.cache_dir}")
        print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"  - Hits: {stats['hits']}")
        print(f"  - Misses: {stats['misses']}")
        print(f"  - Saves: {stats['saves']}")
        print(f"Hit Rate: {stats['hit_rate']}")
        print("=" * 60 + "\n")

    def __repr__(self):
        return f"XVectorCache(cache_dir={self.cache_dir}, enabled={self.enabled})"
