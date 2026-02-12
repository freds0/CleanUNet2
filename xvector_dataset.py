"""
Extended dataset for X-Vector training with file path tracking.

This dataset extends the standard MelDataset to include audio file paths
in the returned batch, enabling X-Vector caching by file path.
"""

import torch
from spec_dataset import MelDataset
from typing import Tuple


class XVectorMelDataset(MelDataset):
    """
    Extended MelDataset that returns file paths alongside audio data.

    This enables X-Vector caching by allowing the model to use file paths
    as cache keys, avoiding redundant x-vector extraction.

    Returns:
        tuple: (noisy_audio, noisy_spec, clean_audio, clean_spec, clean_path)
            where clean_path is the full path to the clean audio file
    """

    def __init__(self, return_paths: bool = True, **kwargs):
        """
        Initialize XVectorMelDataset.

        Args:
            return_paths (bool): Whether to return file paths (for caching)
            **kwargs: Arguments passed to parent MelDataset
        """
        super().__init__(**kwargs)
        self.return_paths = return_paths

    def __getitem__(self, index: int):
        """
        Get a single item with optional file path.

        Returns:
            tuple: If return_paths=True: (noisy_audio, noisy_spec, clean_audio, clean_spec, clean_path)
                   If return_paths=False: (noisy_audio, noisy_spec, clean_audio, clean_spec)
        """
        # Get standard outputs from parent class
        noisy_audio, noisy_spec, clean_audio, clean_spec = super().__getitem__(index)

        if self.return_paths:
            # Get file paths
            clean_rel, noisy_rel = self.audio_files[index]
            import os
            clean_path = os.path.join(self.data_dir, clean_rel)

            # Return with clean path for x-vector caching
            return noisy_audio, noisy_spec, clean_audio, clean_spec, clean_path
        else:
            # Standard return without paths
            return noisy_audio, noisy_spec, clean_audio, clean_spec


def xvector_collate_fn(batch):
    """
    Custom collate function for X-Vector training with file path tracking.

    Args:
        batch: List of tuples from XVectorMelDataset

    Returns:
        tuple: (audios_stacked, specs_stacked, clean_audios_stacked, clean_specs_stacked, clean_paths)
            where clean_paths is a list of file paths
    """
    # Check if batch contains paths (5-element tuples)
    if len(batch[0]) == 5:
        audios, specs, clean_audios, clean_specs, clean_paths = zip(*batch)
        include_paths = True
    else:
        audios, specs, clean_audios, clean_specs = zip(*batch)
        include_paths = False

    # Stack tensors (try direct stack first, fall back to padding)
    try:
        audios_stacked = torch.stack(audios)
        specs_stacked = torch.stack(specs)
        clean_audios_stacked = torch.stack(clean_audios)
        clean_specs_stacked = torch.stack(clean_specs)
    except RuntimeError:
        # Fall back to padding on time dimension
        def pad_list(tensors, dim=-1):
            shapes = [t.shape for t in tensors]
            max_len = max(s[-1] for s in shapes)
            padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[-1])) for t in tensors]
            return torch.stack(padded)

        audios_stacked = pad_list([a.squeeze() for a in audios])
        specs_stacked = pad_list([s for s in specs])
        clean_audios_stacked = pad_list([c.squeeze() for c in clean_audios])
        clean_specs_stacked = pad_list([cs for cs in clean_specs])

    if include_paths:
        return audios_stacked, specs_stacked, clean_audios_stacked, clean_specs_stacked, list(clean_paths)
    else:
        return audios_stacked, specs_stacked, clean_audios_stacked, clean_specs_stacked
