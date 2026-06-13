import os
import random
from glob import glob
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset

from spec_dataset import MelDataset, custom_collate_fn


# ---------------------------------------------------------------------------
# CleanOnlyDataset: clean files + noise augmentation
# ---------------------------------------------------------------------------
class CleanOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset with only clean audio files. Creates noisy versions on-the-fly
    via augmentation (requires AddBackgroundNoise or similar).
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        segment_size: int = 8192,
        sampling_rate: int = 16000,
        n_fft: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        augmentations: list = None,
        return_audio_paths: bool = False,
    ):
        super().__init__()
        import torchaudio
        import torchaudio.transforms as T
        from augmentation import AudioAugmenter

        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.return_audio_paths = return_audio_paths

        # Find all clean audio files
        self.clean_files = self._find_audio_files(clean_dir)
        if not self.clean_files:
            raise ValueError(f"No audio files found in: {clean_dir}")
        print(f"[CleanOnlyDataset] Found {len(self.clean_files)} clean files in {clean_dir}")

        # Build augmentation pipeline
        if augmentations:
            self.augmenter = AudioAugmenter(augmentations, device='cpu')
        else:
            # Default: add background noise from noise_dir
            default_aug = [{
                'name': 'AddBackgroundNoise',
                'params': {
                    'background_paths': noise_dir,
                    'min_snr_in_db': 3.0,
                    'max_snr_in_db': 30.0,
                    'p': 1.0,
                }
            }]
            self.augmenter = AudioAugmenter(default_aug, device='cpu')

        self.spectrogram_fn = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_size, win_length=win_size,
            power=1.0, normalized=True, center=False
        )

    def _find_audio_files(self, directory: str) -> List[str]:
        exts = ['*.wav', '*.flac', '*.mp3', '*.ogg']
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
        return sorted(set(files))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        from spec_dataset import load_wav

        clean_path = self.clean_files[index]
        clean_audio, _ = load_wav(clean_path, self.sampling_rate)
        clean_audio = clean_audio / (clean_audio.abs().max() + 1e-9)

        # Crop/pad
        if clean_audio.size(1) >= self.segment_size:
            start = random.randint(0, clean_audio.size(1) - self.segment_size)
            start = start - (start % 2)
            clean_audio = clean_audio[:, start:start + self.segment_size]
        else:
            pad = self.segment_size - clean_audio.size(1)
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad))

        # Create noisy version via augmentation
        noisy_audio = self.augmenter.apply(clean_audio, self.sampling_rate)

        noisy_spec = self.spectrogram_fn(noisy_audio).squeeze(0)
        clean_spec = self.spectrogram_fn(clean_audio).squeeze(0)

        noisy_audio = noisy_audio.squeeze().unsqueeze(0)
        clean_audio = clean_audio.squeeze().unsqueeze(0)

        if self.return_audio_paths:
            rel_path = os.path.relpath(clean_path, self.clean_dir)
            return noisy_audio, noisy_spec, clean_audio, clean_spec, rel_path
        return noisy_audio, noisy_spec, clean_audio, clean_spec


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class CleanUNetDataModule(pl.LightningDataModule):
    """
    DataModule supporting three dataset formats:

    1. **Paired** (default): filelist with "clean_path|noisy_path" lines.
       Config: data_dir + train_list_path

    2. **Clean + Noise**: clean audio folder + noise folder.
       Config: datasets entry with type "clean_with_noise"

    3. **Mixed**: multiple datasets combined.
       Config: datasets list with type/weight per entry.
    """

    def __init__(
        self,
        data_dir: str = ".",
        train_list_path: str = None,
        val_list_path: str = None,
        val_split: float = None,
        batch_size: int = 8,
        num_workers: int = 4,
        persistent_workers: bool = False,
        segment_size: int = None,
        sampling_rate: int = 16000,
        augmentations: list = None,
        use_preextracted_embeddings: bool = True,
        quick_test: bool = False,
        quick_test_samples: int = 30,
        # Multi-dataset support
        datasets: list = None,
        noise_dir: str = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.augmentations = augmentations
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.quick_test = quick_test
        self.quick_test_samples = quick_test_samples if quick_test else None
        self.datasets_config = datasets
        self.noise_dir = noise_dir

        # Validate: need either datasets list OR train_list_path
        if self.datasets_config is None and self.train_list_path is None:
            raise ValueError(
                "Provide either 'train_list_path' (for paired dataset) "
                "or 'datasets' list (for multi-dataset training)."
            )

        if self.datasets_config is None:
            if self.val_list_path is None and self.val_split is None:
                raise ValueError(
                    "Provide 'val_list_path' or 'val_split' for validation."
                )
            if self.val_split is not None and not (0.0 < self.val_split < 1.0):
                raise ValueError(f"val_split must be in (0, 1), got: {self.val_split}")

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Create train/val datasets."""

        if self.datasets_config is not None:
            self._setup_multi_dataset()
        else:
            self._setup_single_dataset()

        # Quick test: subset
        if self.quick_test and self.quick_test_samples:
            n = self.quick_test_samples
            self.train_dataset = Subset(self.train_dataset, range(min(n, len(self.train_dataset))))
            self.val_dataset = Subset(self.val_dataset, range(min(n, len(self.val_dataset))))

        print(f"[DataModule] Train samples: {len(self.train_dataset)}")
        print(f"[DataModule] Val samples:   {len(self.val_dataset)}")

    # ------------------------------------------------------------------
    def _setup_single_dataset(self):
        """Standard paired dataset setup."""
        ds_kwargs = {}
        if self.segment_size:
            ds_kwargs["segment_size"] = self.segment_size
        if self.sampling_rate:
            ds_kwargs["sampling_rate"] = self.sampling_rate
        if self.use_preextracted_embeddings:
            ds_kwargs["return_audio_paths"] = True

        if self.val_list_path is not None:
            # Separate val file
            train_kwargs = ds_kwargs.copy()
            if self.augmentations:
                train_kwargs["augmentations"] = self.augmentations

            self.train_dataset = MelDataset(
                data_dir=self.data_dir,
                data_files=self.train_list_path,
                **train_kwargs
            )
            self.val_dataset = MelDataset(
                data_dir=self.data_dir,
                data_files=self.val_list_path,
                **ds_kwargs
            )
        else:
            # Auto-split
            full_ds = MelDataset(
                data_dir=self.data_dir,
                data_files=self.train_list_path,
                **ds_kwargs
            )
            total = len(full_ds)
            val_size = int(total * self.val_split)
            train_size = total - val_size

            gen = torch.Generator().manual_seed(42)
            train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=gen)

            self.val_dataset = Subset(full_ds, val_idx.indices)

            if self.augmentations:
                train_kwargs = ds_kwargs.copy()
                train_kwargs["augmentations"] = self.augmentations
                full_ds_aug = MelDataset(
                    data_dir=self.data_dir,
                    data_files=self.train_list_path,
                    **train_kwargs
                )
                self.train_dataset = Subset(full_ds_aug, train_idx.indices)
            else:
                self.train_dataset = Subset(full_ds, train_idx.indices)

    # ------------------------------------------------------------------
    def _setup_multi_dataset(self):
        """
        Multi-dataset setup.

        Each entry in datasets list:
          - type: "paired" or "clean_with_noise"
          - For "paired": data_dir, train_list_path, val_list_path (optional)
          - For "clean_with_noise": clean_dir, noise_dir
        """
        train_datasets = []
        val_datasets = []

        ds_common = {
            "segment_size": self.segment_size or 8192,
            "sampling_rate": self.sampling_rate,
        }

        for i, ds_cfg in enumerate(self.datasets_config):
            ds_type = ds_cfg.get("type", "paired")
            print(f"\n[DataModule] Dataset {i+1}: type={ds_type}")

            if ds_type == "paired":
                data_dir = ds_cfg.get("data_dir", self.data_dir)
                train_list = ds_cfg.get("train_list_path", self.train_list_path)
                val_list = ds_cfg.get("val_list_path")
                val_split = ds_cfg.get("val_split", 0.1)

                train_kwargs = {
                    "data_dir": data_dir,
                    "data_files": train_list,
                    "segment_size": ds_common["segment_size"],
                    "sampling_rate": ds_common["sampling_rate"],
                }
                if self.use_preextracted_embeddings:
                    train_kwargs["return_audio_paths"] = True
                if self.augmentations:
                    train_kwargs["augmentations"] = self.augmentations

                if val_list:
                    val_kwargs = train_kwargs.copy()
                    val_kwargs.pop("augmentations", None)
                    val_kwargs["data_files"] = val_list

                    train_datasets.append(MelDataset(**train_kwargs))
                    val_datasets.append(MelDataset(**val_kwargs))
                else:
                    # Auto-split
                    full_kwargs = train_kwargs.copy()
                    full_kwargs.pop("augmentations", None)
                    full_ds = MelDataset(**full_kwargs)

                    total = len(full_ds)
                    val_size = int(total * val_split)
                    train_size = total - val_size
                    gen = torch.Generator().manual_seed(42 + i)
                    t_idx, v_idx = random_split(range(total), [train_size, val_size], generator=gen)

                    val_datasets.append(Subset(full_ds, v_idx.indices))

                    if self.augmentations:
                        aug_ds = MelDataset(**train_kwargs)
                        train_datasets.append(Subset(aug_ds, t_idx.indices))
                    else:
                        train_datasets.append(Subset(full_ds, t_idx.indices))

            elif ds_type == "clean_with_noise":
                clean_dir = ds_cfg["clean_dir"]
                noise_dir = ds_cfg.get("noise_dir", self.noise_dir)
                val_split = ds_cfg.get("val_split", 0.1)

                if not noise_dir:
                    raise ValueError(
                        f"Dataset {i+1}: 'noise_dir' required for clean_with_noise type"
                    )

                ds = CleanOnlyDataset(
                    clean_dir=clean_dir,
                    noise_dir=noise_dir,
                    segment_size=ds_common["segment_size"],
                    sampling_rate=ds_common["sampling_rate"],
                    augmentations=self.augmentations,
                    return_audio_paths=self.use_preextracted_embeddings,
                )

                # Split into train/val
                total = len(ds)
                val_size = int(total * val_split)
                train_size = total - val_size
                gen = torch.Generator().manual_seed(42 + i)
                t_idx, v_idx = random_split(range(total), [train_size, val_size], generator=gen)

                train_datasets.append(Subset(ds, t_idx.indices))
                val_datasets.append(Subset(ds, v_idx.indices))

            else:
                raise ValueError(f"Unknown dataset type: {ds_type}")

            print(f"  Train: {len(train_datasets[-1])}, Val: {len(val_datasets[-1])}")

        # Combine
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers,
        )
