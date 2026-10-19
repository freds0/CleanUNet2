"""Lightning data module for the VoiceBank-DEMAND style (clean, noisy) file lists."""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import PairDataset, seed_worker


class CleanUNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for noisy/clean speech pairs.

    - Training: random crops of `segment_size` samples, shuffled, drop_last.
    - Validation: whole utterances of a held-out split (batch_size=1, fixed order), so validation
      metrics are deterministic and never computed on the test set.

    Args:
        data_dir: root directory the filelist paths are relative to.
        train_list_path / val_list_path: filelists with lines "clean|noisy".
        batch_size: training batch size.
        segment_size: training crop length in samples.
        val_max_files: optionally evaluate only the first N validation files.
        concat_utterances / crossfade: training examples made of n utterances joined with a crossfade
            of `crossfade` samples (long training sequences, IMPROVEMENT_PLAN.md N7a).
        ext_noise: training crops whose noise is replaced by another corpus (dataset.py, noise_bank.py):
            {dir, p, snr_db: [low, high], weights: {noise, babble, music}, babble_talkers: [min, max]}.
    """
    def __init__(self, data_dir: str, train_list_path: str, val_list_path: str, batch_size: int = 16,
                 num_workers: int = 8, persistent_workers: bool = True, segment_size: int = 32768,
                 sample_rate: int = 16000, val_max_files: int = None, concat_utterances: int = 1,
                 crossfade: int = 160, ext_noise: dict = None, **unused):
        super().__init__()
        self.concat_utterances = concat_utterances
        self.crossfade = crossfade
        self.ext_noise = ext_noise
        self.data_dir = data_dir
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.val_max_files = val_max_files
        if unused:
            print(f"[INFO] CleanUNetDataModule ignores keys: {sorted(unused)}")

    def setup(self, stage=None):
        """Build the training set (random crops, optional external noise) and the validation set (whole files)."""
        self.train_dataset = PairDataset(self.data_dir, self.train_list_path, segment_size=self.segment_size,
                                         sample_rate=self.sample_rate, concat_utterances=self.concat_utterances,
                                         crossfade=self.crossfade, ext_noise=self.ext_noise)
        self.val_dataset = PairDataset(self.data_dir, self.val_list_path, segment_size=None,
                                       sample_rate=self.sample_rate, max_files=self.val_max_files)

    def train_dataloader(self):
        """Shuffled training batches; drop_last, seeded workers."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=True, worker_init_fn=seed_worker)

    def val_dataloader(self):
        """Validation utterances one at a time, in a fixed order."""
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=min(4, self.num_workers),
                          persistent_workers=self.persistent_workers, worker_init_fn=seed_worker)
