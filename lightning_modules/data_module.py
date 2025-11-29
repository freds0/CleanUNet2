import pytorch_lightning as pl
from torch.utils.data import DataLoader
from spec_dataset import MelDataset, custom_collate_fn


class CleanUNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading noisy/clean speech pairs.

    This module wraps:
    - Training dataset
    - Validation dataset
    - DataLoaders with multi-worker support
    - Custom collate function for variable-length spectrograms

    Args:
        data_dir (str): Root directory containing all audio files.
        train_list_path (str): Path to training CSV/filelist.
        val_list_path (str): Path to validation CSV/filelist.
        batch_size (int): Batch size for all dataloaders.
        num_workers (int): Number of worker processes for dataloading.
        persistent_workers (bool): Keep workers alive between epochs (faster).
    """
    def __init__(
        self,
        data_dir: str,
        train_list_path: str,
        val_list_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        persistent_workers: bool = False,
        segment_size: int = None
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.segment_size = segment_size

        self.train_dataset = None
        self.val_dataset = None

    # ---------------------------------------------------------
    # Setup datasets
    # ---------------------------------------------------------
    def setup(self, stage=None):
        """
        Called by Lightning at the beginning of training/validation/testing.

        Loads training and validation datasets into memory.
        """
        dataset_kwargs = {}
        if self.segment_size is not None:
            dataset_kwargs["segment_size"] = self.segment_size

        self.train_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.train_list_path,
            **dataset_kwargs
        )

        self.val_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.val_list_path,
            **dataset_kwargs
        )

    # ---------------------------------------------------------
    # Train DataLoader
    # ---------------------------------------------------------
    def train_dataloader(self):
        """
        Returns DataLoader used in training.

        Uses:
        - custom_collate_fn for handling variable-length spectrograms
        - shuffling enabled
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers
        )

    # ---------------------------------------------------------
    # Validation DataLoader
    # ---------------------------------------------------------
    def val_dataloader(self):
        """
        Returns DataLoader used during validation.

        No shuffling to ensure deterministic metrics.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers
        )

