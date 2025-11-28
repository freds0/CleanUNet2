import pytorch_lightning as pl
from torch.utils.data import DataLoader
from spec_dataset import MelDataset
from spec_dataset import custom_collate_fn

class CleanUNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_list_path, val_list_path, batch_size=8, num_workers=4, persistent_workers: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        self.train_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.train_list_path,
            # outros parâmetros que você precisa
        )
        self.val_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.val_list_path,
            # outros parâmetros que você precisa
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,  # 👈 Aqui está o uso!
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,  # 👈 E aqui também
            persistent_workers=self.persistent_workers
        )
