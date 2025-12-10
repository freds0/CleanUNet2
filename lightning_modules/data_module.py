import pytorch_lightning as pl
from torch.utils.data import DataLoader
from spec_dataset import MelDataset, custom_collate_fn

class CleanUNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading noisy/clean speech pairs.
    
    Now supports passing extra dataset arguments (like sampling_rate, n_fft) via **kwargs.
    """
    def __init__(
        self,
        data_dir: str,
        train_list_path: str,
        val_list_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        persistent_workers: bool = False,
        segment_size: int = None,
        **kwargs # <--- Captura sampling_rate, n_fft, hop_size, etc. do YAML
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.segment_size = segment_size
        
        # Armazena os parâmetros extras para passar ao MelDataset
        self.dataset_kwargs = kwargs

        self.train_dataset = None
        self.val_dataset = None

    # ---------------------------------------------------------
    # Setup datasets
    # ---------------------------------------------------------
    def setup(self, stage=None):
        """
        Loads training and validation datasets into memory, injecting configuration parameters.
        """
        # Prepara os argumentos para o MelDataset
        dataset_params = self.dataset_kwargs.copy()
        
        # Garante que o segment_size explícito tenha prioridade
        if self.segment_size is not None:
            dataset_params["segment_size"] = self.segment_size

        # Instancia o Dataset de Treino
        self.train_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.train_list_path,
            split=True,       # Treino geralmente usa segmentos cortados
            shuffle=True,     # Embaralha a lista de arquivos
            **dataset_params  # Injeta: sampling_rate, n_fft, etc.
        )

        # Instancia o Dataset de Validação
        self.val_dataset = MelDataset(
            data_dir=self.data_dir,
            data_files=self.val_list_path,
            split=False,      # Validação pode usar áudio completo ou segmentado (ajuste se necessário)
            shuffle=False,    # Não embaralha lista na validação
            **dataset_params
        )

    # ---------------------------------------------------------
    # Train DataLoader
    # ---------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )

    # ---------------------------------------------------------
    # Validation DataLoader
    # ---------------------------------------------------------
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )