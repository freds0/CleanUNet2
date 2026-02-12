import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from spec_dataset import MelDataset, custom_collate_fn
import torch
import math


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
        val_list_path (str, optional): Path to validation CSV/filelist.
            If not provided, will use val_split to split training data.
        val_split (float, optional): Fraction of training data to use for validation (0.0 to 1.0).
            Only used if val_list_path is not provided. Example: 0.1 = 10% for validation.
        batch_size (int): Batch size for all dataloaders.
        num_workers (int): Number of worker processes for dataloading.
        persistent_workers (bool): Keep workers alive between epochs (faster).
        segment_size (int): Length of audio segments for training (samples).
        sampling_rate (int): Target sampling rate for audio (default: 16000 Hz).
            Audio files will be automatically resampled to this rate.
        augmentations (list): List of augmentation configurations for training.
    """
    def __init__(
        self,
        data_dir: str,
        train_list_path: str,
        val_list_path: str = None,
        val_split: float = None,
        batch_size: int = 8,
        num_workers: int = 4,
        persistent_workers: bool = False,
        segment_size: int = None,
        sampling_rate: int = 16000,
        augmentations: list = None,
        use_xvector_cache: bool = False
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
        self.use_xvector_cache = use_xvector_cache

        # Validar configuração
        if self.val_list_path is None and self.val_split is None:
            raise ValueError(
                "Você deve especificar 'val_list_path' (caminho para arquivo de validação) "
                "OU 'val_split' (porcentagem para split automático, ex: 0.1 para 10%)"
            )

        if self.val_split is not None:
            if not (0.0 < self.val_split < 1.0):
                raise ValueError(f"val_split deve estar entre 0.0 e 1.0, recebido: {self.val_split}")

        self.train_dataset = None
        self.val_dataset = None

    # ---------------------------------------------------------
    # Setup datasets
    # ---------------------------------------------------------
    def setup(self, stage=None):
        """
        Called by Lightning at the beginning of training/validation/testing.

        Loads training and validation datasets into memory.
        Supports two modes:
        1. Separate validation file (val_list_path)
        2. Automatic split from training data (val_split)
        """
        dataset_kwargs = {}
        if self.segment_size is not None:
            dataset_kwargs["segment_size"] = self.segment_size
        if self.sampling_rate is not None:
            dataset_kwargs["sampling_rate"] = self.sampling_rate

        # OPÇÃO 1: val_list_path fornecido - usar arquivo separado para validação
        if self.val_list_path is not None:
            print(f"Usando arquivo separado para validação: {self.val_list_path}")

            # Select dataset class based on cache configuration
            if self.use_xvector_cache:
                print("[INFO] Using XVectorMelDataset with file path tracking for caching")
                from xvector_dataset import XVectorMelDataset
                dataset_class = XVectorMelDataset
                # Add return_paths parameter
                dataset_kwargs["return_paths"] = True
            else:
                dataset_class = MelDataset

            # Training dataset with augmentation
            train_kwargs = dataset_kwargs.copy()
            if self.augmentations is not None:
                train_kwargs["augmentations"] = self.augmentations

            self.train_dataset = dataset_class(
                data_dir=self.data_dir,
                data_files=self.train_list_path,
                **train_kwargs
            )

            # Validation dataset WITHOUT augmentation
            self.val_dataset = dataset_class(
                data_dir=self.data_dir,
                data_files=self.val_list_path,
                **dataset_kwargs
            )

            print(f"✅ Dataset de treino: {len(self.train_dataset)} amostras")
            print(f"✅ Dataset de validação: {len(self.val_dataset)} amostras")

        # OPÇÃO 2: val_split fornecido - fazer split automático
        else:
            print(f"Fazendo split automático com {self.val_split*100:.1f}% para validação")

            # Carregar dataset completo SEM augmentation primeiro (para fazer split justo)
            full_dataset = MelDataset(
                data_dir=self.data_dir,
                data_files=self.train_list_path,
                **dataset_kwargs
            )

            # Calcular tamanhos dos splits
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            print(f"Total de amostras: {total_size}")
            print(f"  - Treino: {train_size} ({(1-self.val_split)*100:.1f}%)")
            print(f"  - Validação: {val_size} ({self.val_split*100:.1f}%)")

            # Fazer split aleatório determinístico
            generator = torch.Generator().manual_seed(42)  # Seed fixo para reprodutibilidade
            train_indices, val_indices = random_split(
                range(total_size),
                [train_size, val_size],
                generator=generator
            )

            # Dataset de validação (subset sem augmentation)
            self.val_dataset = Subset(full_dataset, val_indices.indices)

            # Dataset de treino com augmentation
            if self.augmentations is not None:
                print(f"Aplicando augmentation ao dataset de treino")
                # Carregar novamente COM augmentation
                train_kwargs = dataset_kwargs.copy()
                train_kwargs["augmentations"] = self.augmentations
                full_dataset_with_aug = MelDataset(
                    data_dir=self.data_dir,
                    data_files=self.train_list_path,
                    **train_kwargs
                )
                self.train_dataset = Subset(full_dataset_with_aug, train_indices.indices)
            else:
                self.train_dataset = Subset(full_dataset, train_indices.indices)

            print(f"✅ Split concluído com sucesso!")

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
        # Select collate function based on cache configuration
        if self.use_xvector_cache:
            from xvector_dataset import xvector_collate_fn
            collate_fn = xvector_collate_fn
        else:
            collate_fn = custom_collate_fn

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
        # Select collate function based on cache configuration
        if self.use_xvector_cache:
            from xvector_dataset import xvector_collate_fn
            collate_fn = xvector_collate_fn
        else:
            collate_fn = custom_collate_fn

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers
        )

