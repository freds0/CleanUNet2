import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from spec_dataset import MelDataset, custom_collate_fn
import torch
import math

class CleanUNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading noisy/clean speech pairs.

    Now supports:
    - Extra dataset arguments (like sampling_rate, n_fft) via **kwargs
    - Data augmentation configuration
    - Automatic validation split (val_split parameter)

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
        augmentations (list): List of augmentation configurations (optional).
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
        augmentations: list = None,
        **kwargs # Captures sampling_rate, n_fft, hop_size, etc. from YAML
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
        self.augmentations = augmentations

        # Validar configuração
        if self.val_list_path is None and self.val_split is None:
            raise ValueError(
                "Você deve especificar 'val_list_path' (caminho para arquivo de validação) "
                "OU 'val_split' (porcentagem para split automático, ex: 0.1 para 10%)"
            )

        if self.val_split is not None:
            if not (0.0 < self.val_split < 1.0):
                raise ValueError(f"val_split deve estar entre 0.0 e 1.0, recebido: {self.val_split}")

        # Store extra parameters to pass to MelDataset
        self.dataset_kwargs = kwargs

        self.train_dataset = None
        self.val_dataset = None

    # ---------------------------------------------------------
    # Setup datasets
    # ---------------------------------------------------------
    def setup(self, stage=None):
        """
        Loads training and validation datasets into memory, injecting configuration parameters.
        Augmentation is only applied to training dataset.
        Supports two modes:
        1. Separate validation file (val_list_path)
        2. Automatic split from training data (val_split)
        """
        # Prepare arguments for MelDataset
        dataset_params = self.dataset_kwargs.copy()

        # Ensure explicit segment_size takes priority
        if self.segment_size is not None:
            dataset_params["segment_size"] = self.segment_size

        # OPÇÃO 1: val_list_path fornecido - usar arquivo separado para validação
        if self.val_list_path is not None:
            print(f"Usando arquivo separado para validação: {self.val_list_path}")

            # Add augmentation to training dataset only (not validation)
            train_kwargs = dataset_params.copy()
            if self.augmentations is not None:
                train_kwargs["augmentations"] = self.augmentations
                print(f"[INFO] Data augmentation will be applied to training dataset")

            # Instantiate Training Dataset (with augmentation)
            self.train_dataset = MelDataset(
                data_dir=self.data_dir,
                data_files=self.train_list_path,
                split=True,       # Training uses segmented audio
                shuffle=True,     # Shuffle file list
                **train_kwargs    # Includes augmentation config
            )

            # Instantiate Validation Dataset (without augmentation)
            self.val_dataset = MelDataset(
                data_dir=self.data_dir,
                data_files=self.val_list_path,
                split=False,      # Validation can use full audio or segmented
                shuffle=False,    # Don't shuffle validation list
                **dataset_params  # No augmentation for validation
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
                split=True,
                shuffle=True,
                **dataset_params
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
                train_kwargs = dataset_params.copy()
                train_kwargs["augmentations"] = self.augmentations
                full_dataset_with_aug = MelDataset(
                    data_dir=self.data_dir,
                    data_files=self.train_list_path,
                    split=True,
                    shuffle=True,
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