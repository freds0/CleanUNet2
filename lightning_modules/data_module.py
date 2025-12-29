"""
Speech Enhancement Dataset
Loads paired clean and noisy audio files for speech enhancement training
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random


class SpeechEnhancementDataset(Dataset):
    """
    Dataset class for speech enhancement.
    Loads paired clean and noisy audio files.
    """
    
    def __init__(
        self,
        filelist_path,
        root_dir=None,  # Added root_dir argument
        segment_length=None,
        sampling_rate=16000,
        split='train',
        augment=False
    ):
        """
        Args:
            filelist_path (str): Path to the filelist file.
            root_dir (str, optional): Root directory to prepend to relative paths.
            ...
        """
        super().__init__()
        
        self.filelist_path = filelist_path
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.split = split
        self.augment = augment
        
        # Load file pairs
        self.audio_files = self._load_filelist(filelist_path)
        
        print(f"[Dataset] Loaded {len(self.audio_files)} audio pairs from {filelist_path}")
        if root_dir:
            print(f"[Dataset] Root directory: {root_dir}")
        print(f"[Dataset] Split: {split}")
        print(f"[Dataset] Segment length: {segment_length if segment_length else 'Full length'}")
        print(f"[Dataset] Sampling rate: {sampling_rate} Hz")
        print(f"[Dataset] Augmentation: {'Enabled' if augment else 'Disabled'}")
    
    def _load_filelist(self, filelist_path):
        """Load audio file pairs from filelist."""
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")
        
        audio_files = []
        
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse line (format: clean_path|noisy_path)
                parts = line.split('|')
                
                if len(parts) != 2:
                    print(f"[Warning] Skipping invalid line {line_num}: {line}")
                    continue
                
                clean_path, noisy_path = parts[0].strip(), parts[1].strip()
                
                # Prepend root_dir if provided
                if self.root_dir:
                    clean_path = os.path.join(self.root_dir, clean_path)
                    noisy_path = os.path.join(self.root_dir, noisy_path)
                
                # Check if files exist
                if not os.path.exists(clean_path):
                    print(f"[Warning] Clean file not found: {clean_path}")
                    continue
                
                if not os.path.exists(noisy_path):
                    print(f"[Warning] Noisy file not found: {noisy_path}")
                    continue
                
                audio_files.append((clean_path, noisy_path))
        
        if len(audio_files) == 0:
            raise ValueError(f"No valid audio pairs found in {filelist_path}")
        
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        clean_path, noisy_path = self.audio_files[idx]
        
        # Load audio files
        clean_audio, clean_sr = self._load_audio(clean_path)
        noisy_audio, noisy_sr = self._load_audio(noisy_path)
        
        # Resample if necessary
        if clean_sr != self.sampling_rate:
            clean_audio = self._resample(clean_audio, clean_sr, self.sampling_rate)
        
        if noisy_sr != self.sampling_rate:
            noisy_audio = self._resample(noisy_audio, noisy_sr, self.sampling_rate)
        
        # Ensure both audios have the same length
        min_length = min(clean_audio.shape[-1], noisy_audio.shape[-1])
        clean_audio = clean_audio[..., :min_length]
        noisy_audio = noisy_audio[..., :min_length]
        
        # Extract segment if segment_length is specified
        if self.segment_length is not None and self.split == 'train':
            clean_audio, noisy_audio = self._extract_segment(
                clean_audio, noisy_audio, self.segment_length
            )
        
        # Apply augmentation if enabled (only during training)
        if self.augment and self.split == 'train':
            clean_audio, noisy_audio = self._augment(clean_audio, noisy_audio)
        
        # Generate file ID (use filename without extension)
        file_id = os.path.splitext(os.path.basename(clean_path))[0]
        
        # Ensure audio is in float32
        clean_audio = clean_audio.float()
        noisy_audio = noisy_audio.float()
        
        # Normalize to [-1, 1] range
        clean_audio = self._normalize_audio(clean_audio)
        noisy_audio = self._normalize_audio(noisy_audio)
        
        return {
            'clean': clean_audio,
            'noisy': noisy_audio,
            'file_id': file_id
        }
    
    def _load_audio(self, audio_path):
        try:
            audio, sr = torchaudio.load(audio_path)
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def _resample(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resampler(audio)
    
    def _extract_segment(self, clean_audio, noisy_audio, segment_length):
        audio_length = clean_audio.shape[-1]
        
        # If audio is shorter than segment_length, pad it
        if audio_length < segment_length:
            pad_length = segment_length - audio_length
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_length))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_length))
            start_idx = 0
        else:
            # Random start position
            max_start = audio_length - segment_length
            start_idx = random.randint(0, max_start)
        
        # Extract segment
        end_idx = start_idx + segment_length
        clean_segment = clean_audio[..., start_idx:end_idx]
        noisy_segment = noisy_audio[..., start_idx:end_idx]
        
        return clean_segment, noisy_segment
    
    def _normalize_audio(self, audio):
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _augment(self, clean_audio, noisy_audio):
        # Apply random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            clean_audio = clean_audio * gain
            noisy_audio = noisy_audio * gain
        
        # Apply random polarity inversion
        if random.random() < 0.1:
            clean_audio = -clean_audio
            noisy_audio = -noisy_audio
        
        return clean_audio, noisy_audio
    
    def get_audio_length(self, idx):
        clean_path, _ = self.audio_files[idx]
        try:
            info = torchaudio.info(clean_path)
            return info.num_frames
        except Exception as e:
            print(f"[Warning] Could not get audio info for {clean_path}: {str(e)}")
            return 0


def collate_fn(batch):
    """
    Custom collate function for batching audio samples.
    """
    # Find maximum length in batch
    max_length = max([item['clean'].shape[-1] for item in batch])
    
    clean_batch = []
    noisy_batch = []
    file_ids = []
    
    for item in batch:
        clean = item['clean']
        noisy = item['noisy']
        
        # Pad if necessary
        if clean.shape[-1] < max_length:
            pad_length = max_length - clean.shape[-1]
            clean = torch.nn.functional.pad(clean, (0, pad_length))
            noisy = torch.nn.functional.pad(noisy, (0, pad_length))
        
        clean_batch.append(clean)
        noisy_batch.append(noisy)
        file_ids.append(item['file_id'])
    
    return {
        'clean': torch.stack(clean_batch),
        'noisy': torch.stack(noisy_batch),
        'file_id': file_ids
    }


class SpeechEnhancementDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for speech enhancement.
    Wraps SpeechEnhancementDataset for easy use with Lightning Trainer.
    """
    def __init__(
        self,
        train_filelist,
        val_filelist,
        batch_size,
        num_workers,
        root_dir=None,  # <--- THIS IS THE CRITICAL ADDITION
        sample_rate=16000,
        segment_length=None
    ):
        super().__init__()
        self.train_filelist = train_filelist
        self.val_filelist = val_filelist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir  # Store root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.
        """
        # Create training dataset
        if stage == 'fit' or stage is None:
            self.train_ds = SpeechEnhancementDataset(
                filelist_path=self.train_filelist,
                root_dir=self.root_dir,  # Pass root_dir to Dataset
                segment_length=self.segment_length,
                sampling_rate=self.sample_rate,
                split='train',
                augment=True
            )
            
            # Create validation dataset
            self.val_ds = SpeechEnhancementDataset(
                filelist_path=self.val_filelist,
                root_dir=self.root_dir,  # Pass root_dir to Dataset
                segment_length=None, 
                sampling_rate=self.sample_rate,
                split='val',
                augment=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


# Example usage and testing
if __name__ == '__main__':
    """
    Test the dataset implementation.
    """
    
    print("="*60)
    print("Testing SpeechEnhancementDataset")
    print("="*60)
    
    # Create a dummy filelist for testing
    dummy_filelist = 'test_filelist.txt'
    
    # Note: You need to create actual audio files or modify paths for real testing
    # This is just an example structure
    
    # Test dataset creation
    try:
        dataset = SpeechEnhancementDataset(
            filelist_path=dummy_filelist,
            segment_length=64000,  # 4 seconds at 16kHz
            sampling_rate=16000,
            split='train',
            augment=False
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Number of samples: {len(dataset)}")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            
            print(f"\n✓ Sample retrieved successfully")
            print(f"  Clean audio shape: {sample['clean'].shape}")
            print(f"  Noisy audio shape: {sample['noisy'].shape}")
            print(f"  File ID: {sample['file_id']}")
            
            # Test audio properties
            clean_audio = sample['clean']
            print(f"\n  Audio properties:")
            print(f"    Min value: {clean_audio.min().item():.4f}")
            print(f"    Max value: {clean_audio.max().item():.4f}")
            print(f"    Mean value: {clean_audio.mean().item():.4f}")
            print(f"    Std value: {clean_audio.std().item():.4f}")
        
        # Test with DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        print(f"\n✓ DataLoader created successfully")
        print(f"  Number of batches: {len(dataloader)}")
        
        # Test one batch
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n✓ Batch {batch_idx + 1} retrieved:")
            print(f"  Clean batch shape: {batch['clean'].shape}")
            print(f"  Noisy batch shape: {batch['noisy'].shape}")
            print(f"  File IDs: {batch['file_id']}")
            break
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo test the dataset, create a filelist with the format:")
        print("  /path/to/clean1.wav|/path/to/noisy1.wav")
        print("  /path/to/clean2.wav|/path/to/noisy2.wav")
        print("  ...")
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
