"""
Training script for CleanUNet2 with X-Vector integration
Supports two-stage training strategy
"""

import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_modules.cleanunet_xvector_module import CleanUNet2XVectorModule
from lightning_modules.data_module import SpeechEnhancementDataModule
import torch
import os


def train_stage1(config):
    """
    Train Stage 1: With X-Vectors.
    Extracts and integrates X-Vectors, saves latent vectors for Stage 2.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        best_checkpoint_path (str): Path to best model checkpoint
    """
    print("\n" + "="*70)
    print("STAGE 1: TRAINING WITH X-VECTORS")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # Data module
    print("[Train] Creating data module...")
    data_module = SpeechEnhancementDataModule(
        train_filelist=config['data']['train_filelist'],
        val_filelist=config['data']['val_filelist'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        root_dir=config['data'].get('root_dir', None),
        sample_rate=config['data'].get('sample_rate', 16000),
        segment_length=config['data'].get('segment_length', None)
    )
    
    # Model
    print("[Train] Creating model module...")
    model = CleanUNet2XVectorModule(config, stage='stage1')
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/{config['experiment_name']}/checkpoints",
        filename='epoch={epoch:03d}-pesq={val/pesq:.3f}',
        monitor='val/pesq',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/pesq',
        patience=config.get('early_stopping_patience', 15),
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=config['experiment_name'],
        version=config.get('version', None)
    )
    
    # Trainer
    print("[Train] Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config['trainer'].get('gpus', 1),
        precision=config['trainer'].get('precision', 32),
        accumulate_grad_batches=config['trainer'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['trainer'].get('gradient_clip_val', 5.0),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=config['trainer'].get('val_check_interval', 1.0),
        deterministic=False
    )
    
    # Train
    print("\n[Train] Starting training...\n")
    trainer.fit(model, data_module)
    
    print("\n" + "="*70)
    print("STAGE 1 TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Latents saved to: {config.get('latents_dir', 'stored_latents')}")
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Best PESQ: {checkpoint_callback.best_model_score:.4f}\n")
    
    return checkpoint_callback.best_model_path


def train_stage2(config, stage1_checkpoint=None):
    """
    Train Stage 2: Without X-Vectors (replicating latents).
    Trains the model to replicate latent vectors from Stage 1.
    
    Args:
        config (dict): Configuration dictionary
        stage1_checkpoint (str): Path to Stage 1 checkpoint for weight initialization
    """
    print("\n" + "="*70)
    print("STAGE 2: TRAINING WITHOUT X-VECTORS (REPLICATING LATENTS)")
    print("="*70 + "\n")
    
    # Set random seed
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # Data module
    print("[Train] Creating data module...")
    data_module = SpeechEnhancementDataModule(
        train_filelist=config['data']['train_filelist'],
        val_filelist=config['data']['val_filelist'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        root_dir=config['data'].get('root_dir', None),
        sample_rate=config['data'].get('sample_rate', 16000),
        segment_length=config['data'].get('segment_length', None)
    )
    
    # Model
    print("[Train] Creating model module...")
    if stage1_checkpoint and config.get('load_from_stage1', True):
        print(f"[Train] Loading weights from Stage 1: {stage1_checkpoint}")
        model = CleanUNet2XVectorModule.load_from_checkpoint(
            stage1_checkpoint,
            config=config,
            stage='stage2',
            strict=False  # Allow missing keys for latent_predictor
        )
    else:
        model = CleanUNet2XVectorModule(config, stage='stage2')
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/{config['experiment_name']}/checkpoints",
        filename='epoch={epoch:03d}-pesq={val/pesq:.3f}',
        monitor='val/pesq',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/pesq',
        patience=config.get('early_stopping_patience', 15),
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=config['experiment_name'],
        version=config.get('version', None)
    )
    
    # Trainer
    print("[Train] Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config['trainer'].get('gpus', 1),
        precision=config['trainer'].get('precision', 32),
        accumulate_grad_batches=config['trainer'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['trainer'].get('gradient_clip_val', 5.0),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=config['trainer'].get('val_check_interval', 1.0),
        deterministic=False
    )
    
    # Train
    print("\n[Train] Starting training...\n")
    trainer.fit(model, data_module)
    
    print("\n" + "="*70)
    print("STAGE 2 TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Model is now causal and ready for deployment!")
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Best PESQ: {checkpoint_callback.best_model_score:.4f}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train CleanUNet2 with X-Vector integration'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['stage1', 'stage2', 'both'],
        default='stage1',
        help='Training stage: stage1 (with X-Vectors), stage2 (without X-Vectors), or both'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n[Main] Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"[Main] Experiment name: {config['experiment_name']}")
    print(f"[Main] Training stage(s): {args.stage}\n")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    if 'latents_dir' in config:
        os.makedirs(config['latents_dir'], exist_ok=True)
    
    # Train based on selected stage
    if args.stage == 'stage1' or args.stage == 'both':
        # Train Stage 1
        stage1_ckpt = train_stage1(config)
        
        # If training both stages, proceed to Stage 2
        if args.stage == 'both':
            # Load Stage 2 configuration
            stage2_config_path = args.config.replace('stage1', 'stage2')
            
            if not os.path.exists(stage2_config_path):
                print(f"\n[Warning] Stage 2 config not found: {stage2_config_path}")
                print("[Warning] Skipping Stage 2 training.")
            else:
                print(f"\n[Main] Loading Stage 2 configuration from: {stage2_config_path}")
                
                with open(stage2_config_path, 'r') as f:
                    config_stage2 = yaml.safe_load(f)
                
                # Train Stage 2 with Stage 1 checkpoint
                train_stage2(config_stage2, stage1_checkpoint=stage1_ckpt)
    
    elif args.stage == 'stage2':
        # Train only Stage 2
        stage1_ckpt = config.get('stage1_checkpoint', None)
        train_stage2(config, stage1_checkpoint=stage1_ckpt)
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()