#!/usr/bin/env python3
"""
Hyperparameter sweep for CleanUNet2.

Generates YAML configs for each experiment, runs training for a fixed number
of epochs, and collects validation metrics (PESQ, STOI, SI-SDR, val_loss).

Usage:
    # Run all experiments sequentially:
    python run_experiments.py --epochs 30

    # Run only a specific experiment group:
    python run_experiments.py --epochs 30 --group lr

    # Run a single experiment by name:
    python run_experiments.py --epochs 30 --name exp01_lr_1e-4

    # List all experiments without running:
    python run_experiments.py --list

    # Collect results from already-finished experiments:
    python run_experiments.py --collect-only
"""

import os
import sys
import copy
import yaml
import json
import argparse
import subprocess
import glob
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Base configuration (mirrors configs/train.yaml defaults)
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": 30,           # overridden by --epochs
        "precision": "32-true",
        "log_every_n_steps": 50,
        "check_val_every_n_epoch": 5,
        "gradient_clip_val": 1.0,
        "deterministic": False,
        "benchmark": True,
    },
    "seed": 1234,
    "model": {
        "lr": 1e-5,
        "sample_rate": 16000,
        "conditioning_type": "addition",
        "loss_config": {
            "ell_p": 1,
            "ell_p_lambda": 1.0,
            "stft_lambda": 1.0,
            "sc_lambda": 0.5,
            "mag_lambda": 0.5,
            "stft_config": {
                "fft_sizes": [512, 1024, 2048],
                "hop_sizes": [50, 120, 240],
                "win_lengths": [240, 600, 1200],
            },
        },
    },
    "data": {
        "data_dir": "/home/fred/Projetos/DATASETS/VoiceBank-DEMAND-16k/",
        "train_list_path": "filelists/train.csv",
        "val_list_path": "filelists/test.csv",
        "batch_size": 10,
        "num_workers": 8,
        "persistent_workers": True,
    },
}

# ---------------------------------------------------------------------------
# Experiment definitions
# Each entry: (name, group, description, dict of nested overrides)
# ---------------------------------------------------------------------------
EXPERIMENTS = []


def _add(name, group, desc, overrides):
    EXPERIMENTS.append((name, group, desc, overrides))


# ===== GROUP 1: Learning Rate =====
_add("exp01_lr_1e-3", "lr", "LR = 1e-3",
     {"model.lr": 1e-3})
_add("exp02_lr_5e-4", "lr", "LR = 5e-4",
     {"model.lr": 5e-4})
_add("exp03_lr_1e-4", "lr", "LR = 1e-4",
     {"model.lr": 1e-4})
_add("exp04_lr_5e-5", "lr", "LR = 5e-5",
     {"model.lr": 5e-5})
_add("exp05_lr_1e-5", "lr", "LR = 1e-5 (baseline)",
     {"model.lr": 1e-5})
_add("exp06_lr_5e-6", "lr", "LR = 5e-6",
     {"model.lr": 5e-6})

# ===== GROUP 2: Conditioning Type =====
_add("exp07_cond_addition", "conditioning", "Conditioning: addition",
     {"model.conditioning_type": "addition"})
_add("exp08_cond_concatenation", "conditioning", "Conditioning: concatenation",
     {"model.conditioning_type": "concatenation"})
_add("exp09_cond_film", "conditioning", "Conditioning: FiLM",
     {"model.conditioning_type": "film"})

# ===== GROUP 3: Loss Function Type (L1 vs L2) =====
_add("exp10_loss_L1", "loss_type", "Reconstruction: L1",
     {"model.loss_config.ell_p": 1})
_add("exp11_loss_L2", "loss_type", "Reconstruction: L2 (MSE)",
     {"model.loss_config.ell_p": 2})

# ===== GROUP 4: Loss Weights (waveform / spec / phase balance) =====
_add("exp12_w_wave10_spec1_phase1", "loss_weights",
     "Weights: wave=10, spec=1, phase=1 (baseline)",
     {"model.weight_waveform": 10.0,
      "model.weight_spec": 1.0,
      "model.weight_phase": 1.0})
_add("exp13_w_wave5_spec1_phase1", "loss_weights",
     "Weights: wave=5, spec=1, phase=1",
     {"model.weight_waveform": 5.0,
      "model.weight_spec": 1.0,
      "model.weight_phase": 1.0})
_add("exp14_w_wave10_spec5_phase1", "loss_weights",
     "Weights: wave=10, spec=5, phase=1",
     {"model.weight_waveform": 10.0,
      "model.weight_spec": 5.0,
      "model.weight_phase": 1.0})
_add("exp15_w_wave10_spec1_phase5", "loss_weights",
     "Weights: wave=10, spec=1, phase=5",
     {"model.weight_waveform": 10.0,
      "model.weight_spec": 1.0,
      "model.weight_phase": 5.0})
_add("exp16_w_wave10_spec5_phase5", "loss_weights",
     "Weights: wave=10, spec=5, phase=5",
     {"model.weight_waveform": 10.0,
      "model.weight_spec": 5.0,
      "model.weight_phase": 5.0})
_add("exp17_w_wave1_spec1_phase1", "loss_weights",
     "Weights: wave=1, spec=1, phase=1 (equal)",
     {"model.weight_waveform": 1.0,
      "model.weight_spec": 1.0,
      "model.weight_phase": 1.0})

# ===== GROUP 5: STFT Loss Weights =====
_add("exp18_stft_lambda_0.5", "stft_weights",
     "stft_lambda=0.5",
     {"model.loss_config.stft_lambda": 0.5})
_add("exp19_stft_lambda_2.0", "stft_weights",
     "stft_lambda=2.0",
     {"model.loss_config.stft_lambda": 2.0})
_add("exp20_sc0.8_mag0.2", "stft_weights",
     "sc_lambda=0.8, mag_lambda=0.2",
     {"model.loss_config.sc_lambda": 0.8,
      "model.loss_config.mag_lambda": 0.2})
_add("exp21_sc0.2_mag0.8", "stft_weights",
     "sc_lambda=0.2, mag_lambda=0.8",
     {"model.loss_config.sc_lambda": 0.2,
      "model.loss_config.mag_lambda": 0.8})

# ===== GROUP 6: Batch Size =====
_add("exp22_bs4", "batch_size", "Batch size = 4",
     {"data.batch_size": 4})
_add("exp23_bs8", "batch_size", "Batch size = 8",
     {"data.batch_size": 8})
_add("exp24_bs16", "batch_size", "Batch size = 16",
     {"data.batch_size": 16})

# ===== GROUP 7: Precision =====
_add("exp25_fp16_mixed", "precision", "Mixed precision (16-mixed)",
     {"trainer.precision": "16-mixed"})
_add("exp26_bf16_mixed", "precision", "BFloat16 mixed precision",
     {"trainer.precision": "bf16-mixed"})

# ===== GROUP 8: Gradient Clipping =====
_add("exp27_gradclip_0.5", "grad_clip", "Gradient clip = 0.5",
     {"trainer.gradient_clip_val": 0.5})
_add("exp28_gradclip_none", "grad_clip", "No gradient clipping",
     {"trainer.gradient_clip_val": None})
_add("exp29_gradclip_5.0", "grad_clip", "Gradient clip = 5.0",
     {"trainer.gradient_clip_val": 5.0})

# ===== GROUP 9: Combined Best Candidates =====
_add("exp30_combo_A", "combo",
     "LR=1e-4 + FiLM + L1 + wave=10,spec=5,phase=1",
     {"model.lr": 1e-4,
      "model.conditioning_type": "film",
      "model.loss_config.ell_p": 1,
      "model.weight_waveform": 10.0,
      "model.weight_spec": 5.0,
      "model.weight_phase": 1.0})
_add("exp31_combo_B", "combo",
     "LR=5e-5 + concat + L1 + wave=10,spec=1,phase=5",
     {"model.lr": 5e-5,
      "model.conditioning_type": "concatenation",
      "model.loss_config.ell_p": 1,
      "model.weight_waveform": 10.0,
      "model.weight_spec": 1.0,
      "model.weight_phase": 5.0})
_add("exp32_combo_C", "combo",
     "LR=1e-4 + addition + L1 + fp16-mixed + bs=16",
     {"model.lr": 1e-4,
      "model.conditioning_type": "addition",
      "trainer.precision": "16-mixed",
      "data.batch_size": 16})
_add("exp33_combo_D", "combo",
     "LR=5e-4 + FiLM + L2 + wave=5,spec=5,phase=5",
     {"model.lr": 5e-4,
      "model.conditioning_type": "film",
      "model.loss_config.ell_p": 2,
      "model.weight_waveform": 5.0,
      "model.weight_spec": 5.0,
      "model.weight_phase": 5.0})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(d, dotted_key, value):
    """Set a value in a nested dict using dot notation: 'a.b.c' -> d['a']['b']['c']."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_config(name, overrides, epochs, val_every):
    """Build a full config dict from BASE_CONFIG + overrides."""
    cfg = copy.deepcopy(BASE_CONFIG)

    # Set epochs and validation frequency
    cfg["trainer"]["max_epochs"] = epochs
    cfg["trainer"]["check_val_every_n_epoch"] = val_every

    # Apply experiment-specific overrides
    for key, val in overrides.items():
        _set_nested(cfg, key, val)

    # Unique output directories per experiment
    exp_dir = f"experiments/{name}"
    cfg["checkpoint_dir"] = f"{exp_dir}/checkpoints"
    cfg["logging_dir"] = exp_dir

    # Callbacks with experiment-specific paths
    cfg["callbacks"] = {
        "best_checkpoint": {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "val_loss",
            "dirpath": f"{exp_dir}/checkpoints",
            "filename": f"best-{{epoch:02d}}-{{val_loss:.4f}}",
            "save_top_k": 1,
            "mode": "min",
        },
        "periodic_checkpoint": {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "dirpath": f"{exp_dir}/checkpoints",
            "filename": f"epoch-{{epoch:04d}}",
            "every_n_epochs": val_every,
            "save_top_k": 1,
            "save_last": True,
            "monitor": "val_loss",
            "mode": "min",
        },
    }

    # Logger
    cfg["logger"] = {
        "choice": "tensorboard",
        "tensorboard": {
            "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
            "save_dir": exp_dir,
            "name": "tb_logs",
            "default_hp_metric": False,
        },
    }

    return cfg


def save_config(cfg, name):
    """Save config YAML to experiments/<name>/config.yaml."""
    exp_dir = Path(f"experiments/{name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return str(config_path)


def run_experiment(name, config_path):
    """Run a single training experiment as a subprocess."""
    print(f"\n{'='*70}")
    print(f"  STARTING: {name}")
    print(f"  Config:   {config_path}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, "train.py", "--config", config_path]

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False,
    )

    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (code {result.returncode})"
    print(f"\n{'='*70}")
    print(f"  FINISHED: {name} -> {status}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    return success


def collect_results():
    """
    Parse TensorBoard event files to extract final validation metrics.
    Falls back to checkpoint filenames for val_loss if TB parsing fails.
    """
    results = []

    for exp_name, group, desc, _ in EXPERIMENTS:
        exp_dir = Path(f"experiments/{exp_name}")
        if not exp_dir.exists():
            continue

        entry = {
            "name": exp_name,
            "group": group,
            "description": desc,
            "val_loss": None,
            "pesq": None,
            "stoi": None,
            "si_sdr": None,
            "weighted_score": None,
        }

        # Try to parse TensorBoard logs
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            tb_dirs = list(exp_dir.rglob("events.out.tfevents.*"))
            if tb_dirs:
                tb_log_dir = str(tb_dirs[0].parent)
                ea = EventAccumulator(tb_log_dir)
                ea.Reload()

                scalar_tags = ea.Tags().get("scalars", [])

                metric_map = {
                    "val_loss": "val_loss",
                    "val/pesq": "pesq",
                    "val/stoi": "stoi",
                    "val/si_sdr": "si_sdr",
                    "val/weighted_score": "weighted_score",
                }

                for tb_tag, key in metric_map.items():
                    if tb_tag in scalar_tags:
                        events = ea.Scalars(tb_tag)
                        if events:
                            # Get the last logged value
                            entry[key] = events[-1].value
        except ImportError:
            # tensorboard not available, try checkpoint name fallback
            ckpt_files = list(exp_dir.rglob("best-*.ckpt"))
            if ckpt_files:
                # Extract val_loss from filename like "best-epoch=29-val_loss=0.1234.ckpt"
                fname = ckpt_files[0].stem
                for part in fname.split("-"):
                    if "val_loss" in part:
                        try:
                            entry["val_loss"] = float(part.split("=")[-1])
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[WARNING] Failed to parse results for {exp_name}: {e}")

        results.append(entry)

    return results


def print_results_table(results):
    """Print a formatted comparison table of all experiments."""
    if not results:
        print("No results found. Run experiments first.")
        return

    # Header
    print(f"\n{'='*110}")
    print(f"  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*110}")
    header = f"{'Name':<30} {'Group':<15} {'Val Loss':>10} {'PESQ':>8} {'STOI':>8} {'SI-SDR':>8} {'Score':>8}"
    print(header)
    print("-" * 110)

    # Sort by weighted_score descending (best first), handling None
    def sort_key(r):
        score = r.get("weighted_score")
        if score is None:
            return float("-inf")
        return score

    sorted_results = sorted(results, key=sort_key, reverse=True)

    for r in sorted_results:
        val_loss = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "N/A"
        pesq = f"{r['pesq']:.4f}" if r['pesq'] is not None else "N/A"
        stoi = f"{r['stoi']:.4f}" if r['stoi'] is not None else "N/A"
        si_sdr = f"{r['si_sdr']:.2f}" if r['si_sdr'] is not None else "N/A"
        score = f"{r['weighted_score']:.4f}" if r['weighted_score'] is not None else "N/A"

        print(f"{r['name']:<30} {r['group']:<15} {val_loss:>10} {pesq:>8} {stoi:>8} {si_sdr:>8} {score:>8}")

    print(f"{'='*110}")

    # Find best per group
    print(f"\n  BEST PER GROUP:")
    print(f"-" * 80)
    groups = {}
    for r in sorted_results:
        g = r["group"]
        if g not in groups and r["weighted_score"] is not None:
            groups[g] = r

    for g, r in groups.items():
        score = f"{r['weighted_score']:.4f}" if r['weighted_score'] is not None else "N/A"
        print(f"  {g:<20} -> {r['name']:<30} (score: {score})")

    print()


def save_results_json(results):
    """Save results to a JSON file for later analysis."""
    out_path = Path("experiments/results_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def save_results_csv(results):
    """Save results to a CSV file."""
    out_path = Path("experiments/results_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("name,group,description,val_loss,pesq,stoi,si_sdr,weighted_score\n")
        for r in results:
            vals = [
                r["name"], r["group"], f'"{r["description"]}"',
                str(r["val_loss"] or ""), str(r["pesq"] or ""),
                str(r["stoi"] or ""), str(r["si_sdr"] or ""),
                str(r["weighted_score"] or ""),
            ]
            f.write(",".join(vals) + "\n")
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CleanUNet2 Hyperparameter Sweep")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs per experiment (default: 30)")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N epochs (default: 5)")
    parser.add_argument("--group", type=str, default=None,
                        help="Run only experiments from this group")
    parser.add_argument("--name", type=str, default=None,
                        help="Run only this specific experiment")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments without running")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect and display results (no training)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate config files (no training)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have checkpoints")

    args = parser.parse_args()

    # --- List mode ---
    if args.list:
        print(f"\n{'='*90}")
        print(f"  DEFINED EXPERIMENTS ({len(EXPERIMENTS)} total)")
        print(f"{'='*90}")
        print(f"{'Name':<30} {'Group':<15} {'Description'}")
        print("-" * 90)
        for name, group, desc, overrides in EXPERIMENTS:
            print(f"{name:<30} {group:<15} {desc}")
        print()

        # Count per group
        from collections import Counter
        groups = Counter(g for _, g, _, _ in EXPERIMENTS)
        print("Experiments per group:")
        for g, c in sorted(groups.items()):
            print(f"  {g:<20}: {c}")
        print(f"  {'TOTAL':<20}: {len(EXPERIMENTS)}")
        return

    # --- Collect-only mode ---
    if args.collect_only:
        results = collect_results()
        print_results_table(results)
        save_results_json(results)
        save_results_csv(results)
        return

    # --- Filter experiments ---
    experiments = EXPERIMENTS
    if args.name:
        experiments = [(n, g, d, o) for n, g, d, o in experiments if n == args.name]
        if not experiments:
            print(f"ERROR: No experiment found with name '{args.name}'")
            sys.exit(1)
    elif args.group:
        experiments = [(n, g, d, o) for n, g, d, o in experiments if g == args.group]
        if not experiments:
            print(f"ERROR: No experiments found for group '{args.group}'")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  CleanUNet2 Hyperparameter Sweep")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Validation every: {args.val_every} epochs")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # --- Generate configs and optionally run ---
    completed = 0
    failed = 0
    skipped = 0

    for i, (name, group, desc, overrides) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Preparing: {name} ({desc})")

        # Check if already done
        if args.skip_existing:
            ckpt_dir = Path(f"experiments/{name}/checkpoints")
            if list(ckpt_dir.rglob("best-*.ckpt")) if ckpt_dir.exists() else []:
                print(f"  -> Skipping (checkpoint already exists)")
                skipped += 1
                continue

        # Build and save config
        cfg = build_config(name, overrides, args.epochs, args.val_every)
        config_path = save_config(cfg, name)
        print(f"  -> Config saved: {config_path}")

        if args.generate_only:
            continue

        # Run training
        success = run_experiment(name, config_path)
        if success:
            completed += 1
        else:
            failed += 1

    # --- Summary ---
    if not args.generate_only:
        print(f"\n{'='*70}")
        print(f"  SWEEP COMPLETE")
        print(f"  Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
        print(f"  End time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # Collect and display results
        results = collect_results()
        print_results_table(results)
        save_results_json(results)
        save_results_csv(results)
    else:
        print(f"\nConfig generation complete. {len(experiments)} configs saved in experiments/")
        print(f"Run training with: python run_experiments.py --epochs {args.epochs}")


if __name__ == "__main__":
    main()
