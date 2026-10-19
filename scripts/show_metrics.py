#!/usr/bin/env python
"""Print the scalar metrics logged to TensorBoard by a training run.

Usage:
    python scripts/show_metrics.py logs/stageA_cleanunet_mamba [--tags val/pesq val/stoi val/si_sdr]
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    """Print the last and best value of the selected TensorBoard scalars."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", help="TensorBoard run directory (searched recursively for event files)")
    parser.add_argument("--tags", nargs="*", default=["val/pesq", "val/stoi", "val/si_sdr", "val_loss",
                                                      "train/total_loss_epoch"])
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not files:
        raise SystemExit(f"No TensorBoard event files under {args.log_dir}")
    acc = EventAccumulator(os.path.dirname(files[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    available = acc.Tags()["scalars"]
    tags = [t for t in args.tags if t in available]
    rows = {}
    for t in tags:
        for e in acc.Scalars(t):
            rows.setdefault(e.step, {})[t] = e.value
    print("step".ljust(8) + "".join(t.ljust(24) for t in tags))
    for step in sorted(rows):
        print(str(step).ljust(8) + "".join(
            (f"{rows[step][t]:.4f}" if t in rows[step] else "-").ljust(24) for t in tags))


if __name__ == "__main__":
    main()
