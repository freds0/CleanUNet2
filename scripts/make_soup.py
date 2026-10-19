#!/usr/bin/env python
"""Uniform weight average ("model soup") of Lightning checkpoints of the same architecture (IMPROVEMENT_PLAN.md,
section 14). The runs must share their starting point (e.g. seeds fine-tuned from stage A), so their weights lie in
one basin. The output is a checkpoint that evaluate.py, a distillation teacher or an init can load.

Usage:
    python scripts/make_soup.py --ckpts logs/abl/K0extfull_s{1,2,3}/last.ckpt --out logs/abl/K0extfull_soup/last.ckpt
"""
import argparse
import os

import torch


def soup(paths):
    """Average the state dicts of checkpoints with the same architecture (integer buffers: the first one's). Returns {state_dict, hyper_parameters, soup_of}."""
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    arch = ckpts[0]["hyper_parameters"]["model"]["architecture"]
    for p, c in zip(paths[1:], ckpts[1:]):
        if c["hyper_parameters"]["model"]["architecture"] != arch:
            raise ValueError(f"{p} has another architecture than {paths[0]}")
    keys = [k for k in ckpts[0]["state_dict"] if k.startswith("model.")]
    out = {}
    for k in keys:
        tensors = [c["state_dict"].get(k) for c in ckpts]
        if any(t is None or t.shape != tensors[0].shape for t in tensors):
            raise ValueError(f"parameter {k} is missing or has another shape in one of the checkpoints")
        if tensors[0].is_floating_point():
            out[k] = torch.stack([t.float() for t in tensors]).mean(0).to(tensors[0].dtype)
        else:                                                    # integer buffers: the first checkpoint's
            out[k] = tensors[0]
    return {"state_dict": out, "hyper_parameters": ckpts[0]["hyper_parameters"], "soup_of": list(paths)}


def main():
    """Average --ckpts into --out."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.ckpts) < 2:
        ap.error("a soup needs at least two checkpoints")
    result = soup(args.ckpts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(result, args.out)
    print(f"[INFO] soup of {len(args.ckpts)} checkpoints ({len(result['state_dict'])} tensors) -> {args.out}")


if __name__ == "__main__":
    main()
