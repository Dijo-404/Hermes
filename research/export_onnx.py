#!/usr/bin/env python3
"""
export_onnx.py — convert a trained PSIPredictor .pt checkpoint to ONNX.

Purpose
-------
Loads `psi_predictor.pt` produced by `train.py`, exports the model to
ONNX with `opset_version=11` and a dynamic batch axis, then validates
that ONNX Runtime produces outputs within 1e-5 of PyTorch on 100 random
input windows (plan-executable.md Phase 3 verification).

Contract (used by the C++ side in Phase 4)
------------------------------------------
    Input name : "psi_window"
    Input shape: [batch, 20, 6]   (batch axis is dynamic)
    Output name: "kill_prob"
    Output shape: [batch, 1]
    Opset      : 11 (Android ONNX Runtime prebuilts support this safely)

Usage
-----
    python export_onnx.py --ckpt out/psi_predictor.pt \\
                          --out  out/psi_predictor.onnx [--n-validate 100]

Exit codes
----------
  0 — exported, validation passed, model bytes printed.
  1 — checkpoint missing / unreadable.
  2 — ONNX export raised.
  3 — max-abs-diff exceeded 1e-5 (PyTorch vs onnxruntime mismatch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import WINDOW_LEN
from model import PSIPredictor, count_parameters


INPUT_NAME: str = "psi_window"
OUTPUT_NAME: str = "kill_prob"
OPSET: int = 11
MAX_DIFF: float = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PSIPredictor to ONNX.")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-validate", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(ckpt_path: Path) -> tuple[PSIPredictor, dict]:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = obj.get("arch", {})
    model = PSIPredictor(
        input_size=arch.get("input_size", 6),
        hidden_size=arch.get("hidden_size", 32),
        num_layers=arch.get("num_layers", 1),
    )
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, arch


def export(model: PSIPredictor, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, WINDOW_LEN, model.input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out.as_posix(),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )


def validate(
    model: PSIPredictor, onnx_path: Path, n: int, seed: int
) -> float:
    # Lazy import so the script can at least *try* to export without ORT.
    import onnxruntime as ort  # type: ignore

    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(n):
            x = rng.standard_normal((1, WINDOW_LEN, model.input_size)).astype(
                np.float32
            )
            torch_out = model(torch.from_numpy(x)).detach().numpy()
            ort_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
            diff = float(np.max(np.abs(torch_out - ort_out)))
            if diff > max_diff:
                max_diff = diff
    return max_diff


def main() -> int:
    args = parse_args()
    if not args.ckpt.exists():
        print(f"ERR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1

    model, arch = load_model(args.ckpt)
    pc = count_parameters(model)
    print(f"Loaded model: {arch}; param_count={pc}")

    try:
        export(model, args.out)
    except Exception as e:  # noqa: BLE001  — print and exit deterministically
        print(f"ERR: ONNX export failed: {e}", file=sys.stderr)
        return 2

    size_bytes = args.out.stat().st_size
    print(f"Wrote {args.out} ({size_bytes} bytes)")

    max_diff = validate(model, args.out, args.n_validate, args.seed)
    print(
        f"Validation max-abs-diff over {args.n_validate} random windows: "
        f"{max_diff:.3e} (limit {MAX_DIFF:.0e})"
    )
    if max_diff > MAX_DIFF:
        print(
            f"ERR: PyTorch vs ONNX Runtime diverge by {max_diff} > {MAX_DIFF}",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
