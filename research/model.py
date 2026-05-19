#!/usr/bin/env python3
"""
model.py — PSIPredictor LSTM architecture for ML-driven lmkd.

Purpose
-------
Defines the lightweight LSTM that scores the probability of an imminent
low-memory kill within the next ~200 ms, given a rolling window of PSI +
meminfo samples. The architecture is deliberately small so the exported
ONNX graph fits the ≤ 2 ms ARM64 p99 inference budget (plan-executable.md
Phase 3 / Phase 4 verification).

Architecture (plan-executable.md Phase 3 task 1)
------------------------------------------------
    Input  : FloatTensor[batch, 20, 6]
             6 features = (some_avg10, some_avg60, some_total,
                           full_avg10, full_total, mem_available_kb)
    LSTM   : input_size=6, hidden_size=32, num_layers=1, batch_first=True
    Dropout: p=0.2 (only active during training)
    Linear : 32 -> 1
    Output : sigmoid in [0, 1]  -- "probability of kill in the next ~200ms"

Parameter count (closed form)
-----------------------------
    LSTM   : 4 * (input*hidden + hidden*hidden + 2*hidden)
           = 4 * (6*32 + 32*32 + 2*32) = 5,120
    Linear : hidden + 1 = 33
    Total  : 5,153  (well under the 200,000 budget)

Training vs. export
-------------------
This module exposes the model with a `return_logits` switch on forward(),
so callers can choose:

    - Training: forward(x, return_logits=True) -> raw logits, fed to
      `nn.BCEWithLogitsLoss(pos_weight=...)` for numerically-stable
      class-weighted loss.
    - Inference / ONNX export: forward(x) -> sigmoid probabilities in
      [0, 1], matching the C++ contract in Phase 4 (predict() returns
      float ∈ [0, 1]).

Usage
-----
    from research.model import PSIPredictor
    m = PSIPredictor()
    probs = m(torch.zeros(1, 20, 6))            # inference path
    logits = m(torch.zeros(1, 20, 6),
               return_logits=True)              # training path

Exit codes
----------
N/A — library module, no CLI.
"""

from __future__ import annotations

import torch
from torch import nn


INPUT_SIZE: int = 6
HIDDEN_SIZE: int = 32
NUM_LAYERS: int = 1
DROPOUT_P: float = 0.2
WINDOW: int = 20


class PSIPredictor(nn.Module):
    """LSTM-based kill-probability predictor. See module docstring."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout_p: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        """
        x: FloatTensor[batch, 20, 6]
        returns: FloatTensor[batch, 1]
          - sigmoid probabilities (default), or
          - raw logits if return_logits=True (use for BCEWithLogitsLoss).
        """
        # out: [batch, 20, hidden]; we use only the last timestep.
        out, _ = self.lstm(x)
        last = out[:, -1, :]               # [batch, hidden]
        last = self.dropout(last)
        logits = self.fc(last)             # [batch, 1]
        if return_logits:
            return logits
        return self.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
