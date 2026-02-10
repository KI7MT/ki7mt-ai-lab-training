#!/usr/bin/env python3
"""
ionis_v11_gate.py — IONIS V11 "Gatekeeper" Architecture

Multiplicative Interaction Gates for geographic and frequency modulation
of solar/storm effects. Design proposal for review — no training code.

Problem (from POL-01 and FREQ-01):
    V10 sidecars produce global constants — storm cost is flat +1.12 dB
    everywhere (poles = equator), SFI benefit is flat on all bands.

Solution:
    V11: output = base_snr + sun_scaler * SunSidecar(sfi)
                           + storm_scaler * StormSidecar(kp)

    The DNN splits into a shared trunk → three heads:
      - base_head      → base_snr ∈ (-∞, +∞)
      - sun_scaler_head → sun_scaler ∈ [0.5, 2.0]  (volume control)
      - storm_scaler_head → storm_scaler ∈ [0.5, 2.0]

    Gate function: gate(x) = 0.5 + 1.5 * sigmoid(x)
    Bounded, differentiable, never zero. At init gate ≈ 1.0.

Architecture:
    Trunk:            11 → 512 → 256 (shared representation)
    Base Head:        256 → 128 → 1  (geography/time baseline)
    Sun Scaler Head:  256 → 64  → 1  (geographic SFI modulation)
    Storm Scaler Head:256 → 64  → 1  (geographic Kp modulation)
    Sun Sidecar:      1 → 8 → 1     (monotonic, unchanged from V10)
    Storm Sidecar:    1 → 8 → 1     (monotonic, unchanged from V10)

Parameter budget: 203,573 (V10: 170,547 — +19.4% increase)

Key constraints:
    - Sidecars remain locked: MonotonicMLP, weights clamped [0.5, 2.0]
    - DNN still receives ZERO solar information (Starvation Protocol)
    - Gates can only scale sidecar output [0.5x, 2.0x] — never reverse
    - Gate init = 1.0 (V10-equivalent at start of training)
"""

import math

import torch
import torch.nn as nn


# Feature layout (unchanged from V10)
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

# Gate initialization: sigmoid(x) = 1/3 → x = -ln(2)
GATE_INIT_BIAS = -math.log(2.0)  # ≈ -0.693


class MonotonicMLP(nn.Module):
    """Small MLP with monotonically increasing output.

    Unchanged from V10. Weights constrained non-negative via abs() in
    forward pass and clamped [0.5, 2.0] after optimizer step.
    """

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        out = nn.functional.linear(h, w2, self.fc2.bias)
        return out


def _gate(x):
    """Bounded gate: maps (-∞, +∞) → [0.5, 2.0].

    gate(x) = 0.5 + 1.5 * sigmoid(x)

    At x = -ln(2) ≈ -0.693:  sigmoid(-ln2) = 1/3  →  gate = 0.5 + 0.5 = 1.0
    This gives identity behavior (V10-equivalent) at initialization.
    """
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV11Gate(nn.Module):
    """IONIS V11 "Gatekeeper" — Multiplicative Interaction Gates.

    output = base_snr + gate(sun_scaler) * SunSidecar(sfi)
                      + gate(storm_scaler) * StormSidecar(kp_penalty)

    The shared trunk learns a joint representation from 11 geography/time
    features. Three heads decode this into a base SNR prediction and two
    scalar gate values that modulate the sidecar outputs per-sample.

    This allows the model to learn that:
      - Polar paths experience stronger storm effects (storm_scaler > 1)
      - Low-band paths see less SFI benefit (sun_scaler < 1)
      - Equatorial paths are less affected by storms (storm_scaler < 1)
    """

    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()

        # Shared trunk: geography/time → joint representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
        )

        # Base head: trunk output → base SNR (same role as V10 DNN)
        self.base_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

        # Sun scaler head: trunk output → raw logit for sun gate
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

        # Storm scaler head: trunk output → raw logit for storm gate
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

        # Sidecars: unchanged from V10
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        # Initialize scaler heads to output ≈ -0.693 (gate = 1.0)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        """Zero-init final layer weights, set bias to -ln(2).

        This makes gate(output) ≈ 1.0 for all inputs at initialization,
        so V11 starts as a V10-equivalent model.
        """
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]  # Last nn.Linear in the Sequential
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, GATE_INIT_BIAS)

    def forward(self, x):
        # Split features (unchanged from V10)
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]

        # Shared trunk
        trunk_out = self.trunk(x_deep)

        # Three heads
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)

        # Gates: [0.5, 2.0]
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        # Sidecars (physics — unchanged)
        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        # Multiplicative interaction
        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost

    def get_sun_effect(self, sfi_normalized, device=None):
        """Get raw sun sidecar output for a given normalized SFI."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty, device=None):
        """Get raw storm sidecar output for a given kp_penalty."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Return (sun_gate, storm_gate) tensors for diagnostic use."""
        x_deep = x[:, :DNN_DIM]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(sun_logit), _gate(storm_logit)


def scaler_variance_loss(model, x):
    """Anti-collapse regularization: penalizes constant gate values.

    L_var = -λ * (Var(sun_gate) + Var(storm_gate))

    If the optimizer collapses scalers to a constant (i.e., the gate
    produces the same value for all inputs), this loss increases.
    Applied during Phase B (Gate Awakening) of training.

    Args:
        model: IonisV11Gate instance
        x: input batch tensor (N, 13)

    Returns:
        Negative variance (to be added to loss with positive λ)
    """
    x_deep = x[:, :DNN_DIM]
    trunk_out = model.trunk(x_deep)
    sun_logit = model.sun_scaler_head(trunk_out)
    storm_logit = model.storm_scaler_head(trunk_out)
    sun_gate = _gate(sun_logit)
    storm_gate = _gate(storm_logit)
    return -(sun_gate.var() + storm_gate.var())


# --- Self-verification ---
if __name__ == '__main__':
    print("IONIS V11 Gatekeeper — Architecture Verification")
    print("=" * 55)

    model = IonisV11Gate(dnn_dim=DNN_DIM, sidecar_hidden=8)

    # Parameter count
    trunk_params = sum(p.numel() for p in model.trunk.parameters())
    base_params = sum(p.numel() for p in model.base_head.parameters())
    sun_sc_params = sum(p.numel() for p in model.sun_scaler_head.parameters())
    storm_sc_params = sum(p.numel() for p in model.storm_scaler_head.parameters())
    sun_side_params = sum(p.numel() for p in model.sun_sidecar.parameters())
    storm_side_params = sum(p.numel() for p in model.storm_sidecar.parameters())
    total = sum(p.numel() for p in model.parameters())

    print(f"\nParameter Budget:")
    print(f"  Trunk (11→512→256):       {trunk_params:>8,}")
    print(f"  Base head (256→128→1):    {base_params:>8,}")
    print(f"  Sun scaler (256→64→1):    {sun_sc_params:>8,}")
    print(f"  Storm scaler (256→64→1):  {storm_sc_params:>8,}")
    print(f"  Sun sidecar (1→8→1):      {sun_side_params:>8,}")
    print(f"  Storm sidecar (1→8→1):    {storm_side_params:>8,}")
    print(f"  {'─' * 38}")
    print(f"  Total:                    {total:>8,}")

    expected = 203_573
    status = "PASS" if total == expected else "FAIL"
    print(f"\n  Expected: {expected:,} → {status}")

    # Gate initialization check
    print(f"\nGate Initialization (should be ≈ 1.0):")
    x_test = torch.randn(100, INPUT_DIM)
    sun_gate, storm_gate = model.get_gates(x_test)
    print(f"  Sun gate:   mean={sun_gate.mean():.4f}, std={sun_gate.std():.6f}")
    print(f"  Storm gate: mean={storm_gate.mean():.4f}, std={storm_gate.std():.6f}")
    gate_ok = abs(sun_gate.mean().item() - 1.0) < 0.01
    print(f"  Gate ≈ 1.0: {'PASS' if gate_ok else 'FAIL'}")

    # Forward pass shape
    print(f"\nForward Pass:")
    out = model(x_test)
    print(f"  Input:  {x_test.shape}")
    print(f"  Output: {out.shape}")
    shape_ok = out.shape == (100, 1)
    print(f"  Shape correct: {'PASS' if shape_ok else 'FAIL'}")

    # Summary
    print(f"\n{'=' * 55}")
    all_pass = (total == expected) and gate_ok and shape_ok
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
