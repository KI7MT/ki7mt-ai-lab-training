#!/usr/bin/env python3
"""
verify_v13_combined.py — IONIS V13 Combined Physics Verification

Tests:
  1. Storm Sidecar: Kp 0 should have higher SNR than Kp 9
  2. Sun Sidecar: SFI 200 should have higher SNR than SFI 70
  3. Gate range: all gate values within [0.5, 2.0]
  4. Decomposition math: base + sun_gate*sun_raw + storm_gate*storm_raw = predicted

V13 uses per-source per-band Z-score normalization. Model outputs are in σ units.
Approximate dB conversion: ×6.7 (average σ across sources/bands).
"""

import math
import os

import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v13_combined.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

GATE_INIT_BIAS = -math.log(2.0)

# Approximate dB per σ for interpreting Z-score outputs
DB_PER_SIGMA = 6.7


# ── Model (must match training) ─────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        return nn.functional.linear(h, w2, self.fc2.bias)


def _gate(x):
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    """V13 uses same architecture as V12 (IonisV12Gate)."""
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, GATE_INIT_BIAS)

    def forward(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)
        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def get_sun_effect(self, sfi_normalized):
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()


# ── Test Vector Builder ──────────────────────────────────────────────────────

def get_test_vector(sfi_normalized, kp_penalty):
    """Build a 13-feature test vector for physics checks.

    Features 0-10: trunk inputs (geography & time — fixed mid-lat daytime)
    Feature 11: sfi (Sun Sidecar)
    Feature 12: kp_penalty (Storm Sidecar)
    """
    features = [
        0.16,    # distance (~5000km)
        0.87,    # freq_log (20m band)
        -0.5,    # hour_sin (Noon-ish)
        -0.8,    # hour_cos
        0.0,     # az_sin
        1.0,     # az_cos
        0.1,     # lat_diff
        0.4,     # midpoint_lat (Mid-Latitude)
        0.5,     # season_sin
        0.5,     # season_cos
        1.0,     # day_night_est (Daytime)
        sfi_normalized,
        kp_penalty,
    ]
    return torch.tensor([features], dtype=torch.float32, device=DEVICE)


def decompose(model, x):
    """Full forward decomposition for a single input."""
    x_deep = x[:, :DNN_DIM]
    x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
    x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
    with torch.no_grad():
        trunk_out = model.trunk(x_deep)
        base_snr = model.base_head(trunk_out).item()
        sun_logit = model.sun_scaler_head(trunk_out)
        storm_logit = model.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit).item()
        storm_gate = _gate(storm_logit).item()
        sun_raw = model.sun_sidecar(x_sfi).item()
        storm_raw = model.storm_sidecar(x_kp).item()
    sun_contrib = sun_gate * sun_raw
    storm_contrib = storm_gate * storm_raw
    predicted = base_snr + sun_contrib + storm_contrib
    return {
        'base_snr': base_snr,
        'sun_gate': sun_gate,
        'storm_gate': storm_gate,
        'sun_raw': sun_raw,
        'storm_raw': storm_raw,
        'sun_contrib': sun_contrib,
        'storm_contrib': storm_contrib,
        'predicted': predicted,
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_physics():
    print(f"Loading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)

    dnn_dim = checkpoint.get('dnn_dim', DNN_DIM)
    sidecar_hidden = checkpoint.get('sidecar_hidden', 8)
    model = IonisV12Gate(dnn_dim=dnn_dim, sidecar_hidden=sidecar_hidden).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {checkpoint.get('architecture', 'unknown')}")
    print(f"  Parameters: {total_params:,}")
    print(f"  RMSE: {checkpoint.get('val_rmse', 0):.4f} σ ({checkpoint.get('val_rmse', 0) * DB_PER_SIGMA:.2f} dB equiv)")
    print(f"  Pearson: {checkpoint.get('val_pearson', 0):+.4f}")
    print()
    print(f"  NOTE: V13 outputs are in Z-score (σ) units.")
    print(f"        Approximate dB conversion: ×{DB_PER_SIGMA:.1f}")

    all_pass = True

    # ── TEST 1: Storm Sidecar (Kp) ──
    print("\n" + "=" * 60)
    print("TEST 1: STORM SIDECAR (Kp)")
    print("=" * 60)

    sfi_const = 150.0 / 300.0

    v_quiet = get_test_vector(sfi_normalized=sfi_const, kp_penalty=1.0)
    v_storm = get_test_vector(sfi_normalized=sfi_const, kp_penalty=0.0)

    d_quiet = decompose(model, v_quiet)
    d_storm = decompose(model, v_storm)

    storm_effect_quiet = model.get_storm_effect(1.0)
    storm_effect_storm = model.get_storm_effect(0.0)

    print(f"\nScenario: SFI 150 (fixed), 20m Band, Mid-Day")
    print(f"  SNR at Kp 0 (Quiet):  {d_quiet['predicted']:+.3f} σ ({d_quiet['predicted'] * DB_PER_SIGMA:+.1f} dB)")
    print(f"  SNR at Kp 9 (Storm):  {d_storm['predicted']:+.3f} σ ({d_storm['predicted'] * DB_PER_SIGMA:+.1f} dB)")
    print(f"  Storm Sidecar at Kp 0: {storm_effect_quiet:+.3f} σ")
    print(f"  Storm Sidecar at Kp 9: {storm_effect_storm:+.3f} σ")
    print(f"  Storm gate at Kp 0: {d_quiet['storm_gate']:.4f}")
    print(f"  Storm gate at Kp 9: {d_storm['storm_gate']:.4f}")

    storm_cost = d_quiet['predicted'] - d_storm['predicted']
    print(f"\n  Storm Cost (Kp 0→9): {storm_cost:+.3f} σ ({storm_cost * DB_PER_SIGMA:+.1f} dB)")

    if storm_cost > 0:
        print("  PASS: Signal DROPPED during storm (correct physics)")
    else:
        print("  FAIL: Signal INCREASED during storm (inverted)")
        all_pass = False

    # ── TEST 2: Sun Sidecar (SFI) ──
    print("\n" + "=" * 60)
    print("TEST 2: SUN SIDECAR (SFI)")
    print("=" * 60)

    kp_const = 1.0 - 2.0 / 9.0

    v_low_sfi = get_test_vector(sfi_normalized=70.0 / 300.0, kp_penalty=kp_const)
    v_high_sfi = get_test_vector(sfi_normalized=200.0 / 300.0, kp_penalty=kp_const)

    d_low = decompose(model, v_low_sfi)
    d_high = decompose(model, v_high_sfi)

    sun_effect_low = model.get_sun_effect(70.0 / 300.0)
    sun_effect_high = model.get_sun_effect(200.0 / 300.0)

    print(f"\nScenario: Kp 2 (fixed), 20m Band, Mid-Day")
    print(f"  SNR at SFI 70:   {d_low['predicted']:+.3f} σ ({d_low['predicted'] * DB_PER_SIGMA:+.1f} dB)")
    print(f"  SNR at SFI 200:  {d_high['predicted']:+.3f} σ ({d_high['predicted'] * DB_PER_SIGMA:+.1f} dB)")
    print(f"  Sun Sidecar at SFI 70:  {sun_effect_low:+.3f} σ")
    print(f"  Sun Sidecar at SFI 200: {sun_effect_high:+.3f} σ")
    print(f"  Sun gate at SFI 70:  {d_low['sun_gate']:.4f}")
    print(f"  Sun gate at SFI 200: {d_high['sun_gate']:.4f}")

    sun_benefit = d_high['predicted'] - d_low['predicted']
    print(f"\n  Sun Benefit (SFI 70→200): {sun_benefit:+.3f} σ ({sun_benefit * DB_PER_SIGMA:+.1f} dB)")

    if sun_benefit > 0:
        print("  PASS: Signal IMPROVED with higher SFI (correct physics)")
    else:
        print("  FAIL: Signal DECREASED with higher SFI (inverted)")
        all_pass = False

    # ── TEST 3: Gate Range ──
    print("\n" + "=" * 60)
    print("TEST 3: GATE RANGE [0.5, 2.0]")
    print("=" * 60)

    # Test with random geography inputs
    x_random = torch.randn(1000, INPUT_DIM, device=DEVICE)
    x_random[:, SFI_IDX] = torch.rand(1000, device=DEVICE)      # SFI in [0, 1]
    x_random[:, KP_PENALTY_IDX] = torch.rand(1000, device=DEVICE)  # kp_p in [0, 1]

    x_deep = x_random[:, :DNN_DIM]
    with torch.no_grad():
        trunk_out = model.trunk(x_deep)
        sun_logit = model.sun_scaler_head(trunk_out)
        storm_logit = model.storm_scaler_head(trunk_out)
        sun_gates = _gate(sun_logit)
        storm_gates = _gate(storm_logit)

    sun_min = sun_gates.min().item()
    sun_max = sun_gates.max().item()
    storm_min = storm_gates.min().item()
    storm_max = storm_gates.max().item()

    print(f"\n  1000 random inputs:")
    print(f"  Sun gate:   min={sun_min:.4f}, max={sun_max:.4f}, "
          f"mean={sun_gates.mean().item():.4f}, std={sun_gates.std().item():.4f}")
    print(f"  Storm gate: min={storm_min:.4f}, max={storm_max:.4f}, "
          f"mean={storm_gates.mean().item():.4f}, std={storm_gates.std().item():.4f}")

    sun_range_ok = (sun_min >= 0.5 - 1e-6) and (sun_max <= 2.0 + 1e-6)
    storm_range_ok = (storm_min >= 0.5 - 1e-6) and (storm_max <= 2.0 + 1e-6)

    if sun_range_ok and storm_range_ok:
        print("  PASS: All gate values within [0.5, 2.0]")
    else:
        print("  FAIL: Gate values outside bounds")
        all_pass = False

    # Gate variance (are gates actually differentiating?)
    sun_var = sun_gates.var().item()
    storm_var = storm_gates.var().item()
    print(f"\n  Gate variance (higher = more geographic differentiation):")
    print(f"  Sun gate var:   {sun_var:.6f}")
    print(f"  Storm gate var: {storm_var:.6f}")
    if sun_var < 1e-6 and storm_var < 1e-6:
        print("  NOTE: Gates are flat — model behaves like V10 (no geographic modulation)")
    else:
        print("  Gates are differentiating — model is adding geographic modulation")

    # ── TEST 4: Decomposition Math ──
    print("\n" + "=" * 60)
    print("TEST 4: DECOMPOSITION MATH")
    print("=" * 60)

    test_cases = [
        ("Quiet mid-lat",  150.0 / 300.0, 1.0 - 2.0 / 9.0),
        ("Storm mid-lat",  150.0 / 300.0, 0.0),
        ("Low SFI quiet",  70.0 / 300.0,  1.0),
        ("High SFI storm", 250.0 / 300.0, 0.0),
    ]

    decomp_pass = True
    print(f"\n  {'Case':<18s}  {'base':>7s}  {'sun_c':>7s}  {'stm_c':>7s}  "
          f"{'sum':>7s}  {'pred':>7s}  {'err':>8s}")
    print(f"  {'-'*72}")

    for name, sfi_n, kp_p in test_cases:
        v = get_test_vector(sfi_normalized=sfi_n, kp_penalty=kp_p)
        d = decompose(model, v)
        check_sum = d['base_snr'] + d['sun_contrib'] + d['storm_contrib']
        err = abs(d['predicted'] - check_sum)
        status = "OK" if err < 1e-3 else "MISMATCH"
        if err >= 1e-3:
            decomp_pass = False

        # Also verify against model.forward()
        with torch.no_grad():
            fwd_snr = model(v).item()
        fwd_err = abs(d['predicted'] - fwd_snr)

        print(f"  {name:<18s}  {d['base_snr']:+6.3f}  {d['sun_contrib']:+6.3f}  "
              f"{d['storm_contrib']:+6.3f}  {check_sum:+6.3f}  {d['predicted']:+6.3f}  "
              f"{err:.2e} {status}")

        if fwd_err >= 1e-3:
            print(f"    WARNING: forward() mismatch: {fwd_snr:+.4f} vs decomp {d['predicted']:+.4f}")
            decomp_pass = False

    if decomp_pass:
        print("\n  PASS: base + sun_gate*sun_raw + storm_gate*storm_raw = predicted")
    else:
        print("\n  FAIL: Decomposition does not match")
        all_pass = False

    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = [
        ("Storm Sidecar (Kp 0→9)",  storm_cost > 0),
        ("Sun Sidecar (SFI 70→200)", sun_benefit > 0),
        ("Gate range [0.5, 2.0]",    sun_range_ok and storm_range_ok),
        ("Decomposition math",       decomp_pass),
    ]

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30s}  {status}")

    print()
    if all_pass:
        print("ALL TESTS PASSED — Physics constraints enforced")
        print()
        print("  V13 Combined Model Physics:")
        print(f"    Kp 0→9 storm cost:   {storm_cost:+.3f} σ ({storm_cost * DB_PER_SIGMA:+.1f} dB)")
        print(f"    SFI 70→200 benefit:  {sun_benefit:+.3f} σ ({sun_benefit * DB_PER_SIGMA:+.1f} dB)")
        print(f"    Storm/Sun ratio:     {storm_cost / sun_benefit:.1f}:1" if sun_benefit > 0 else "")
        print(f"    Sun gate range:      [{sun_min:.4f}, {sun_max:.4f}]")
        print(f"    Storm gate range:    [{storm_min:.4f}, {storm_max:.4f}]")
        print()
        print("  V13 trained on WSPR + RBN DXpedition signatures (152 rare DXCC entities)")
    else:
        print("SOME TESTS FAILED — Review model training")


if __name__ == "__main__":
    test_physics()
