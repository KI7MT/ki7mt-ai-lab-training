#!/usr/bin/env python3
"""
verify_v10_final.py — Phase 10 Final Physics Verification

Tests the dual monotonic sidecars:
1. Storm Sidecar: Kp 0 should have higher SNR than Kp 9
2. Sun Sidecar: SFI 200 should have higher SNR than SFI 70

Reports costs in actual dB (model outputs raw dB, no scaling needed).
"""

import torch
import numpy as np
from train_v10_final import IonisDualMono, MonotonicMLP, DNN_DIM, SFI_IDX, KP_PENALTY_IDX

# --- CONFIG ---
MODEL_PATH = "models/ionis_v10_final.pth"
DEVICE = torch.device("mps")


def get_test_vector(sfi_normalized, kp_penalty):
    """Build a 13-feature test vector (V10 Nuclear Option).

    Features 0-10: DNN inputs (geography & time ONLY)
    Feature 11: sfi (Sun Sidecar)
    Feature 12: kp_penalty (Storm Sidecar)
    """
    # Base conditions: 20m band, Noon, mid-latitude path
    # V10: NO interaction terms - DNN has zero solar info
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
        # Sidecar inputs
        sfi_normalized,  # Sun Sidecar (index 11)
        kp_penalty,      # Storm Sidecar (index 12)
    ]
    return torch.tensor([features], dtype=torch.float32).to(DEVICE)


def test_physics():
    print(f"Loading {MODEL_PATH}...")
    model = IonisDualMono(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        print("  > Detected Enhanced Checkpoint (with Metadata). Unpacking...")
        print(f"  > Architecture: {checkpoint.get('architecture', 'unknown')}")
        print(f"  > Constraint: {checkpoint.get('monotonic_constraint', 'unknown')}")
        state_dict = checkpoint["model_state"]
        snr_std = checkpoint.get('snr_std', 8.7)
    else:
        state_dict = checkpoint
        snr_std = 8.7

    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 50)
    print("PHYSICS VERIFICATION: DUAL MONOTONIC SIDECARS")
    print("Phase 9: Sun Sidecar + Storm Sidecar")
    print("=" * 50)

    # --- TEST 1: Storm Sidecar (Kp) ---
    print("\n--- STORM TEST (Kp Sidecar) ---")

    # Hold SFI constant at moderate value
    sfi_const = 150.0 / 300.0  # normalized

    # Kp 0 (quiet): kp_penalty = 1.0
    v_quiet = get_test_vector(sfi_normalized=sfi_const, kp_penalty=1.0)
    with torch.no_grad():
        snr_quiet = model(v_quiet).item()
        storm_effect_quiet = model.get_storm_effect(1.0)

    # Kp 9 (storm): kp_penalty = 0.0
    v_storm = get_test_vector(sfi_normalized=sfi_const, kp_penalty=0.0)
    with torch.no_grad():
        snr_storm = model(v_storm).item()
        storm_effect_storm = model.get_storm_effect(0.0)

    print(f"Scenario: SFI 150 (fixed), 20m Band, Mid-Day")
    print(f"  > SNR at Kp 0 (Quiet):  {snr_quiet:.2f} dB")
    print(f"  > SNR at Kp 9 (Storm):  {snr_storm:.2f} dB")
    print(f"  > Storm Sidecar output at Kp 0: {storm_effect_quiet:.2f} dB")
    print(f"  > Storm Sidecar output at Kp 9: {storm_effect_storm:.2f} dB")

    storm_cost = snr_quiet - snr_storm
    print(f"\n  Storm Cost (Kp 0->9): {storm_cost:.2f} dB")

    if storm_cost > 0:
        print("  PASS: Signal DROPPED during storm (correct physics)")
    else:
        print("  FAIL: Signal INCREASED during storm (inverted)")

    # --- TEST 2: Sun Sidecar (SFI) ---
    print("\n--- SUN TEST (SFI Sidecar) ---")

    # Hold Kp constant at quiet value
    kp_const = 1.0 - 2.0/9.0  # Kp 2, kp_penalty ~0.78

    # SFI 70 (low solar)
    v_low_sfi = get_test_vector(sfi_normalized=70.0/300.0, kp_penalty=kp_const)
    with torch.no_grad():
        snr_low_sfi = model(v_low_sfi).item()
        sun_effect_low = model.get_sun_effect(70.0/300.0)

    # SFI 200 (high solar)
    v_high_sfi = get_test_vector(sfi_normalized=200.0/300.0, kp_penalty=kp_const)
    with torch.no_grad():
        snr_high_sfi = model(v_high_sfi).item()
        sun_effect_high = model.get_sun_effect(200.0/300.0)

    print(f"Scenario: Kp 2 (fixed), 20m Band, Mid-Day")
    print(f"  > SNR at SFI 70:   {snr_low_sfi:.2f} dB")
    print(f"  > SNR at SFI 200:  {snr_high_sfi:.2f} dB")
    print(f"  > Sun Sidecar output at SFI 70:  {sun_effect_low:.2f} dB")
    print(f"  > Sun Sidecar output at SFI 200: {sun_effect_high:.2f} dB")

    sun_benefit = snr_high_sfi - snr_low_sfi
    print(f"\n  Sun Benefit (SFI 70->200): {sun_benefit:.2f} dB")

    if sun_benefit > 0:
        print("  PASS: Signal IMPROVED with higher SFI (correct physics)")
    else:
        print("  FAIL: Signal DECREASED with higher SFI (inverted)")

    # --- SUMMARY ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if storm_cost > 0 and sun_benefit > 0:
        print("BOTH TESTS PASSED — Physics constraints enforced")
        print(f"  Kp 9 storm cost: {storm_cost:.2f} dB")
        print(f"  SFI 70->200 benefit: {sun_benefit:.2f} dB")
    elif storm_cost > 0:
        print("PARTIAL: Storm test passed, Sun test failed")
    elif sun_benefit > 0:
        print("PARTIAL: Sun test passed, Storm test failed")
    else:
        print("BOTH TESTS FAILED — Check model training")


if __name__ == "__main__":
    test_physics()
