import torch
import numpy as np
import pandas as pd
from train_v6_monotonic import IonisMish  # Import your model class

# --- CONFIG ---
MODEL_PATH = "models/ionis_v6_monotonic.pth"
DEVICE = torch.device("mps")

# --- DEFINE THE TEST CASE ---
# We will create a "Standard Day" and only change Kp.
# Features: distance, freq_log, hour_sin, hour_cos, ssn, az_sin, az_cos, 
#           lat_diff, midpoint_lat, season_sin, season_cos, ssn_lat_interact, 
#           day_night_est, sfi, kp, band_sfi_interact, sfi_dist_interact, kp_penalty

def get_test_vector(kp_value):
    # Base conditions: 20m band, Noon, SFI 150 (Good Solar), 5000km path
    
    # 1. Calculate Kp Penalty (The critical feature)
    # Kp 0 -> 1.0 (No penalty)
    # Kp 9 -> 0.0 (Max penalty)
    kp_penalty = 1.0 - (kp_value / 9.0)
    
    # 2. Construct Feature List (Must match training order EXACTLY)
    features = [
        0.16,   # distance (normalized ~5000km)
        0.87,   # freq_log (20m band)
        -0.5,   # hour_sin (Noon-ish)
        -0.8,   # hour_cos
        0.5,    # ssn (Medium Cycle)
        0.0,    # az_sin
        1.0,    # az_cos
        0.1,    # lat_diff
        0.4,    # midpoint_lat (Mid-Latitude)
        0.5,    # season_sin
        0.5,    # season_cos
        0.2,    # ssn_lat_interact
        1.0,    # day_night_est (Daytime)
        0.6,    # sfi (High Solar Activity)
        kp_value / 9.0, # kp (Normalized 0-1)
        0.5,    # band_sfi_interact
        0.5,    # sfi_dist_interact
        kp_penalty # THE TARGET FEATURE (Index 17)
    ]
    return torch.tensor([features], dtype=torch.float32).to(DEVICE)

# --- RUN TEST ---
def test_physics():
    print(f"Loading {MODEL_PATH}...")
    model = IonisMish(input_dim=18).to(DEVICE)
    
    # Load the full checkpoint
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    
    # Unwrap: If it's a dictionary with metadata, pull out the weights
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        print("  > Detected Enhanced Checkpoint (with Metadata). Unpacking...")
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("-" * 40)
    print("PHYSICS VERIFICATION: THE STORM TEST")
    print("-" * 40)

    # Test 1: Quiet Sun (Kp = 0)
    v_quiet = get_test_vector(kp_value=0.0)
    with torch.no_grad():
        snr_quiet = model(v_quiet).item()
    
    # Test 2: Severe Storm (Kp = 9)
    v_storm = get_test_vector(kp_value=9.0)
    with torch.no_grad():
        snr_storm = model(v_storm).item()

    print(f"Scenario: SFI 150, 20m Band, Mid-Day")
    print(f"  > SNR at Kp 0 (Quiet):  {snr_quiet:.2f} dB")
    print(f"  > SNR at Kp 9 (Storm):  {snr_storm:.2f} dB")
    
    diff = snr_quiet - snr_storm
    print("-" * 40)
    if diff > 0:
        print(f"PASS: Signal DROPPED by {diff:.2f} dB during storm.")
        print("The Monotonic Constraint worked.")
    else:
        print(f"FAIL: Signal INCREASED by {abs(diff):.2f} dB.")
        print("The Inversion persists.")

if __name__ == "__main__":
    test_physics()