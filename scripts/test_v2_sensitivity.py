import torch
import torch.nn as nn
import numpy as np


# --- 1. Model Architecture (must match training) ---
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class ProphetV2(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256):
        super().__init__()
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.post = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.pre(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.post(x)


# --- 2. Load Model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load("prophet_v2.pth", weights_only=False, map_location=device)
input_dim = checkpoint['input_dim']
hidden_dim = checkpoint.get('hidden_dim', 256)
model = ProphetV2(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Model loaded: {input_dim} features, {hidden_dim}-unit hidden")
print(f"Features: {checkpoint['features']}")
print(f"Trained on: {checkpoint.get('date_range', 'unknown')}, "
      f"{checkpoint.get('sample_size', 'unknown'):,} rows, "
      f"{checkpoint.get('solar_resolution', 'unknown')} solar")
print(f"Training SNR stats - mean: {checkpoint['snr_mean']:.1f} dB, "
      f"std: {checkpoint['snr_std']:.1f} dB")


# --- 3. Maidenhead Grid → Lat/Lon ---
def grid_to_latlon(grid):
    """Convert a single 4-char Maidenhead grid to (lat, lon)."""
    field_lon = ord(grid[0].upper()) - ord('A')
    field_lat = ord(grid[1].upper()) - ord('A')
    square_lon = int(grid[2])
    square_lat = int(grid[3])
    lon = field_lon * 20.0 - 180.0 + square_lon * 2.0 + 1.0
    lat = field_lat * 10.0 - 90.0 + square_lat * 1.0 + 0.5
    return lat, lon


# --- 4. Reference Path: FN31 (US East) → JO21 (W. Europe) ---
TX_GRID = 'FN31'
RX_GRID = 'JO21'
TX_LAT, TX_LON = grid_to_latlon(TX_GRID)
RX_LAT, RX_LON = grid_to_latlon(RX_GRID)
REF_DISTANCE = 5900.0   # km, approximate great-circle FN31→JO21
REF_AZIMUTH = 50.0      # degrees, approximate bearing
REF_FREQ_HZ = 14_097_100  # 20m WSPR
REF_HOUR = 12
REF_MONTH = 6           # June (summer)


# --- 5. Helper (13-feature model) ---
# Feature order: distance, freq_log, hour_sin, hour_cos,
#   ssn, az_sin, az_cos, lat_diff, midpoint_lat,
#   season_sin, season_cos, ssn_lat_interact, day_night_est
def make_input(distance_km, freq_hz, hour, month, azimuth,
               tx_lat, tx_lon, rx_lat, rx_lon, ssn):
    """Build normalized 13-feature vector from human-readable values."""
    distance = distance_km / 20000.0
    freq_log = np.log10(freq_hz) / 8.0
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    ssn_norm = ssn / 300.0
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)

    # Interaction features (must match training)
    ssn_lat_interact = ssn_norm * abs(midpoint_lat)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_hour = hour + midpoint_lon / 15.0
    day_night_est = np.cos(2.0 * np.pi * local_solar_hour / 24.0)

    return torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          ssn_norm,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos,
          ssn_lat_interact, day_night_est]],
        dtype=torch.float32,
        device=device,
    )


def predict(distance_km=REF_DISTANCE, freq_hz=REF_FREQ_HZ, hour=REF_HOUR,
            month=REF_MONTH, azimuth=REF_AZIMUTH,
            tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
            ssn=100.0):
    """Run a single prediction and return SNR in dB."""
    inputs = make_input(distance_km, freq_hz, hour, month, azimuth,
                        tx_lat, tx_lon, rx_lat, rx_lon, ssn)
    with torch.no_grad():
        return model(inputs).item()


# --- 6. Solar Condition Scenarios ---
print(f"\n{'='*62}")
print(f"  Prophet V2.1 - SNR Sensitivity Analysis")
print(f"  Reference path: {TX_GRID} -> {RX_GRID} ({REF_DISTANCE:.0f} km, 20m WSPR)")
print(f"  Solar feature: SSN (sunspot number)")
print(f"  New features: ssn_lat_interact, day_night_est")
print(f"{'='*62}")

scenarios = {
    "Solar minimum (SSN 10) ": dict(ssn=10),
    "Quiet Sun     (SSN 50) ": dict(ssn=50),
    "Moderate      (SSN 100)": dict(ssn=100),
    "Active        (SSN 150)": dict(ssn=150),
    "Solar maximum (SSN 200)": dict(ssn=200),
}

print(f"\n  Solar Condition Scenarios (12:00 UTC, June):")
print(f"  {'-'*56}")
results = {}
for name, params in scenarios.items():
    snr = predict(**params)
    results[name] = snr
    print(f"  {name:44s} -> {snr:+6.1f} dB")

keys = list(scenarios.keys())
delta = results[keys[-1]] - results[keys[0]]
print(f"\n  SSN 10->200 delta: {delta:+.1f} dB")
if delta > 0:
    print(f"  CORRECT: Higher SSN improves SNR (HF propagation physics)")
else:
    print(f"  WARNING: Inverted! Higher SSN should improve HF SNR")


# --- 7. SSN Sweep ---
print(f"\n{'='*62}")
print(f"  SSN Sweep (full range)")
print(f"  {'-'*56}")
ssn_values = np.arange(0, 310, 20)
ssn_snrs = []
for ssn in ssn_values:
    snr = predict(ssn=float(ssn))
    ssn_snrs.append(snr)
    print(f"  SSN {ssn:3d}  ->  {snr:+6.1f} dB")
print(f"\n  SSN 0->300 delta: {ssn_snrs[-1] - ssn_snrs[0]:+.1f} dB")


# --- 8. Distance x Latitude Matrix ---
print(f"\n{'='*62}")
print(f"  Distance x Latitude Matrix (SSN=100, 12 UTC, June)")
print(f"  {'-'*56}")

# Paths with both lat and lon for day/night feature
paths = {
    "Equatorial (5N)":  (5.0,  0.0,  5.0,  30.0),   # tx_lat, tx_lon, rx_lat, rx_lon
    "Mid-lat (40N)":    (40.0, -73.0, 50.0, 1.0),    # like FN31->JO21
    "High-lat (65N)":   (65.0, -20.0, 65.0, 20.0),   # auroral zone
}
distances = [1000, 3000, 5000, 8000, 12000]

# Header
header = f"  {'Path':<20s}"
for d in distances:
    header += f"  {d:>6d}km"
print(header)
print(f"  {'-'*56}")

for path_name, (tlat, tlon, rlat, rlon) in paths.items():
    row = f"  {path_name:<20s}"
    for d in distances:
        snr = predict(distance_km=d, tx_lat=tlat, tx_lon=tlon,
                      rx_lat=rlat, rx_lon=rlon, ssn=100)
        row += f"  {snr:+6.1f}"
    print(row)


# --- 9. SSN Impact by Latitude ---
print(f"\n{'='*62}")
print(f"  SSN Impact by Latitude (5000 km, SSN 10 vs SSN 200)")
print(f"  {'-'*56}")

for path_name, (tlat, tlon, rlat, rlon) in paths.items():
    snr_low = predict(distance_km=5000, tx_lat=tlat, tx_lon=tlon,
                      rx_lat=rlat, rx_lon=rlon, ssn=10)
    snr_high = predict(distance_km=5000, tx_lat=tlat, tx_lon=tlon,
                       rx_lat=rlat, rx_lon=rlon, ssn=200)
    delta = snr_high - snr_low
    print(f"  {path_name:<20s}  SSN 10: {snr_low:+6.1f}  "
          f"SSN 200: {snr_high:+6.1f}  Delta: {delta:+5.1f} dB")


# --- 10. Seasonal Sweep ---
print(f"\n{'='*62}")
print(f"  Seasonal Sweep ({TX_GRID}->{RX_GRID}, SSN=100)")
print(f"  {'-'*56}")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
seasonal_snrs = []
for m in range(1, 13):
    snr = predict(month=m, ssn=100)
    seasonal_snrs.append(snr)
    print(f"  {month_names[m-1]:>3s}  ->  {snr:+6.1f} dB")

best_month = month_names[np.argmax(seasonal_snrs)]
worst_month = month_names[np.argmin(seasonal_snrs)]
print(f"\n  Best: {best_month} ({max(seasonal_snrs):+.1f} dB), "
      f"Worst: {worst_month} ({min(seasonal_snrs):+.1f} dB), "
      f"Range: {max(seasonal_snrs) - min(seasonal_snrs):.1f} dB")


# --- 11. Band Comparison ---
print(f"\n{'='*62}")
print(f"  Band Comparison ({TX_GRID}->{RX_GRID}, SSN=100, 12 UTC, June)")
print(f"  {'-'*56}")
bands = [
    ('160m',  1_836_600),
    ( '80m',  3_568_600),
    ( '40m',  7_038_600),
    ( '30m', 10_138_700),
    ( '20m', 14_097_100),
    ( '17m', 18_104_600),
    ( '15m', 21_094_600),
    ( '10m', 28_124_600),
]
for label, hz in bands:
    snr = predict(freq_hz=hz, ssn=100)
    print(f"  {label:>4s} ({hz/1e6:7.3f} MHz)  ->  {snr:+6.1f} dB")


# --- 12. Day/Night Comparison ---
print(f"\n{'='*62}")
print(f"  Day vs Night ({TX_GRID}->{RX_GRID}, SSN=100, June)")
print(f"  {'-'*56}")
for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
    snr = predict(hour=hour, ssn=100)
    print(f"  {hour:02d}:00 UTC  ->  {snr:+6.1f} dB")


print(f"\n{'='*62}")
