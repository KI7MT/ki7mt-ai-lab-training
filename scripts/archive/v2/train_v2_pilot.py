import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import clickhouse_connect
import numpy as np
import re

# --- 1. Configuration & Device ---
CH_HOST = '192.168.1.90'
BATCH_SIZE = 4096
EPOCHS = 50
LEARNING_RATE = 0.001
SAMPLE_SIZE = 10_000_000
VAL_SPLIT = 0.2

INPUT_DIM = 13
HIDDEN_DIM = 256
DATE_START = '2020-01-01'
DATE_END = '2026-02-04'
SOLAR_RESOLUTION = 'daily'

# Phase 3.5: Stratified Band Sampling — equal rows per HF band
#   snr BETWEEN -35 AND 25:     remove receiver glitches and local overloads
#   band IN (1..10):            HF bands only (160m-10m), no VHF/UHF
#   distance BETWEEN 500 AND 18000: focus on ionospheric skip only
#   Stratified sampling via UNION ALL of 10 per-band queries (memory-efficient)
PHASE = 'Phase 3.5: Stratified Band Sampling'
HF_BANDS = list(range(1, 11))  # bands 1-10 (160m through 10m)
ROWS_PER_BAND = SAMPLE_SIZE // len(HF_BANDS)  # 1M rows per band for 10M total

# Feature set — 13 features (v2.1):
#   Core 11 from v2.0 plus 2 interaction features.
#   ssn_lat_interact: SSN effect varies by latitude (higher lat = more impact)
#   day_night_est:    crude local solar time proxy (is midpoint illuminated?)
FEATURES = [
    'distance', 'freq_log', 'hour_sin', 'hour_cos',
    'ssn',
    'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos',
    'ssn_lat_interact', 'day_night_est',
]

# WSPR band ID → dial frequency in Hz (standard codes only)
# Band IDs 1-15 match the integer values stored in wspr.spots_raw.band
# Leaked values (band > 15) are excluded by SQL filter.
BAND_TO_HZ = {
     1:  1_836_600,     # 160m
     2:  3_568_600,     # 80m
     3:  5_287_200,     # 60m
     4:  7_038_600,     # 40m
     5: 10_138_700,     # 30m
     6: 14_097_100,     # 20m
     7: 18_104_600,     # 17m
     8: 21_094_600,     # 15m
     9: 24_924_600,     # 12m
    10: 28_124_600,     # 10m
    11: 50_294_000,     # 6m
    12: 70_091_000,     # 4m
    13: 144_489_000,    # 2m
    14: 432_300_000,    # 70cm
    15: 1_296_500_000,  # 23cm
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"IONIS V2 ({PHASE}): Using {device}")


# --- 2. Model Architecture ---
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


class IONIS_V2(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256):
        super().__init__()
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.post = nn.Linear(hidden_dim, 1)  # Raw SNR output in dB

    def forward(self, x):
        x = torch.relu(self.pre(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.post(x)


# --- 3. Maidenhead Grid Utilities ---
# Matches algorithm in clickhouse_loader.cpp:27-77

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')

def clean_grids(raw_grids):
    """Clean FixedString(8) grid values from ClickHouse.

    Handles bytes (null-padded), strings, and other types.
    Returns list of 4-char uppercase grid strings.
    Invalid grids default to 'JJ00' (equatorial Atlantic).
    """
    out = []
    for g in raw_grids:
        if isinstance(g, (bytes, bytearray)):
            s = g.decode('ascii', errors='ignore').rstrip('\x00')
        else:
            s = str(g).rstrip('\x00')
        m = GRID_RE.search(s)
        out.append(m.group(0).upper() if m else 'JJ00')
    return out


def grid_to_latlon(grids):
    """Convert list of 4-char Maidenhead grids to (lat, lon) arrays.

    Fully vectorized via numpy char-code arithmetic.
    """
    grid4 = np.array(grids, dtype='U4')
    codes = grid4.view('U1').reshape(-1, 4)
    b = codes.view(np.uint32).astype(np.float32)

    field_lon  = b[:, 0] - ord('A')   # 0-17
    field_lat  = b[:, 1] - ord('A')
    square_lon = b[:, 2] - ord('0')   # 0-9
    square_lat = b[:, 3] - ord('0')

    lon = field_lon * 20.0 - 180.0 + square_lon * 2.0 + 1.0
    lat = field_lat * 10.0 - 90.0 + square_lat * 1.0 + 0.5

    return lat, lon


def band_to_hz(band_ids):
    """Map WSPR band ID array to frequency in Hz using lookup table.

    Only standard band IDs 1-15 are expected (SQL filters out others).
    Unknown values default to 14097100 (20m) as safety fallback.
    """
    default_hz = 14_097_100
    return np.array([BAND_TO_HZ.get(int(b), default_hz) for b in band_ids],
                    dtype=np.float64)


# --- 4. ClickHouse Dataset ---
# Predicts actual observed SNR from path geometry + solar + geography.
# Features (13):
#   [0]  distance         - Great-circle km, normalized /20000
#   [1]  freq_log         - log10(Hz) normalized /8
#   [2]  hour_sin         - sin(2*pi*hour/24) cyclical time encoding
#   [3]  hour_cos         - cos(2*pi*hour/24) cyclical time encoding
#   [4]  ssn              - Sunspot number, normalized /300
#   [5]  az_sin           - sin(2*pi*azimuth/360) bearing encoding
#   [6]  az_cos           - cos(2*pi*azimuth/360) bearing encoding
#   [7]  lat_diff         - abs(tx_lat - rx_lat) / 180
#   [8]  midpoint_lat     - (tx_lat + rx_lat) / 2 / 90
#   [9]  season_sin       - sin(2*pi*month/12) seasonal encoding
#   [10] season_cos       - cos(2*pi*month/12) seasonal encoding
#   [11] ssn_lat_interact - ssn_norm * abs(midpoint_lat_norm), physics interaction
#   [12] day_night_est    - cos(2*pi*local_solar_hour/24), daylight proxy
# Target:
#   SNR in dB (raw, not normalized)
#
# Phase 3.5 SQL filters (Stratified Band Sampling):
#   snr BETWEEN -35 AND 25      — valid WSPR decode range
#   band IN (1..10)              — HF bands only, stratified sampling
#   distance BETWEEN 500 AND 18000 — ionospheric skip only

class WSPRDataset(Dataset):
    def __init__(self, host, limit=10_000_000):
        print(f"Connecting to {host}...")
        client = clickhouse_connect.get_client(host=host, port=8123)

        # Phase 3.5: Stratified band sampling via UNION ALL.
        # 10 independent per-band subqueries, each with its own LIMIT.
        # Memory-efficient: ClickHouse handles 10 small ORDER BY operations
        # across many cores instead of one giant window function.
        rows_per_band = ROWS_PER_BAND
        band_list = ','.join(str(b) for b in HF_BANDS)

        # Build UNION ALL of 10 per-band queries
        band_subqueries = []
        for band_id in HF_BANDS:
            sq = f"""SELECT * FROM (
                SELECT s.snr, s.distance, s.band,
                       toHour(s.timestamp) AS hour, toMonth(s.timestamp) AS month,
                       s.azimuth,
                       toString(s.grid) AS grid, toString(s.reporter_grid) AS reporter_grid,
                       sol.ssn
                FROM wspr.spots_raw s
                INNER JOIN (
                    SELECT date, max(ssn) AS ssn
                    FROM solar.indices_raw FINAL
                    GROUP BY date
                    HAVING ssn > 0
                ) sol ON toDate(s.timestamp) = sol.date
                WHERE s.band = {band_id}
                  AND s.timestamp >= '{DATE_START}' AND s.timestamp < '{DATE_END}'
                  AND s.snr BETWEEN -35 AND 25
                  AND s.distance BETWEEN 500 AND 18000
                  AND length(s.grid) >= 4 AND length(s.reporter_grid) >= 4
                ORDER BY cityHash64(toString(s.timestamp))
                LIMIT {rows_per_band}
            )"""
            band_subqueries.append(sq)

        query = "\nUNION ALL\n".join(band_subqueries)

        print(f"Running query ({DATE_START} to {DATE_END}, {SOLAR_RESOLUTION} solar, "
              f"stratified UNION ALL, HF bands 1-10)...")
        print(f"  Filters: snr[-35,25] band[{band_list}] distance[500,18000]")
        print(f"  Stratified: {rows_per_band:,} rows per band, {len(HF_BANDS)} bands")
        try:
            result = client.query(query)
            rows = result.result_rows
        except Exception as e:
            print(f"  UNION ALL stratified query failed: {e}")
            print(f"  Falling back to non-stratified query with HF bands 1-10...")
            fallback_query = f"""
                WITH solar_daily AS (
                    SELECT date, max(ssn) AS ssn
                    FROM solar.indices_raw FINAL
                    GROUP BY date
                    HAVING ssn > 0
                )
                SELECT s.snr, s.distance, s.band,
                       toHour(s.timestamp), toMonth(s.timestamp), s.azimuth,
                       toString(s.grid), toString(s.reporter_grid),
                       sol.ssn
                FROM wspr.spots_raw s
                INNER JOIN solar_daily sol
                    ON toDate(s.timestamp) = sol.date
                WHERE s.timestamp >= '{DATE_START}' AND s.timestamp < '{DATE_END}'
                  AND s.snr BETWEEN -35 AND 25
                  AND s.band IN ({band_list})
                  AND s.distance BETWEEN 500 AND 18000
                  AND length(s.grid) >= 4 AND length(s.reporter_grid) >= 4
                LIMIT {limit}
            """
            result = client.query(fallback_query)
            rows = result.result_rows

        n = len(rows)
        print(f"Query returned: {n:,} rows")

        if n == 0:
            raise RuntimeError("Query returned 0 rows — check filters and solar data coverage")

        # Per-band row counts (stratification check)
        band_col = [r[2] for r in rows]
        band_arr_check = np.array(band_col, dtype=np.int32)
        print(f"\n  Per-band row counts (stratified):")
        for b in sorted(HF_BANDS):
            cnt = int((band_arr_check == b).sum())
            hz = BAND_TO_HZ.get(b, 0)
            label = f"{hz/1e6:.3f}MHz" if hz else f"band={b}"
            print(f"    Band {b:2d} ({label:>10s}): {cnt:>10,} rows")

        # Unpack columns via zip transpose (one pass over rows)
        print("\nUnpacking columns...")
        cols = list(zip(*rows))
        snr       = np.array(cols[0], dtype=np.float32)
        distance  = np.array(cols[1], dtype=np.float32)
        band_id   = np.array(cols[2], dtype=np.int32)
        hour      = np.array(cols[3], dtype=np.float32)
        month     = np.array(cols[4], dtype=np.float32)
        azimuth   = np.array(cols[5], dtype=np.float32)
        raw_tx    = list(cols[6])
        raw_rx    = list(cols[7])
        ssn_arr   = np.array(cols[8], dtype=np.float32)

        # Map band IDs to actual frequency in Hz (standard codes only)
        print("Mapping band IDs to frequencies...")
        frequency = band_to_hz(band_id)
        unique_bands = np.unique(band_id)
        print(f"  Bands in dataset: {sorted(unique_bands)} "
              f"({', '.join(f'{b}={BAND_TO_HZ.get(int(b), 0)/1e6:.3f}MHz' for b in sorted(unique_bands))})")

        # Clean grids: handle FixedString null padding and validate format
        print("Cleaning grids...")
        tx_grids = clean_grids(raw_tx)
        rx_grids = clean_grids(raw_rx)

        print("Computing grid coordinates...")
        tx_lat, tx_lon = grid_to_latlon(tx_grids)
        rx_lat, rx_lon = grid_to_latlon(rx_grids)

        # Vectorized feature engineering
        print("Engineering features...")
        f_distance     = distance / 20000.0
        f_freq_log     = np.log10(frequency.astype(np.float32)) / 8.0
        f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
        f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
        f_ssn          = ssn_arr / 300.0
        f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
        f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
        f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
        f_midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
        f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
        f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)

        # Feature interaction: SSN * |midpoint_lat|
        f_ssn_lat_interact = f_ssn * np.abs(f_midpoint_lat)

        # Day/night estimator: crude local solar time at path midpoint
        midpoint_lon = (tx_lon + rx_lon) / 2.0
        local_solar_hour = hour + midpoint_lon / 15.0
        f_day_night_est = np.cos(2.0 * np.pi * local_solar_hour / 24.0)

        features = np.column_stack([
            f_distance, f_freq_log, f_hour_sin, f_hour_cos,
            f_ssn,
            f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
            f_season_sin, f_season_cos,
            f_ssn_lat_interact, f_day_night_est,
        ])

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(snr, dtype=torch.float32).unsqueeze(1)

        self.snr_mean = self.y.mean().item()
        self.snr_std = self.y.std().item()

        print(f"\nDataset loaded: {len(self.X):,} rows, {INPUT_DIM} features")
        print(f"SNR range: {self.y.min().item():.0f} to {self.y.max().item():.0f} dB")
        print(f"SNR mean: {self.snr_mean:.1f} dB, std: {self.snr_std:.1f} dB")

        # Feature diagnostics
        print(f"\nFeature statistics (normalized):")
        print(f"  {'Feature':<20s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}")
        print(f"  {'-'*52}")
        for i, name in enumerate(FEATURES):
            col = features[:, i]
            print(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  {col.max():8.4f}")

        # SSN-SNR correlation diagnostic (Phase 3.5 target: > 0.1)
        ssn_snr_corr = np.corrcoef(ssn_arr, snr)[0, 1]
        print(f"\n  SSN-SNR Pearson correlation: {ssn_snr_corr:+.4f}")
        if ssn_snr_corr > 0.1:
            print(f"  TARGET MET: correlation > 0.1 after data purge")
        elif ssn_snr_corr > 0:
            print(f"  Positive but below 0.1 target. Filters helped but more cleanup may be needed.")
        else:
            print(f"  WARNING: Still negative after filtering!")

        # SSN-SNR correlation by band
        print(f"\n  SSN-SNR correlation by band:")
        band_counts = {}
        for b in unique_bands:
            mask = band_id == b
            band_counts[int(b)] = mask.sum()
        top_bands = sorted(band_counts, key=band_counts.get, reverse=True)[:5]
        for b in top_bands:
            mask = band_id == b
            if mask.sum() > 100:
                corr = np.corrcoef(ssn_arr[mask], snr[mask])[0, 1]
                hz = BAND_TO_HZ.get(b, 0)
                label = f"{hz/1e6:.1f}MHz" if hz else f"band={b}"
                print(f"    {label:>10s} ({mask.sum():>8,} rows): r = {corr:+.4f}")

        # Temporal diversity check
        print(f"\n  Temporal diversity:")
        month_dist = dict(zip(*np.unique(month.astype(int), return_counts=True)))
        print(f"    Month distribution: {month_dist}")
        print(f"    SSN range in data: {ssn_arr.min():.0f} to {ssn_arr.max():.0f}")

        # Raw value samples
        print(f"\nRaw value samples (first 5 rows):")
        print(f"  band_id:   {band_id[:5]}")
        print(f"  freq (Hz): {frequency[:5]}")
        print(f"  ssn:       {ssn_arr[:5]}")
        print(f"  snr:       {snr[:5]}")
        print(f"  distance:  {distance[:5]}")
        print(f"  tx_grid:   {tx_grids[:5]}")
        print(f"  tx_lat:    {tx_lat[:5]}")
        print(f"  rx_lat:    {rx_lat[:5]}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 5. Training Loop ---
def main():
    dataset = WSPRDataset(CH_HOST, limit=SAMPLE_SIZE)

    # Train/validation split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nTrain: {train_size:,}, Validation: {val_size:,}")

    model = IONIS_V2(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {INPUT_DIM} -> {HIDDEN_DIM} -> 1, {params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print("Starting Training...\n")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'snr_mean': dataset.snr_mean,
                'snr_std': dataset.snr_std,
                'input_dim': INPUT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'features': FEATURES,
                'date_range': f'{DATE_START} to {DATE_END}',
                'sample_size': len(dataset),
                'solar_resolution': SOLAR_RESOLUTION,
                'phase': PHASE,
            }, "models/ionis_v2.pth")
            marker = " *"

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"Train: {train_loss:.4f}  "
            f"Val: {val_loss:.4f}  "
            f"(RMSE: {np.sqrt(val_loss):.2f} dB){marker}"
        )

    print(f"\nTraining complete. Best validation RMSE: {np.sqrt(best_val_loss):.2f} dB")
    print("Model saved as models/ionis_v2.pth")


if __name__ == "__main__":
    main()
