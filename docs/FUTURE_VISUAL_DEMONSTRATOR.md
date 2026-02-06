# IONIS Visual Demonstrator (Future)

**Source:** Gemini Pro suggestion, 2026-02-05
**Status:** Parked for later (Step J or K on D-to-Z roadmap)
**Priority:** Nice-to-have after core validation complete

---

## Concept

Interactive web map where you click TX and RX points, adjust SFI/Kp/hour sliders, and watch V12 predict SNR in real-time. Line color indicates band condition (green=excellent, cyan=good, orange=marginal, grey=closed).

## Stack

- **Backend:** FastAPI + Uvicorn (Python, runs on M3)
- **Frontend:** Leaflet.js map + vanilla HTML/JS
- **Dependencies:** `pip install fastapi uvicorn`

## Why This Matters

1. **Demos:** Show stakeholders the model isn't a black box
2. **Debugging:** Visually spot geographic anomalies
3. **Education:** Ham radio operators can explore propagation
4. **Validation:** Click known contest paths, verify predictions

## Implementation Notes

Gemini provided complete server.py and index.html — see below.

---

## server.py

```python
# File: server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import math
import uvicorn

# Import your V12 model structure
from model_v12_baseline import IonisV12

# --- Configuration ---
MODEL_PATH = "models/ionis_v12_signatures.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI(title="IONIS V12 Oracle API")

# Enable CORS so the HTML file can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Once on Startup
print(f"Loading V12 Model from {MODEL_PATH}...")
try:
    model = IonisV12().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("V12 Digital Twin Online.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

class PredictionRequest(BaseModel):
    lat_tx: float
    lon_tx: float
    lat_rx: float
    lon_rx: float
    freq: float
    sfi: float
    kp: float
    hour: float
    month: int = 3

@app.post("/predict")
async def get_prediction(req: PredictionRequest):
    # 1. Geometry (Haversine)
    R = 6371.0
    phi1, phi2 = math.radians(req.lat_tx), math.radians(req.lat_rx)
    dlambda = math.radians(req.lon_rx - req.lon_tx)
    a = math.sin((math.radians(req.lat_rx-req.lat_tx))/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist_km = R * c
    dist_norm = dist_km / 20000.0

    # Bearing
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    theta = math.atan2(y, x)

    # 2. Physics Tensors
    freq_log = math.log10(req.freq) / math.log10(30.0)
    hour_rad = (req.hour / 24.0) * 2 * math.pi
    month_rad = (req.month / 12.0) * 2 * math.pi
    dn_est = -math.cos(hour_rad)

    features = torch.tensor([[
        dist_norm,
        freq_log,
        math.sin(hour_rad), math.cos(hour_rad),
        math.sin(theta), math.cos(theta),
        abs(req.lat_tx-req.lat_rx)/180.0,
        ((req.lat_tx+req.lat_rx)/2)/90.0,
        math.sin(month_rad), math.cos(month_rad),
        dn_est,
        req.sfi,
        req.kp
    ]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        snr = model(features).item()

    # Context
    condition = "CLOSED"
    color = "#555555" # Grey
    if snr > -28:
        condition = "MARGINAL (WSPR)"
        color = "#ffcc00" # Orange
    if snr > -20:
        condition = "GOOD (CW)"
        color = "#00ccff" # Cyan
    if snr > -10:
        condition = "EXCELLENT (Voice)"
        color = "#00ff00" # Green

    return {
        "snr": round(snr, 2),
        "distance": round(dist_km, 0),
        "condition": condition,
        "color": color
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## index.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>IONIS V12 Digital Twin</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        body { margin: 0; padding: 0; background: #1a1a1a; color: #fff; font-family: sans-serif; }
        #map { height: 100vh; width: 100vw; }
        #controls {
            position: absolute; top: 20px; right: 20px; z-index: 1000;
            background: rgba(0,0,0,0.8); padding: 20px; border-radius: 8px;
            width: 300px;
        }
        input, button { width: 100%; margin-bottom: 10px; padding: 5px; background: #333; color: #fff; border: 1px solid #555; }
        h1 { margin-top: 0; font-size: 18px; color: #00ffcc; }
        .stat { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; }
        .val { font-weight: bold; color: #fff; }
    </style>
</head>
<body>

<div id="controls">
    <h1>IONIS V12 Control</h1>
    <div class="stat"><span>State:</span> <span class="val" id="state-txt">Select TX</span></div>
    <hr>
    <label>Frequency (MHz)</label>
    <input type="number" id="freq" value="14.05">
    <label>Solar Flux (SFI)</label>
    <input type="number" id="sfi" value="150">
    <label>K-Index</label>
    <input type="number" id="kp" value="2">
    <label>Hour (UTC)</label>
    <input type="number" id="hour" value="14">
    <button onclick="resetMap()">Reset Pins</button>
    <hr>
    <div class="stat"><span>SNR:</span> <span class="val" id="res-snr">--</span></div>
    <div class="stat"><span>Dist:</span> <span class="val" id="res-dist">--</span></div>
    <div class="stat"><span>Cond:</span> <span class="val" id="res-cond">--</span></div>
</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
<script>
    // Initialize Map
    var map = L.map('map').setView([20, 0], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    var txMarker = null;
    var rxMarker = null;
    var polyline = null;

    map.on('click', function(e) {
        if (!txMarker) {
            txMarker = L.marker(e.latlng).addTo(map).bindPopup("TX").openPopup();
            document.getElementById('state-txt').innerText = "Select RX";
        } else if (!rxMarker) {
            rxMarker = L.marker(e.latlng).addTo(map).bindPopup("RX").openPopup();
            document.getElementById('state-txt').innerText = "Running Inference...";
            runPrediction();
        }
    });

    async function runPrediction() {
        if (!txMarker || !rxMarker) return;

        const payload = {
            lat_tx: txMarker.getLatLng().lat,
            lon_tx: txMarker.getLatLng().lng,
            lat_rx: rxMarker.getLatLng().lat,
            lon_rx: rxMarker.getLatLng().lng,
            freq: parseFloat(document.getElementById('freq').value),
            sfi: parseFloat(document.getElementById('sfi').value),
            kp: parseFloat(document.getElementById('kp').value),
            hour: parseFloat(document.getElementById('hour').value)
        };

        try {
            const res = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            // Draw Line
            if (polyline) map.removeLayer(polyline);
            polyline = L.polyline([txMarker.getLatLng(), rxMarker.getLatLng()], {
                color: data.color,
                weight: 5,
                opacity: 0.7
            }).addTo(map);

            // Update UI
            document.getElementById('res-snr').innerText = data.snr + " dB";
            document.getElementById('res-dist').innerText = data.distance + " km";
            document.getElementById('res-cond').innerText = data.condition;
            document.getElementById('res-cond').style.color = data.color;
            document.getElementById('state-txt').innerText = "Done";

        } catch (err) {
            console.error(err);
            alert("API Error: Ensure server.py is running!");
        }
    }

    function resetMap() {
        if (txMarker) map.removeLayer(txMarker);
        if (rxMarker) map.removeLayer(rxMarker);
        if (polyline) map.removeLayer(polyline);
        txMarker = null;
        rxMarker = null;
        polyline = null;
        document.getElementById('state-txt').innerText = "Select TX";
        document.getElementById('res-snr').innerText = "--";
    }
</script>
</body>
</html>
```

---

## Integration Notes

1. **Model import needs adjustment** — Gemini referenced `model_v12_baseline.py` but our architecture is `IonisV12Gate` in `oracle_v12.py`
2. **Feature vector needs alignment** — Gemini's feature order differs slightly from our training
3. **Should use oracle_v12.py directly** — wrap `IonisOracle.predict()` instead of reimplementing

## When to Implement

After Step I (Ground Truth Validation with contest logs). Visual demo is more compelling when we can show "IONIS predicted this, contest log confirmed it."
