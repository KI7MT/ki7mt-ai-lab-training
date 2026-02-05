# Ionis Phase 11: Production Validation & Expansion Roadmap

## 1. Overview

The successful completion of **V10 (Phase 10)** has resolved the critical "Kp Inversion" problem through the implementation of Dual Monotonic Sidecars. **Phase 11** focuses on quantifying model performance across diverse global conditions and preparing the architectural groundwork for V11.

---

## 2. Robust Model Testing (The V10 "Stress Test")

Before moving to V11, the V10 model must undergo rigorous evaluation beyond simple RMSE/Pearson metrics.

### **A. Path-Specific Holdout Tests**

* **The Polar Challenge:** Test the model on paths crossing the Auroral Ovals. *Expectation:* Does the Kp Sidecar correctly penalize high-latitude paths more severely than equatorial ones, even with a global monotonic constraint?
* **Long-Path vs. Short-Path:** Evaluate accuracy on NVIS (Near Vertical Incidence Skywave) vs. 10,000km+ multi-hop paths.
* **Blind Geographic Regions:** Test the model on a region it saw less of (e.g., South America or Africa) to check for "Spatial Generalization".

### **B. Temporal Stability Analysis**

* **The Dawn/Dusk Transition:** HF physics changes rapidly during Grey Line propagation. We need to verify if the DNN's `hour_sin/cos` and `day_night_est` features capture the sudden SNR swings correctly.
* **Solar Cycle Extrapolation:** Feed the model extreme SFI values (e.g., SFI=350) to see if the MonotonicMLP maintains a sane, non-exploding benefit curve.

---

## 3. Design Goals for Ionis V11

Once V10 is validated, V11 will introduce "Second-Order Physics" that the current sidecars cannot handle alone.

### **A. Multi-Frequency Cross-Modulation**

Current Sidecars apply a global solar benefit regardless of the band. In reality, higher SFI benefits 10m more than 160m (which may suffer from increased D-layer absorption).

* **V11 Proposal:** Implement **Frequency-Aware Sidecars** where the operating band acts as a gatekeeper for the solar boost.

### **B. Seasonal Path Interactions**

The ionosphere's height changes with the seasons. A path that works in June may fail in December due to the "Seasonal Anomaly."

* **V11 Proposal:** Introduce interaction terms between `midpoint_lat` and `season_cos` to capture the shift in ionization density.

---

## 4. Engineering & Infrastructure Updates

To support faster iteration, we will implement the following in the `-training` repo:

| Feature | Description |
| --- | --- |
| **The "Ionis-Eval" Suite** | A standardized set of 50 globally distributed paths to run against every new checkpoint. |
| **Sidecar Visualization** | Automated scripts to plot the Sun/Storm response curves (Benefit vs SFI) after every epoch. |
| **Quantization Pipeline** | Converting `.pth` weights to specialized formats (CoreML/ONNX) for the M3 Ultra Neural Engine. |

---

## 5. Next High-Value Step

To kick off Phase 11, we should build a **"Global SNR Heatmap Generator."** This script would take a single transmitter location (e.g., your QTH) and generate a world map of predicted SNR for a specific band and solar condition. This is the fastest way to "visually" debug if the DNN is making geographic mistakes that the RMSE doesn't show.

**Would you like me to draft the `generate_global_heatmap.py` script for V10 so we can see the first "Visual Audit" of your model?**