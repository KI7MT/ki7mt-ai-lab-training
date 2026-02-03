import streamlit as st
import torch
import torch.nn as nn
import clickhouse_connect
import pandas as pd
import plotly.express as px
import numpy as np

# --- BRAIN ARCHITECTURE ---
class PropagationNet(nn.Module):
    def __init__(self):
        super(PropagationNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

# --- CONFIG & LOAD ---
TR_HOST = '10.25.3.1'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

@st.cache_resource
def load_model():
    model = PropagationNet().to(device)
    model.load_state_dict(torch.load("prophet_v1.pth"))
    model.eval()
    return model

st.set_page_config(page_title="Sovereign AI Propagation Dashboard", layout="wide")
st.title("🛰️ Sovereign AI: Propagation Command")

# --- ENHANCED DATA FETCHING ---
def get_historical_trends():
    try:
        client = clickhouse_connect.get_client(host=TR_HOST, username='default', database='wspr')
        # Pull last 96 entries (24 hours at 15-min intervals)
        query = "SELECT timestamp, kp_index, solar_flux FROM wspr.live_conditions ORDER BY timestamp DESC LIMIT 96"
        df_hist = client.query_df(query)
        return df_hist
    except:
        return pd.DataFrame()

# --- NEW UI ELEMENT: TREND LINE ---
st.divider()
st.subheader("24-Hour Solar Trend")
df_trends = get_historical_trends()

if not df_trends.empty:
    # Plotting Kp and SFI over time
    fig_trend = px.line(df_trends, x='timestamp', y=['kp_index'], 
                        title="Geomagnetic Activity (Last 24h)",
                        labels={"value": "Index", "variable": "Metric"})
    fig_trend.update_layout(template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Gathering historical data... Check back in an hour.")

kp, sfi = get_live_data()
model = load_model()

# --- DASHBOARD UI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Space Weather")
    st.metric("Planetary K-Index", f"{kp:.1f}", delta="-0.5" if kp < 4 else "STORM", delta_color="inverse")
    st.metric("Solar Flux (SFI)", f"{sfi:.0f}")
    
    # User can override for "What If" scenarios
    st.divider()
    sim_kp = st.slider("Simulate Kp Index", 0.0, 9.0, float(kp))
    sim_sfi = st.slider("Simulate SFI", 50, 300, int(sfi))

# --- PREDICTION ENGINE ---
geo_health = 1.0 - (sim_kp / 9.0)
sfi_health = min(1.0, sim_sfi / 250.0)

# Generate Map Data
distances = np.linspace(0, 1, 100)
inputs = torch.tensor([[d, sfi_health, geo_health] for d in distances], dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(inputs).cpu().numpy().flatten()

df = pd.DataFrame({"Distance (km)": distances * 20000, "Quality": preds})

with col2:
    st.subheader("Predicted Signal Reach")
    fig = px.area(df, x="Distance (km)", y="Quality", color_discrete_sequence=['#00ff00'])
    fig.update_layout(template="plotly_dark", yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    if geo_health < 0.4:
        st.error("🚨 HIGH GEOMAGNETIC DISTURBANCE: DX paths are likely closed.")
    else:
        st.success("✅ OPEN SKY: Conditions are favorable for long-haul communication.")

