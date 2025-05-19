import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go

# App Setup
st.set_page_config(page_title="PVD Summary Explorer", layout="wide")
st.title("📁 PVD Prozess-JSON Analyse")

# MongoDB-Verbindung
client = MongoClient("mongodb://localhost:27017/")
db = client["PVD_P_Process_Json_Summary"]
collection = db["Standard_Collection"]

# --- Seitenleiste: Ratio-Filter ---
st.sidebar.header("🔍 Filter nach Ratios")
cs_min, cs_max = st.sidebar.slider("Cs/(Sn+Pb)", 0.0, 2.0, (0.8, 1.2), step=0.05)
sn_min, sn_max = st.sidebar.slider("Sn/(Sn+Pb)", 0.0, 1.0, (0.0, 1.0), step=0.05)
br_min, br_max = st.sidebar.slider("Br/(Br+I)", 0.0, 1.0, (0.0, 1.0), step=0.05)

# MongoDB-Filter auf alle drei Ratios
mongo_filter = {
    "values.Target Composition.Cs/(Sn+Pb)": {"$gte": cs_min, "$lte": cs_max},
    "values.Target Composition.Sn/(Sn+Pb)": {"$gte": sn_min, "$lte": sn_max},
    "values.Target Composition.Br/(Br+I)": {"$gte": br_min, "$lte": br_max}
}

# Gefilterte Dokumente abrufen
filtered_docs = list(collection.find(mongo_filter))

# --- Verteilungsanalyse für alle gefilterten Dokumente ---
st.subheader("📊 Verteilungen für Ratios")
cols = st.columns(3)
ratio_ranges = {
    "Cs/(Sn+Pb)": (0, 2),
    "Sn/(Sn+Pb)": (0, 1),
    "Br/(Br+I)": (0, 1),
}
for idx, ratio in enumerate(["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]):
    values = [
        doc.get("values", {}).get("Target Composition", {}).get(ratio)
        for doc in filtered_docs
        if doc.get("values", {}).get("Target Composition", {}).get(ratio) is not None
    ]
    if values:
        min_x, max_x = ratio_ranges[ratio]
        fig = px.histogram(values, nbins=20, title=f"Histogramm: {ratio}")
        fig.update_layout(
            xaxis=dict(title=ratio, range=[min_x, max_x], showline=True, linewidth=2, linecolor='black'),
            yaxis=dict(title='Anzahl', showline=True, linewidth=2, linecolor='black'),
            bargap=0.1
        )
        cols[idx].plotly_chart(fig, use_container_width=True)

# Verfügbare Sample-IDs sammeln
# Verfügbare Sample-IDs sammeln
sample_options = []
sample_id_to_doc = {}
for doc in filtered_docs:
    sid = doc.get("metadata", {}).get("Sample ID")
    if sid:
        sample_options.append(sid)
        sample_id_to_doc[sid] = doc

sample_options = sorted(set(sample_options))
st.sidebar.write(f"Gefundene Samples: {len(sample_options)}")
selected_sample = st.selectbox("Gefiltertes Sample auswählen (Sample ID)", sample_options)

# --- Einzeldokument laden ---
# --- Einzeldokument laden ---
entry = sample_id_to_doc.get(selected_sample)


if entry:
    st.subheader(f"📄 Details für Sample: {selected_sample}")

    # --- Vergleich Target vs Measured: Elemente und Verhältnisse ---
    target_comp = entry.get("values", {}).get("Target Composition", {})
    measured_comp = entry.get("values", {}).get("QCM Composition", {})

    element_keys = [k for k in target_comp if 'at%' in k or k in ['Cs', 'Sn', 'Pb', 'I', 'Br']]
    ratio_keys = [k for k in target_comp if '/' in k]

    def make_bar_plot(keys, title):
        elements = sorted(set(keys))
        df_bar = pd.DataFrame({
            "Key": elements,
            "Target": [target_comp.get(el, 0) for el in elements],
            "Measured": [measured_comp.get(el, 0) for el in elements]
        })
        fig = go.Figure(data=[
            go.Bar(name='Target', x=df_bar["Key"], y=df_bar["Target"], marker_color='blue'),
            go.Bar(name='Measured', x=df_bar["Key"], y=df_bar["Measured"], marker_color='red')
        ])
        fig.update_layout(
            barmode='group',
            title=title,
            xaxis_title="Element / Ratio",
            yaxis_title="Wert (at.% oder Verhältnis)",
            xaxis=dict(showline=True, linewidth=2, linecolor='black'),
            yaxis=dict(showline=True, linewidth=2, linecolor='black'),
        )
        return fig

    col1, col2 = st.columns(2)
    col1.plotly_chart(make_bar_plot(element_keys, "Elementare Zusammensetzung"), use_container_width=True)
    col2.plotly_chart(make_bar_plot(ratio_keys, "Verhältnisse"), use_container_width=True)

    # --- Metadaten ---
    st.markdown("### 🔹 Metadaten")
    st.json(entry.get("metadata", {}))

    st.markdown("### 🔹 Werte")
    st.json(entry.get("values", {}))

    # --- Zeitreihenanzeige ---
    if st.checkbox("📉 Zeitreihen anzeigen"):
        ts = entry.get("time_series", {})
        time = ts.get("Time", {}).get("values", [])

        all_series = []
        for mat, matdata in ts.items():
            if mat == "Time":
                continue
            for category, sub in matdata.items():
                for signal in sub:
                    all_series.append((mat, category, signal))

        selected_series = st.multiselect("Wähle Zeitreihen", [f"{m}/{c}/{s}" for m, c, s in all_series])

        for sel in selected_series:
            m, c, s = sel.split("/")
            values = ts.get(m, {}).get(c, {}).get(s, {}).get("values", [])
            unit = ts.get(m, {}).get(c, {}).get(s, {}).get("unit", "")
            if values:
                st.line_chart(pd.DataFrame({f"{sel} [{unit}]": values}, index=time))