import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser
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

# --- Abweichungen über Zeit berechnen ---
ratios = ["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]
deviation_data = []
for doc in filtered_docs:
    sample_id = doc.get("metadata", {}).get("Sample_ID", "Unknown")
    date_str = doc.get("metadata", {}).get("Date", "")
    try:
        timestamp = parser.parse(date_str)
    except:
        timestamp = None
    target = doc.get("values", {}).get("Target Composition", {})
    measured = doc.get("values", {}).get("QCM Composition", {})
    for ratio in ratios:
        t_val = target.get(ratio)
        m_val = measured.get(ratio)
        if t_val is not None and m_val is not None:
            deviation = m_val - t_val
            deviation_data.append({
                "Sample_ID": sample_id,
                "Ratio": ratio,
                "Target": t_val,
                "Measured": m_val,
                "Abweichung": deviation,
                "Date": timestamp
            })

# In DataFrame umwandeln
# In DataFrame umwandeln
# In DataFrame umwandeln
if deviation_data:
    df_dev = pd.DataFrame(deviation_data)
    st.dataframe(df_dev)
    st.write(df_dev["Date"].map(type).value_counts())

    if df_dev["Date"].notna().sum() == 0:
        st.info("ℹ️ Keine gültigen Zeitstempel für Abweichungen gefunden.")
    st.subheader("📈 Abweichung Ist-Soll über Zeit")
    for ratio in ratios:
        df_r = df_dev[df_dev["Ratio"] == ratio].dropna()
        if not df_r.empty:
        #    fig = px.scatter(df_r, x="Date", y="Abweichung", color="Sample ID",
        #                     title=f"Abweichung für {ratio}",
        #                     labels={"Abweichung": "Measured - Target"})
        #    fig.update_traces(mode="lines+markers")
        #    fig.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        #                      yaxis=dict(showline=True, linewidth=2, linecolor='black'))
        #    st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()

            df_r_up = df_r[df_r["Abweichung"] >= 0]
            df_r_down = df_r[df_r["Abweichung"] < 0]

            fig.add_trace(go.Bar(
                x=df_r_up["Date"],
                y=df_r_up["Abweichung"],
                name="Abweichung +",
                marker_color="green"
            ))
            fig.add_trace(go.Bar(
                x=df_r_down["Date"],
                y=df_r_down["Abweichung"],
                name="Abweichung -",
                marker_color="red"
            ))

            fig.update_layout(
                title=f"Abweichung für {ratio} (bar chart)",
                xaxis_title="Datum",
                yaxis_title="Measured - Target",
                xaxis=dict(showline=True, linewidth=2, linecolor='black'),
                yaxis=dict(showline=True, linewidth=2, linecolor='black'),
                barmode="relative"
            )

            fig.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor='black'),
                              yaxis=dict(showline=True, linewidth=2, linecolor='black'))
            st.plotly_chart(fig, use_container_width=True)



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




# --- Abweichungen über Zeit berechnen ---
ratios = ["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]
deviation_data = []
for doc in filtered_docs:
    sample_id = doc.get("metadata", {}).get("Sample_ID", "Unknown")
    date_str = doc.get("metadata", {}).get("Date", "")
    try:
        timestamp = datetime.strptime(date_str, "%Y/%m/%d")
    except:
        timestamp = None
    target = doc.get("values", {}).get("Target Composition", {})
    measured = doc.get("values", {}).get("QCM Composition", {})
    for ratio in ratios:
        t_val = target.get(ratio)
        m_val = measured.get(ratio)
        if t_val is not None and m_val is not None:
            deviation = m_val - t_val
            deviation_data.append({
                "Sample_ID": sample_id,
                "Ratio": ratio,
                "Target": t_val,
                "Measured": m_val,
                "Abweichung": deviation,
                "Date": timestamp
            })

# In DataFrame umwandeln
if deviation_data:
    df_dev = pd.DataFrame(deviation_data)
    st.subheader("📈 Abweichung Ist-Soll über Zeit")
    for ratio in ratios:
        df_r = df_dev[df_dev["Ratio"] == ratio].dropna()
        if not df_r.empty:
            fig = px.scatter(df_r, x="Date", y="Abweichung", color="Sample_ID",
                             title=f"Abweichung für {ratio}",
                             labels={"Abweichung": "Measured - Target"})
            fig.update_traces(mode="lines+markers")
            fig.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor='black'),
                              yaxis=dict(showline=True, linewidth=2, linecolor='black'))
            st.plotly_chart(fig, use_container_width=True)



# Verfügbare Sample-IDs sammeln
# Verfügbare Sample-IDs sammeln
xrf_client = MongoClient("mongodb://localhost:27017/")
xrf_db = xrf_client["Fatima_Results_Dataframes"]
xrf_coll = xrf_db["Standard_Collection"]

only_with_xrf = st.sidebar.checkbox("Nur Samples mit XRF-Werten anzeigen")

sample_options = []
sample_id_to_doc = {}
for doc in filtered_docs:
    sid = doc.get("metadata", {}).get("Sample_ID")
    if not sid:
        continue
    if only_with_xrf:
        sid_xrf = sid.replace("-", "_")
        if not xrf_coll.find_one({"Sample_ID": sid_xrf}):
            continue
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
    # Optional: XRF-Werte aus anderer Datenbank abrufen
    xrf_client = MongoClient("mongodb://localhost:27017/")
    xrf_db = xrf_client["Fatima_Results_Dataframes"]
    xrf_coll = xrf_db["Standard_Collection"]
    st.write(selected_sample)
    xrf_sample_id = selected_sample.replace("-", "_")
    st.write(xrf_sample_id)
    xrf_entry = xrf_coll.find_one({"Sample_ID": xrf_sample_id})
    xrf_scaled = {}
    if xrf_entry:
        for el in ["Cs [%]", "Sn [%]", "Pb [%]", "I [%]", "Br [%]"]:
            val_dict = xrf_entry.get(el, {})
            val = val_dict.get(' at') if isinstance(val_dict, dict) else None
            if val is not None:
                el_clean = el.replace(" [%]", "")
                xrf_scaled[el_clean] = val * 20  # normiert auf at.%
                st.write(xrf_scaled[el_clean])
    else:
        st.write('xrf not found')

    target_comp = entry.get("values", {}).get("Target Composition", {})
    measured_comp = entry.get("values", {}).get("QCM Composition", {})

    element_keys = [k for k in target_comp if 'at%' in k or k in ['Cs', 'Sn', 'Pb', 'I', 'Br']]
    ratio_keys = [k for k in target_comp if '/' in k]


    def make_bar_plot(keys, title):
        elements = sorted(set(keys))
        df_bar = pd.DataFrame({
            "Key": elements,
            "Target": [target_comp.get(el, 0) for el in elements],
            "Measured": [measured_comp.get(el, 0) for el in elements],
            "XRF": [xrf_scaled.get(el.replace(" at%", "").replace(" [%]", ""), None) for el in elements] if xrf_scaled else [None] * len(elements)
        })
        fig = go.Figure(data=[
            go.Bar(name='Target', x=df_bar["Key"], y=df_bar["Target"], marker_color='blue'),
            go.Bar(name='Measured', x=df_bar["Key"], y=df_bar["Measured"], marker_color='red')
        ])
        if xrf_scaled:
            fig.add_trace(go.Bar(name='XRF (norm.)', x=df_bar["Key"], y=df_bar["XRF"], marker_color='orange'))
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