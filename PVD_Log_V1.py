import streamlit as st
import pandas as pd
from log_parser import process_log_dataframe_dynamic
import matplotlib.pyplot as plt
from log_parser import load_log_csv

st.set_page_config(page_title="Log File Analysis", layout="wide")
st.title("📊 Thin Film Process Log Analysis")

uploaded_file = st.file_uploader("Upload log file (.csv)", type="csv")

import tempfile

if uploaded_file is not None:
    # Zwischenspeichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    df, metadata = load_log_csv(tmp_path)
    results = process_log_dataframe_dynamic(df, metadata=metadata)


    st.subheader("📁 Raw Data Preview")
    st.write(metadata)
    st.dataframe(df.head())

    with st.spinner("Processing data..."):
        results = process_log_dataframe_dynamic(df)

    st.subheader("🧪 Extracted Results")
    st.json(results)

    # Plot: Element composition (at%)
    st.subheader("Extracted Results")
    # Plot: Element composition (at%)
    col1, col2 = st.columns(2)
    if "QCM at%" in results:
        at_data = results["QCM at%"]
        with col1:
            st.subheader("📈 QCM Element Composition (at%)")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            ax1.bar(at_data.keys(), at_data.values())
            ax1.set_ylabel("at%")
            ax1.set_title("Elemental Composition")
            st.pyplot(fig1)

    # Plot: PV-based target ratios vs. QCM-based measured ratios
    with col2:
        st.subheader("⚖️ Target vs Measured Ratios")
        target = results.get("Target Composition (from PV)", {})
        measured = results.get("Measured Composition (from TSP)", {})
        qcm_ratios = results.get("Deviation (QCM at% vs PV ratios)", {})

        ratio_labels = list(set(target.keys()).union(measured.keys()).union(qcm_ratios.keys()))
        target_vals = [target.get(k, 0) for k in ratio_labels]
        measured_vals = [measured.get(k, 0) for k in ratio_labels]

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        x = range(len(ratio_labels))
        ax2.bar(x, target_vals, width=0.25, label='Target (PV)', align='center')
        ax2.bar([i + 0.25 for i in x], measured_vals, width=0.25, label='Measured (TSP)', align='center')
        ax2.set_xticks([i + 0.25 / 2 for i in x])
        ax2.set_xticklabels(ratio_labels, rotation=45)
        ax2.set_ylabel("Ratio")
        ax2.set_title("Ratio Comparison")
        ax2.legend()
        st.pyplot(fig2)

st.subheader("🧮 Selected Element Ratios from QCM at%")
st.json(results.get("Element Ratios from QCM at%", {}))