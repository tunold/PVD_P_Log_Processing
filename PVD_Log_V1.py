import streamlit as st
import pandas as pd
from log_parser import process_log_dataframe_dynamic
import matplotlib.pyplot as plt

st.set_page_config(page_title="Log File Analysis", layout="wide")
st.title("📊 Thin Film Process Log Analysis")

uploaded_file = st.file_uploader("Upload log file (.csv)", type="csv")

def load_log_csv(path_or_buffer):
    """
    Läd eine PVD-Logdatei mit Kommentar-Header und tab-separierten Daten.
    Gibt (df, metadata) zurück.
    """
    import datetime

    metadata = {}
    with open(path_or_buffer, 'r', encoding='utf-8') as fh:
        line = fh.readline().strip()

        # Lies Metadaten
        while line.startswith('#'):
            if ':' in line:
                key = line.split(':')[0][1:].strip()
                value = str.join(':', line.split(':')[1:]).strip()
                metadata[key] = value
            line = fh.readline().strip()

        # Lese restliche Datei mit pandas
        df = pd.read_csv(fh, sep='\t')

    # Versuche Zeitinformationen zu kombinieren
    if 'Date' in metadata and 'Time' in df.columns:
        try:
            start_time = datetime.datetime.strptime(f'{metadata["Date"]}T{df["Time"].values[0]}', '%Y/%m/%dT%H:%M:%S')
            end_time = datetime.datetime.strptime(f'{metadata["Date"]}T{df["Time"].values[-1]}', '%Y/%m/%dT%H:%M:%S')
            metadata['Start Time'] = str(start_time)
            metadata['End Time'] = str(end_time)
        except Exception as e:
            metadata['TimeParseError'] = str(e)

    return df, metadata

if uploaded_file is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    df, metadata = load_log_csv(tmp_path)
    results = process_log_dataframe_dynamic(df, metadata=metadata)

    st.subheader("🧪 Extracted Results")
    st.json(results)

    # Plot: Element composition (at%)
    if "QCM at%" in results:
        at_data = results["QCM at%"]
        with st.expander("📈 QCM Element Composition (at%)", expanded=False):
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            ax1.bar(at_data.keys(), at_data.values())
            ax1.set_ylabel("at%")
            ax1.set_title("Elemental Composition")
            st.pyplot(fig1)

        with st.expander("⚖️ Target vs Measured Ratios", expanded=False):
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
            ax2.set_xticks([i + 0.25/2 for i in x])
            ax2.set_xticklabels(ratio_labels, rotation=45)
            ax2.set_ylabel("Ratio")
            ax2.set_title("Ratio Comparison")
            ax2.legend()
            st.pyplot(fig2)

    # Additional ratios from QCM at%
    if "Element Ratios from QCM at%" in results:
        st.subheader("🧮 Selected Element Ratios from QCM at%")
        st.json(results["Element Ratios from QCM at%"])

    # Optional Zeitreihen-Plot
    with st.expander("📉 Time Series Plot"):
        ts_raw = results.get("Time Series Raw")
        ts_filtered = results.get("Time Series Filtered")
        if ts_raw is not None:
            use_filtered = st.checkbox("Only show shutter open phase (filtered)", value=False)
            ts_df = ts_filtered if use_filtered else ts_raw

            variables = [col for col in ts_df.columns if col != "time_seconds"]
            st.write("Select multiple variables per subplot (4 plots total):")
            selected_vars = [
                st.multiselect(f"Y-axis variables for plot {i+1}", variables, key=f"multi_var{i}")
                for i in range(4)
            ]

            fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
            axs = axs.flatten()
            for i, var_list in enumerate(selected_vars):
                for var in var_list:
                    axs[i].plot(ts_df["time_seconds"], ts_df[var], label=var)
                if var_list:
                    axs[i].set_title(", ".join(var_list))
                    axs[i].set_xlabel("Time (s)")
                    axs[i].legend()
            plt.tight_layout()
            st.pyplot(fig)