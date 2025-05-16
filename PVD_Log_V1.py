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
    df = results.get("Time Series Raw", df)
    results["Time Series Raw"] = df  # aktualisierte Spalten wieder zurückschreiben

    # QCM-RATET1_x in nmol/s berechnen
    for col in df.columns:
        if col.startswith("QCM RATET1") and col.split()[-1] in df.columns:
            continue  # skip invalid entries
        if col.startswith("QCM RATET1"):
            material = col.split()[-1]
            rate_nm_s = df[col]
            from log_parser import MATERIAL_PROPERTIES, AVOGADRO

            props = MATERIAL_PROPERTIES.get(material)
            if props:
                rho = props['density']
                M = props['molar_mass']
                rate_nmol_s = rate_nm_s * 1e-7 * rho / M * 1e9 #* AVOGADRO   # [nmol/s]
                df[f"QCM RATE {material} nmol_s"] = rate_nmol_s
    df = results.get("Time Series Raw", df)


    # PV vs. QCM-Rate Abweichung berechnen
    pv_cols = [col for col in df.columns if col.endswith("PV")]
    qcm_rate_cols = [col for col in df.columns if col.startswith("QCM RATE") and col.endswith("nmol_s")]
    deviations = {}
    for pv_col in pv_cols:
        mat = pv_col.split()[1]
        matching_qcm = [q for q in qcm_rate_cols if mat in q]
        if matching_qcm:
            qcm_col = matching_qcm[0]
            pv_mean = df[pv_col].mean()
            qcm_mean = df[qcm_col].mean()
            if pv_mean:
                deviation = round((qcm_mean - pv_mean) / pv_mean * 100, 2)
                deviations[f"{mat} QCM vs PV [%]"] = deviation

    if deviations:
        results["QCM-PV Rate Deviations [%]"] = deviations

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

    # Abweichungen zwischen PV und QCM Rate
    if "QCM-PV Rate Deviations [%]" in results:
        st.subheader("📏 QCM vs PV Rate Deviations")
        st.json(results["QCM-PV Rate Deviations [%]"])

    # Optional Zeitreihen-Plot
    with st.expander("📉 Time Series Plot"):
        ts_raw = results.get("Time Series Raw", df)
        ts_filtered = results.get("Time Series Filtered", df)
        if ts_raw is not None:
            use_filtered = st.checkbox("Only show shutter open phase (filtered)", value=False)
            ts_df = ts_filtered.copy() if use_filtered else ts_raw.copy()

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