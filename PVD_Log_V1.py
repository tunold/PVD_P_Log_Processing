import streamlit as st
import pandas as pd
from log_parser import process_log_dataframe_dynamic
import matplotlib.pyplot as plt
from itertools import islice
import numpy as np

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

    # Skaliere alle Aout-Werte zur besseren Darstellung
    for col in df.columns:
        if "Aout" in col:
            df[col] = df[col] / 100

    results = process_log_dataframe_dynamic(df, metadata=metadata)
    df = results.get("Time Series Raw", df)
    results["Time Series Raw"] = df  # aktualisierte Spalten wieder zurückschreiben

    # Compute elemental composition and ratios from TSP
    element_keys = ["Cs", "Sn", "Pb", "I", "Br"]
    tsp_cols = [col for col in df.columns if col.endswith("TSP")]
    tsp_totals = {el: 0.0 for el in element_keys}
    for col in tsp_cols:
        for el in element_keys:
            if el in col:
                tsp_totals[el] += df[col].mean()
    tsp_sum = sum(tsp_totals.values())
    if tsp_sum > 0:
        element_tsp_at = {k: round(v / tsp_sum * 100, 2) for k, v in tsp_totals.items()}
        results["Elemental Composition (from TSP)"] = element_tsp_at
        A = tsp_totals.get("Cs", 0)
        B = tsp_totals.get("Sn", 0) + tsp_totals.get("Pb", 0)
        I = tsp_totals.get("I", 0)
        Br = tsp_totals.get("Br", 0)
        if B > 0:
            results["Measured Composition (from TSP)"] = {
                "Cs/(Sn+Pb)": round(A / B, 2),
                "Sn/(Sn+Pb)": round(tsp_totals.get("Sn", 0) / B, 2),
                "Br/(Br+I)": round(Br / (Br + I), 2) if (Br + I) > 0 else 0.0
            }

    # TSP vs PV Summary Table
    with st.expander("📊 TSP vs PV Summary Table", expanded=True):
        rows = ["PbI2", "CsI", "CsBr", "SnI2", "Cs", "Sn", "Pb", "I", "Br", "Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]
        data = {"Name": rows, "TSP": [], "PV": []}

        # Mittelwerte für Materialien
        pv_cols = [col for col in df.columns if col.endswith("PV")]
        tsp_cols = [col for col in df.columns if col.endswith("TSP")]
        shutter_open = df["Shutter ShutterAngle0..1"] > 45


        def get_mean(col, use_filter=False):
            if col in df.columns:
                if use_filter:
                    return df.loc[shutter_open, col].mean()
                return df[col].mean()
            return 0.0


        material_map = {
            "PbI2": "PbI2",
            "CsI": "CsI",
            "CsBr": "CsBr",
            "SnI2": "SnI2"
        }

        for mat in ["PbI2", "CsI", "CsBr", "SnI2"]:
            tsp_val = get_mean(next((c for c in tsp_cols if mat in c), ""))
            pv_val = get_mean(next((c for c in pv_cols if mat in c), ""), use_filter=True)
            data["TSP"].append(round(tsp_val, 3))
            data["PV"].append(round(pv_val, 3))

        # Elemente und Verhältnisse (TSP + PV at%)
        for el in ["Cs", "Sn", "Pb", "I", "Br"]:
            tsp_val = results.get("Elemental Composition (from TSP)", {}).get(el, 0.0)
            pv_val = results.get("Elemental Composition (from PV)", {}).get(el, 0.0)
            data["TSP"].append(round(tsp_val, 2))
            data["PV"].append(round(pv_val, 2))

        for r in ["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]:
            tsp_val = results.get("Measured Composition (from TSP)", {}).get(r, 0.0)
            pv_val = results.get("Target Composition (from PV)", {}).get(r, 0.0)
            data["TSP"].append(round(tsp_val, 2))
            data["PV"].append(round(pv_val, 2))

        comp_df = pd.DataFrame(data)
        st.dataframe(comp_df, use_container_width=False, height=500)

    # Fixed 2x2 plot of TSP, PV, Aout with deviation highlighting
    with st.expander("📉 TSP vs PV vs Aout (per source)", expanded=False):
        source_ids = ["PbI2", "CsI", "CsBr", "SnI2"]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axs = axs.flatten()
        for i, source in enumerate(source_ids):
            tsp_col = next((c for c in df.columns if source in c and "TSP" in c), None)
            pv_col = next((c for c in df.columns if source in c and "PV" in c), None)
            aout_col = next((c for c in df.columns if source in c and "Aout" in c), None)

            if tsp_col and pv_col and aout_col:
                ax = axs[i]
                ax.plot(df["time_seconds"], df[tsp_col], label="TSP")
                ax.plot(df["time_seconds"], df[pv_col], label="PV")
                ax.plot(df["time_seconds"], df[aout_col], label="Aout")
                ax.set_ylim(-0.1, 1.5)
                ax.set_title(source)
                ax.set_xlabel("Time (s)")
                ax.legend()
                # Hintergrund färben bei Abweichung > 30% während Shutter offen
                shutter_open = df["Shutter ShutterAngle0..1"] > 45
                if shutter_open.any():
                    pv_open = df.loc[shutter_open, pv_col]
                    tsp_open = df.loc[shutter_open, tsp_col]
                    deviation_series = abs(pv_open - tsp_open) / tsp_open.replace(0, np.nan)
                    deviation_mean = deviation_series.mean()
                    if deviation_mean > 0.3:
                        ax.set_facecolor("#ffe6e6")  # hellrot

                    # Berechne Mittelwerte während Shutter offen
                if shutter_open.any():
                    pv_mean = df.loc[shutter_open, pv_col].mean()
                    tsp_mean = df[tsp_col].mean()
                    if tsp_mean != 0:
                        deviation_pct = 100 * (pv_mean - tsp_mean) / tsp_mean
                        deviation_str = f"Δ = {deviation_pct:+.1f}%"
                    else:
                        deviation_str = "Δ = n/a"
                    text_str = f"TSP: {tsp_mean:.2f} PV: {pv_mean: .2f} {deviation_str}"
                    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))



        plt.tight_layout()
        st.pyplot(fig)


    # Display TSP settings, compositions, ratios, and totals in table
    with st.expander("📋 Process Overview Table", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Target Composition from TSP**")
            st.write(results.get("Elemental Composition (from TSP)", {}))
            st.markdown("**Target Ratios**")
            st.write(results.get("Measured Composition (from TSP)", {}))

            st.markdown("**⚙️ TSP Setpoints (nmol/s)**")
            tsp_means = {col: round(df[col].mean(), 3) for col in tsp_cols}
            st.write(tsp_means)

        with col2:
            st.markdown("**📏 QCM Composition (from Thickness)**")
            st.write(results.get("QCM at%", {}))
            st.markdown("**Measured Ratios from QCM**")
            st.write(results.get("Element Ratios from QCM at%", {}))

            total_thickness = results.get("Total_thickness", (0,))[0]
            process_time = results.get("Process_time", (0,))[0]
            rate = total_thickness / (process_time * 60) if process_time else 0
            st.markdown("**🕒 Deposition Summary**")
            st.write({
                "Deposition Time (min)": round(process_time, 1),
                "Total Thickness (nm)": round(total_thickness, 1),
                "Deposition Rate (nm/s)": round(rate, 3)
            })

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

    results["Time Series Raw"] = df.copy()  # wichtig, damit auch neue Spalten erhalten bleiben!

    #df = results.get("Time Series Raw", df)


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
    # Get the first key-value pair
    # Get the first 5 key-value pairs
    first_five = dict(islice(results.items(), 7))

    # Display the result
    for key, value in first_five.items():
        st.write(f"{key}:    {value}")

    st.subheader("🧪 Summary Table")
    import numpy as np

    summary_data = []

    # (1) Target composition
    target_comp = results.get("Target Composition (from PV)", {})
    target_at = results.get("Elemental Composition (from PV)", {})
    summary_data.append(["Target Composition (PV)", "", ""])
    for el, val in target_at.items():
        summary_data.append(["", f"{el} at%", val])
    for ratio, val in target_comp.items():
        summary_data.append(["", f"{ratio}", val])

    # (2) Measured QCM composition
    qcm_at = results.get("QCM at%", {})
    qcm_ratios = results.get("Element Ratios from QCM at%", {})
    summary_data.append(["Measured Composition (QCM)", "", ""])
    for el, val in qcm_at.items():
        summary_data.append(["", f"{el} at%", val])
    for ratio, val in qcm_ratios.items():
        summary_data.append(["", f"{ratio}", val])

    # (3) Time / Thickness
    t_sec = results.get("Process_time", (0,))[0] * 60  # in seconds
    total_thickness = results.get("Total_thickness", (0,))[0]
    rate = total_thickness / t_sec if t_sec else 0
    summary_data.append(["Process Info", "Time (s)", round(t_sec, 1)])
    summary_data.append(["", "Total Thickness (nm)", round(total_thickness, 1)])
    summary_data.append(["", "Deposition Rate (nm/s)", round(rate, 3)])

    # (4) QCM XLIFE min/max per QCM
    qcm_life = {k: v for k, v in df.items() if "QCM XLIFE" in k}
    for k, v in qcm_life.items():
        summary_data.append(["QCM Lifetime", f"{k} min", round(np.nanmin(v), 1)])
        summary_data.append(["", f"{k} max", round(np.nanmax(v), 1)])

    # (5) Source Power min/max (Aout)
    aout_cols = [col for col in df.columns if "Aout" in col and "Substrate" not in col]
    for col in aout_cols:
        summary_data.append(["Source Power", f"{col} min", round(df[col].min(), 2)])
        summary_data.append(["", f"{col} max", round(df[col].max(), 2)])

    # (6) Pressure min/max
    for p_col in ["Vacuum Pressure1", "Vacuum Pressure2"]:
        summary_data.append(["Vacuum", f"{p_col} min", round(df[p_col].min(), 3)])
        summary_data.append(["", f"{p_col} max", round(df[p_col].max(), 3)])

    summary_df = pd.DataFrame(summary_data, columns=["Category", "Parameter", "Value"])
    st.dataframe(summary_df, use_container_width=True)


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

    # Vergleich QCM-Rate vs PV (automatisch)
    with st.expander("📐 QCM RATE vs PV (auto)", expanded=False):
        fig_cmp, axs_cmp = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        axs_cmp = axs_cmp.flatten()

        qcm_pv_pairs = []
        # finde alle QCM RATE-Spalten mit Materialnamen
        for col in df.columns:
            if col.startswith("QCM RATE") and col.endswith("nmol_s"):
                mat = col.replace("QCM RATE ", "").replace(" nmol_s", "")
                # finde passende PV-Spalte, z. B. '1 - PbI2 PV'
                pv_matches = [c for c in df.columns if c.endswith(" PV") and mat in c]
                if pv_matches:
                    qcm_pv_pairs.append((mat, col, pv_matches[0]))

        #st.write("Spaltenübersicht:", df.columns.tolist())
        for i, (mat, qcm_col, pv_col) in enumerate(qcm_pv_pairs[:4]):
            qcm_values = df[qcm_col]
            pv_values = df[pv_col]
            axs_cmp[i].plot(df["time_seconds"], qcm_values, label=f"QCM RATE {mat}")
            axs_cmp[i].plot(df["time_seconds"], pv_values, '--', label=f"{mat} PV", color='red')

            deviation = (qcm_values.mean() - pv_values.mean()) / max(pv_values.mean(), 1e-6)
            if abs(deviation) > 0.2:
                axs_cmp[i].set_facecolor('#ffe6e6')  # hellrot bei >20% Abweichung
            elif abs(deviation) > 0.1:
                axs_cmp[i].set_facecolor('#fff5cc')  # hellgelb bei >10%

            axs_cmp[i].set_title(mat)
            axs_cmp[i].set_xlabel("Time (s)")
            axs_cmp[i].set_ylim(-0.05,1.5)
            axs_cmp[i].legend()
            axs_cmp[i].text(0.98, 0.02, f"{deviation * 100:.1f} %", transform=axs_cmp[i].transAxes,
                            fontsize=8, ha='right', va='bottom',
                            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.tight_layout()
        st.pyplot(fig_cmp)

# 🔧 Element Composition Vergleich + TSP-Korrektur
    with st.expander("🧪 Adjust TSP Based on Measured Composition", expanded=False):
        st.markdown("Gebe die gemessene Elementzusammensetzung in at% ein:")
        elements = ['Cs', 'Sn', 'Pb', 'I', 'Br']
        measured_input = {el: st.number_input(f"{el} at%", min_value=0.0, max_value=100.0, value=0.0, step=0.1) for el in elements}

        # Hole TSP-basierte Zusammensetzung
        tsp_comp = results.get("Measured Composition (from TSP)", {})
        st.markdown("---")
        st.markdown("Vergleich mit TSP-basierter Zielvorgabe:")
        st.json(tsp_comp)

        # Korrektur-Vorschlag
        if sum(measured_input.values()) > 0:
            st.markdown("---")
            st.subheader("💡 TSP-Korrekturvorschlag")
            corrected = {}
            scale_factors = {}
            for el in elements:
                if el in tsp_comp and measured_input[el] > 0:
                    scale = tsp_comp[el] / measured_input[el]
                    corrected[el] = round(scale * 100, 1)
                    scale_factors[el] = round(scale, 2)
            st.markdown("**Skalierungsfaktoren für TSP pro Element:**")
            st.json(scale_factors)
            st.markdown("**TSP-Korrekturziel (relativ in %):**")
            st.json(corrected)
