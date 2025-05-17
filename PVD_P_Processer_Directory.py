import streamlit as st
import pandas as pd
from log_parser import process_log_dataframe_dynamic
import matplotlib.pyplot as plt

st.set_page_config(page_title="Log File Analysis", layout="wide")
st.title("📊 Thin Film Process Log Analysis")

import os
from io import StringIO

uploaded_dir = st.text_input("🔍 Enter directory path for batch analysis (leave empty for single upload):")

def load_log_csv(path_or_buffer):
    """
    Läd eine PVD-Logdatei mit Kommentar-Header und tab-separierten Daten.
    Gibt (df, metadata) zurück.
    """
    import datetime

    metadata = {}
    with open(path_or_buffer, 'r', encoding='utf-8') as fh:
        # lies alle Zeilen
        lines = fh.readlines()

    # Extrahiere Metadaten
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('"Time') or line.strip().startswith('Time'):
            data_start = i
            break
        if ':' in line and line.strip().startswith('#'):
            key = line.split(':')[0][1:].strip()
            value = str.join(':', line.split(':')[1:]).strip()
            metadata[key] = value

    # Verwende StringIO für pandas
    data_str = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(data_str), sep='\t')

    # Versuche Zeitinformationen zu kombinieren
    if 'Date' in metadata and 'Time' in df.columns:
        try:
            import datetime
            start_time = datetime.datetime.strptime(f'{metadata["Date"]}T{df["Time"].values[0]}', '%Y/%m/%dT%H:%M:%S')
            end_time = datetime.datetime.strptime(f'{metadata["Date"]}T{df["Time"].values[-1]}', '%Y/%m/%dT%H:%M:%S')
            metadata['Start Time'] = str(start_time)
            metadata['End Time'] = str(end_time)
        except Exception as e:
            metadata['TimeParseError'] = str(e)

    return df, metadata



if uploaded_dir:
    summary_all = []
    files = [f for f in os.listdir(uploaded_dir) if f.endswith(".csv")]
    for fname in files:
        fpath = os.path.join(uploaded_dir, fname)

        df, metadata = load_log_csv(fpath)
        metadata["Filename"] = fname

        if "Time" not in df.columns:
            st.warning(f"❌ 'Time' column missing in file: {fname} — skipped.")
            continue

        # Nur falls df korrekt ist, Aout skalieren
        for col in df.columns:
            if "Aout" in col:
                df[col] = df[col] / 100

        results = process_log_dataframe_dynamic(df, metadata=metadata)


    # Extrahiere summary_data wie im Einzelprozess
        summary_data = {}
        summary_data["Filename"] = fname
        summary_data["Sample ID"] = metadata.get("Substrate Number", "")
        summary_data["Date"] = metadata.get("Date", "")
        summary_data["Operator"] = metadata.get("operator", "")
        summary_data["QCM Total Thickness"] = results.get("Total_thickness", (0,))[0]
        summary_data["Deposition Time (min)"] = results.get("Process_time", (0,))[0]
        summary_data["Deposition Rate (nm/s)"] = round(summary_data["QCM Total Thickness"] / (summary_data["Deposition Time (min)"] * 60), 3) if summary_data["Deposition Time (min)"] else 0
        for el in ["Cs", "Sn", "Pb", "I", "Br"]:
            summary_data[f"Measured {el} at%"] = results.get("QCM at%", {}).get(el, 0)
        ratios = results.get("Element Ratios from QCM at%", {})
        for r in ["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]:
            summary_data[f"Measured {r}"] = ratios.get(r, 0)
        summary_all.append(summary_data)
    summary_df_all = pd.DataFrame(summary_all)
    st.dataframe(summary_df_all, use_container_width=True)
    st.download_button("📥 Download all summaries", summary_df_all.to_csv(index=False).encode("utf-8"), "all_summaries.csv")
else:
    uploaded_file = st.file_uploader("Upload log file (.csv)", type="csv")

    if uploaded_file is not None:
        import tempfile
        import numpy as np
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        df, metadata = load_log_csv(tmp_path)
        # Nur im Einzeldatei-Modus verfügbar
        metadata["Filename"] = uploaded_file.name

        for col in df.columns:
            if "Aout" in col:
                df[col] = df[col] / 100

        results = process_log_dataframe_dynamic(df, metadata=metadata)

        # Compute elemental composition and ratios from TSP
        # Compute elemental composition and ratios from PV during shutter open
        pv_totals = {el: 0.0 for el in element_keys}
        pv_cols = [col for col in df.columns if col.endswith("PV")]
        shutter_open = df["Shutter ShutterAngle0..1"] > 45
        for col in pv_cols:
            for el in element_keys:
                if el in col:
                    pv_totals[el] += df.loc[shutter_open, col].mean()
        pv_sum = sum(pv_totals.values())
        if pv_sum > 0:
            element_pv_at = {k: round(v / pv_sum * 100, 2) for k, v in pv_totals.items()}
            results["Elemental Composition (from PV)"] = element_pv_at
            A = pv_totals.get("Cs", 0)
            B = pv_totals.get("Sn", 0) + pv_totals.get("Pb", 0)
            I = pv_totals.get("I", 0)
            Br = pv_totals.get("Br", 0)
            if B > 0:
                results["Target Composition (from PV)"] = {
                    "Cs/(Sn+Pb)": round(A / B, 2),
                    "Sn/(Sn+Pb)": round(pv_totals.get("Sn", 0) / B, 2),
                    "Br/(Br+I)": round(Br / (Br + I), 2) if (Br + I) > 0 else 0.0
                }
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
            data = {"Name": rows, "TSP": [], "PV": [], "QCM": []}

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
                thickness_dict = results.get("QCM Recorded Thickness", {})
                qcm_val = 0.0
                for k, v in thickness_dict.items():
                    if mat.lower() in k.lower():
                        qcm_val = v[0]
                        break
                data["QCM"].append(round(qcm_val, 1))

            # Elemente und Verhältnisse (TSP + PV at%)
            for el in ["Cs", "Sn", "Pb", "I", "Br"]:
                tsp_val = results.get("Elemental Composition (from TSP)", {}).get(el, 0.0)
                pv_val = results.get("Elemental Composition (from PV)", {}).get(el, 0.0)
                qcm_val = results.get("QCM at%", {}).get(el, 0.0)
                data["TSP"].append(round(tsp_val, 2))
                data["PV"].append(round(pv_val, 2))
                data["QCM"].append(round(qcm_val, 2))

            # Verhältnisse manuell ausrechnen aus den Dictionaries
            def calc_ratios(source):
                A = source.get("Cs", 0)
                Sn = source.get("Sn", 0)
                Pb = source.get("Pb", 0)
                B = Sn + Pb
                I = source.get("I", 0)
                Br = source.get("Br", 0)
                return {
                    "Cs/(Sn+Pb)": round(A / B, 2) if B else 0,
                    "Sn/(Sn+Pb)": round(Sn / B, 2) if B else 0,
                    "Br/(Br+I)": round(Br / (Br + I), 2) if (Br + I) else 0
                }

            qcm_ratios = calc_ratios(results.get("QCM at%", {}))
            tsp_ratios = results.get("Measured Composition (from TSP)", {})
            pv_ratios = results.get("Target Composition (from PV)", {})

            for r in ["Cs/(Sn+Pb)", "Sn/(Sn+Pb)", "Br/(Br+I)"]:
                tsp_val = tsp_ratios.get(r, 0.0)
                pv_val = pv_ratios.get(r, 0.0)
                qcm_val = qcm_ratios.get(r, 0.0)
                data["TSP"].append(round(tsp_val, 2))
                data["PV"].append(round(pv_val, 2))
                data["QCM"].append(round(qcm_val, 2))

                comp_df = pd.DataFrame(data)

            # Letzte Zeile: Gesamtdicke
            total_thickness = results.get("Total_thickness", (0,))[0]
            thickness_row = {"Name": "Total Thickness", "TSP": "", "PV": "", "QCM": round(total_thickness, 1), "% Deviation": ""}
            comp_df = pd.concat([comp_df, pd.DataFrame([thickness_row])], ignore_index=True)
            comp_df["% Deviation"] = ((comp_df["PV"] - comp_df["TSP"]) / comp_df["TSP"]).round(3) * 100

            def highlight_deviation(val):
                if isinstance(val, (int, float)):
                    if abs(val) > 20:
                        return 'background-color: #ffe6e6'  # red for large deviations
                    elif abs(val) > 10:
                        return 'background-color: #fff3cd'  # yellow for medium deviations
                return ''

            styled_df = comp_df.style.applymap(highlight_deviation, subset=["% Deviation"])
            st.dataframe(styled_df, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True)

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
                        text_str = f"TSP: {tsp_mean:.2f} PV: {pv_mean:.2f} {deviation_str}"
                        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                                verticalalignment='top', fontsize=8,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                    ax.set_title(source)
                    ax.set_xlabel("Time (s)")
                    ax.legend()

            plt.tight_layout()
            st.pyplot(fig)
        with st.expander("🌡️ Source Temperatures (T over time)", expanded=False):
            fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
            axs = axs.flatten()
            source_ids = ["PbI2", "CsI", "CsBr", "SnI2"]
            for i, source in enumerate(source_ids):
                t_col = next((c for c in df.columns if source in c and " T" in c), None)
                if t_col:
                    ax = axs[i]
                    ax.plot(df["time_seconds"], df[t_col], label="T")
                    ax.set_title(source)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("T (°C)")
                    ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("📋 Summary Table Export", expanded=True):
            summary_data = {}
            # Basis-Metadaten
            summary_data["Sample ID"] = metadata.get("Substrate Number", "")
            summary_data["Operator"] = metadata.get("operator", "")
            summary_data["Date"] = metadata.get("Date", "")
            summary_data["Time"] = metadata.get("Time", "")
            summary_data["Controller Settings"] = metadata.get("Controller settings", "")
            summary_data["Filename"] = metadata.get("Filename", "")
            summary_data["Recipe"] = metadata.get("Recipe", "")
            summary_data["Process ID"] = metadata.get("process ID", "unknown")

            # TSP, PV, Aout for sources 1–4
            for i, mat in enumerate(["PbI2", "CsI", "CsBr", "SnI2"], start=1):
                tsp_col = next((c for c in df.columns if mat in c and "TSP" in c), None)
                pv_col = next((c for c in df.columns if mat in c and "PV" in c), None)
                aout_col = next((c for c in df.columns if mat in c and "Aout" in c), None)
                t_col = next((c for c in df.columns if mat in c and " T" in c), None)
                qcm_col = next((c for c in df.columns if mat in c and "QCM Thickness" in c), None)
                xl_col = next((c for c in df.columns if mat in c and "XLIFE" in c), None)

                summary_data[f"TSP Source {i}"] = round(df[tsp_col].mean(), 3) if tsp_col else None
                summary_data[f"PV Source {i}"] = round(df[pv_col].mean(), 3) if pv_col else None
                summary_data[f"Aout Source {i}"] = round(df[aout_col].max(), 2) if aout_col else None
                summary_data[f"Max T Source {i}"] = round(df[t_col].max(), 1) if t_col else None
                qcm_value = comp_df.loc[comp_df['Name'] == mat, 'QCM']
                summary_data[f"QCM Thickness Source {i}"] = round(float(qcm_value.values[0]), 2) if not qcm_value.empty else None
                summary_data[f"QCM Life Source {i}"] = round(df[xl_col].max(), 1) if xl_col else None

            summary_data["QCM Total Thickness"] = round(sum(v for k, v in summary_data.items() if "QCM Thickness Source" in k and isinstance(v, (int, float))), 2)
            summary_data["Deposition Time (min)"] = results.get("Process_time", (0,))[0]
            summary_data["Deposition Rate (nm/s)"] = round(summary_data["QCM Total Thickness"] / (summary_data["Deposition Time (min)"] * 60), 3) if summary_data["Deposition Time (min)"] else 0

            # Elemente (aus TSP und QCM at%)
            for i, el in enumerate(["Cs", "Sn", "Pb", "I", "Br"], start=1):
                summary_data[f"Target Element {i} ({el})"] = results.get("Elemental Composition (from TSP)", {}).get(el, 0)
                summary_data[f"Measured Element {i} ({el})"] = results.get("QCM at%", {}).get(el, 0)

            # Verhältnisse
            summary_data["Target Ratio Cs/(Sn+Pb)"] = round(results.get("Measured Composition (from TSP)", {}).get("Cs/(Sn+Pb)", 0), 2)
            summary_data["Target Ratio Sn/(Sn+Pb)"] = round(results.get("Measured Composition (from TSP)", {}).get("Sn/(Sn+Pb)", 0), 2)
            summary_data["Target Ratio Br/(Br+I)"] = round(results.get("Measured Composition (from TSP)", {}).get("Br/(Br+I)", 0), 2)
            summary_data["Measured Ratio Cs/(Sn+Pb)"] = float(comp_df.loc[comp_df['Name'] == "Cs/(Sn+Pb)", 'QCM'].values[0])
            summary_data["Measured Ratio Sn/(Sn+Pb)"] = float(comp_df.loc[comp_df['Name'] == "Sn/(Sn+Pb)", 'QCM'].values[0])
            summary_data["Measured Ratio Br/(Br+I)"] = round(float(comp_df.loc[comp_df['Name'] == "Br/(Br+I)", 'QCM'].values[0]), 2)

            # Drücke
            p1 = df["Vacuum Pressure1"]
            p2 = df["Vacuum Pressure2"]
            summary_data["Pressure1 min"] = f"{p1.min():.2e}"
            summary_data["Pressure1 max"] = f"{p1.max():.2e}"
            summary_data["Pressure2 min"] = f"{p2.min():.2e}"
            summary_data["Pressure2 max"] = f"{p2.max():.2e}"

            summary_rows = [(k, v, "") for k, v in summary_data.items()]
            unit_map = {
                "TSP": "nmol/s", "PV": "nmol/s", "Aout": "%", "Max T": "°C", "QCM Thickness": "nm",
                "QCM Total Thickness": "nm", "Deposition Time": "min", "Deposition Rate": "nm/s",
                "Element": "at.%", "Ratio": "-", "Pressure": "mbar", "Life": "h"
            }
            for i, row in enumerate(summary_rows):
                for key, unit in unit_map.items():
                    if key in row[0]:
                        summary_rows[i] = (row[0], row[1], unit)
                        break
            summary_df = pd.DataFrame(summary_rows, columns=["Parameter", "Value", "Unit"])
            st.dataframe(summary_df, use_container_width=True)

            # Download
            csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Summary as CSV", csv, file_name="summary_table.csv", mime="text/csv")


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
