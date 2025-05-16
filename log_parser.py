"""
log_parser.py — Modul zur Konvertierung und Auswertung von Log-Dateien
Dynamische Extraktion von Prozessdaten, QCM-Dicken, Mittelwerten und Verhältnissen.
"""

import pandas as pd
import numpy as np
import re

AVOGADRO = 6.022e23

# Materialdaten: Dichte [g/cm^3], Molmasse [g/mol], Stöchiometrie {Element: Anzahl}
MATERIAL_PROPERTIES = {
    'PbI2':  {'density': 6.16, 'molar_mass': 461.0,  'stoichiometry': {'Pb': 1, 'I': 2}},
    'CsI':   {'density': 4.51, 'molar_mass': 259.81, 'stoichiometry': {'Cs': 1, 'I': 1}},
    'SnI2':  {'density': 5.32, 'molar_mass': 372.52, 'stoichiometry': {'Sn': 1, 'I': 2}},
    'CsBr':  {'density': 4.43, 'molar_mass': 292.81, 'stoichiometry': {'Cs': 1, 'Br': 1}}
}




def extract_materials(df):
    materials = {}
    pattern = re.compile(r"(\d+) - ([^ ]+) TSP")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            number, material = match.groups()
            materials[int(number)] = material
    return materials

def map_qcm_thickness_columns(df, ordered_materials):
    mapping = {}
    qcm_columns = [
        'QCM1 C_THIK_1',
        'QCM1 C_THIK_2',
        'QCM2 C_THIK_1',
        'QCM2 C_THIK_2'
    ]
    for col, material in zip(qcm_columns, ordered_materials):
        if col in df.columns:
            mapping[col] = f"QCM Thickness {material}"
    return mapping

def compute_composition_ratios(source_values, reference_values=None):
    ratios = {}
    if reference_values is None:
        reference_values = source_values

    A_cations = [k for k in source_values if any(x in k for x in ['Cs', 'Rb', 'FA', 'MA']) and 'Br' not in k]
    B_metals = [k for k in source_values if any(x in k for x in ['Sn', 'Pb', 'Bi']) and 'I' in k]
    Br_sources = [k for k in source_values if 'Br' in k]
    I_sources = [k for k in source_values if 'I' in k and k not in Br_sources]

    def safe_div(numerator, denominator):
        return round(numerator / denominator, 2) if denominator else np.nan

    A_sum = sum(source_values[k] for k in A_cations)
    B_sum = sum(source_values[k] for k in B_metals)
    A_sum_PV = sum(reference_values.get(k, 0) for k in A_cations)
    B_sum_PV = sum(reference_values.get(k, 0) for k in B_metals)

    if A_sum > 0 and B_sum > 0:
        ratios['A_B'] = safe_div(A_sum, B_sum)
    if A_sum_PV > 0 and B_sum_PV > 0:
        ratios['A_B_PV'] = safe_div(A_sum_PV, B_sum_PV)

    I_total = sum(source_values[k] for k in I_sources)
    I_total_PV = sum(reference_values.get(k, 0) for k in I_sources)
    Br_total = sum(source_values[k] for k in Br_sources)
    Br_total_PV = sum(reference_values.get(k, 0) for k in Br_sources)

    if I_total > 0 and Br_total > 0:
        ratios['Br_I'] = safe_div(Br_total, I_total)
    if I_total_PV > 0 and Br_total_PV > 0:
        ratios['Br_I_PV'] = safe_div(Br_total_PV, I_total_PV)

    return ratios

def convert_pv_to_element_rates(pv_dict):
    """
    Rechne PV-Werte (in nmol/s) in Atommengen pro Sekunde um.
    """
    element_flux = {}
    for material, rate_nmol in pv_dict.items():
        props = MATERIAL_PROPERTIES.get(material)
        if not props:
            continue
        atoms_total = rate_nmol * 1e-9 * AVOGADRO  # mol/s → Atome/s
        for el, count in props['stoichiometry'].items():
            element_flux[el] = element_flux.get(el, 0) + atoms_total * count
    return element_flux

def convert_thickness_to_at_percent(thickness_dict):
    """
    Rechne QCM-Dicken in Atomprozente um basierend auf Dichte und Stöchiometrie.
    thickness_dict: Dict mit {"QCM Thickness PbI2": (wert, "nm"), ...}
    """
    element_counts = {}
    for material_key, (thickness_nm, _) in thickness_dict.items():
        material = material_key.replace("QCM Thickness ", "")
        props = MATERIAL_PROPERTIES.get(material)
        if not props:
            continue
        t_cm = thickness_nm * 1e-7
        mass = t_cm * props['density']  # g/cm^2
        mol = mass / props['molar_mass']  # mol/cm^2
        atoms_total = mol * AVOGADRO  # Moleküle pro cm^2
        for el, count in props['stoichiometry'].items():
            element_counts[el] = element_counts.get(el, 0) + atoms_total * count

    total_atoms = sum(element_counts.values())
    if total_atoms == 0:
        return {}
    return {el: round(100 * n / total_atoms, 2) for el, n in element_counts.items()}

def process_log_dataframe_dynamic(df, metadata=None):
    if metadata is None:
        metadata = {}

    results = {}

    df['time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['time_seconds'] = df['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df.drop(columns=['Process Time in seconds'], errors='ignore', inplace=True)

    materials_map = extract_materials(df)
    ordered_materials = [materials_map[k] for k in sorted(materials_map)]
    df.rename(columns=map_qcm_thickness_columns(df, ordered_materials), inplace=True)

    T_cols = [f'{num} - {mat} T' for num, mat in materials_map.items()]
    TSP_cols = [f'{num} - {mat} TSP' for num, mat in materials_map.items()]
    PV_cols = [f'{num} - {mat} PV' for num, mat in materials_map.items()]

    filtered_df = df[(df[T_cols].gt(100).any(axis=1)) & (df['Shutter ShutterAngle0..1'] > 45)]

    TSP_mean = filtered_df[TSP_cols].mean().to_dict()
    for pv_col, tsp_col in zip(PV_cols, TSP_cols):
        condition = filtered_df[pv_col] < 0.05
        filtered_df.loc[condition, tsp_col] = 0
        if condition.any():
            TSP_mean[tsp_col] = 0

    results['TSP_mean_values'] = TSP_mean
    mean_values = {materials_map[int(k.split(' - ')[0])]: v for k, v in TSP_mean.items()}

    PV_mean = filtered_df[PV_cols].mean().to_dict()
    results['PV_mean_values'] = PV_mean
    mean_pv_values = {materials_map[int(k.split(' - ')[0])]: v for k, v in PV_mean.items()}

    # Nutze TSP-Mittelwerte zur Orientierung, PV ist in nmol/s (soll)
        # Berechne Ziel-Zusammensetzung aus PV (nmol/s → atomar)
    element_flux_from_pv = convert_pv_to_element_rates(mean_pv_values)
    results['Target Composition (from PV)'] = compute_composition_ratios(element_flux_from_pv)
    results['Measured Composition (from TSP)'] = compute_composition_ratios(mean_values, mean_values)

    qcm_thickness_cols = [col for col in df.columns if 'QCM Thickness' in col]
    max_thickness = filtered_df[qcm_thickness_cols].max() * 100
    thickness_dict = {k: (round(v, 2), 'nm') for k, v in max_thickness.items()}
    results['QCM Recorded Thickness'] = thickness_dict
    results['Total_thickness'] = (round(max_thickness.sum(), 2), 'nm')

    # Berechne Atomprozente aus QCM-Dicken
    results['QCM at%'] = convert_thickness_to_at_percent(thickness_dict)

    process_time = (filtered_df['time_seconds'].max() - filtered_df['time_seconds'].min()) / 60
    results['Process_time'] = (round(process_time, 2), 'min')

        # Vergleiche Ziel- und Ist-Zusammensetzung (optional)
    target = results.get('Target Composition (from PV)', {})
    measured = compute_composition_ratios(results.get('QCM at%', {}))
    comparison = {}
    for key in target:
        if key in measured:
            t = target[key]
            m = measured[key]
            comparison[f'delta_{key}'] = round(m - t, 2)
    results['Deviation (QCM at% vs PV ratios)'] = comparison

        # Speichere Zeitreihen (ungefiltert und gefiltert) zur optionalen Visualisierung
    results["Time Series Raw"] = df.copy()
    results["Time Series Filtered"] = filtered_df.copy()

    return {**metadata, **results}

