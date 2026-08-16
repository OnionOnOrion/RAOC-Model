"""
RAOC Ecotoxicity Layer
======================

Extension to the RAOC kinetic model: converts a simulated treatment trajectory
into predicted aqueous-phase ecotoxicity toward the green alga
Scenedesmus obliquus.

The kinetic model resolves the parent compound and two lumped intermediate
pools, which is not enough to evaluate toxicity: the pools must be resolved
into the individual transformation products whose potencies are known. This
layer performs that speciation using measured pilot-scale product profiles,
then combines the products under concentration addition.

Chain
-----
    1. Reaction progress    xi = 1 - M_SMT(t) / M_SMT(0)
    2. Speciation           product concentrations from templates in xi
    3. Toxic units          TU_j = C_j / EC50_j, summed (concentration
                            addition)
    4. Dose-response        inhibition = TU^n / (1 + TU^n)

Speciation templates
--------------------
    composite sheets   product-to-parent concentration ratios R_j(xi) measured
                       on the pilot RAOC, transferred to any operating
                       condition through xi and capped by the simulated
                       intermediate pool
    TiO2-only sheets   ADMP from direct measurement in UV-on time; the
                       remaining products scaled from the composite ratios at
                       matched xi, with ADMP as the anchor

Resolved products are ADMP, p-AP, SMT-OH and ADMP-OH. Whatever the pool holds
beyond these is carried as an unresolved remainder ("Other") and assigned an
EC50 under several scenarios.

Input
-----
A long-format table with one row per condition and sample time, holding
'#', 't(h)', 'Ms_SMT(mg)', 'Cw_SMT(mg/L)', 'Cw_I1(mg/L)' and 'Cw_I2(mg/L)'.
`records_to_frame` builds this from `raoc_simulator.run_conditions` output.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ============================================================================
# SECTION 1: REFERENCE DATA
# ============================================================================

# -------- Pilot-scale SMT degradation, post-adsorption-equilibrium ---------
# Establishes the mapping from UV-on time to reaction progress xi.
JEM_SMT_DATA = pd.DataFrame({
    't_h':             [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96],
    'water_peak_area': [1960, 1861, 1594, 1377, 1005, 420.8,
                        195.5, 34.6, 18.5, 6.55, 2.18, 1.02],
    'C_water_mgL':     [0.62237802, 0.59127431, 0.50738855, 0.43921173,
                        0.32233718, 0.13879387, 0.06800936, 0.01745798,
                        0.01239970, 0.00864526, 0.00413051, 0.00093845],
    'water_SMT_mg':    [62.237802, 59.127431, 50.738855, 43.921173,
                        32.233718, 13.879387, 6.800936, 1.745798,
                        1.239970, 0.864526, 0.413051, 0.093845],
    'sheet_peak_area': [69631, 61224, 51783, 38762, 22547, 15544,
                        8874, 5614, 2384, 1401, 865, 492],
    'sheet_SMT_mg':    [972.292913, 854.901714, 723.072251, 541.253434,
                        314.835178, 217.048743, 123.912156, 78.391125,
                        33.288999, 19.562872, 12.078433, 6.870045],
    'system_SMT_mg':   [1034.530714, 914.029145, 773.811105, 585.174607,
                        347.068896, 230.928130, 130.713092, 80.136923,
                        34.528969, 20.427398, 12.491484, 6.963890],
})

M_SMT_INIT = JEM_SMT_DATA['system_SMT_mg'].iloc[0]
JEM_SMT_DATA['xi'] = 1.0 - JEM_SMT_DATA['system_SMT_mg'] / M_SMT_INIT

V_BULK_PILOT = 100.0        # pilot bulk volume                           [L]


def t_to_xi(t_h):
    """Map pilot UV-on time [h] to reaction progress xi [-]."""
    return np.interp(t_h, JEM_SMT_DATA['t_h'], JEM_SMT_DATA['xi'])


# -------- Digitised product profiles ---------------------------------------
# Peak areas normalised within each species to its own system-phase maximum,
# so the system column of each frame peaks at 1.00. Absolute concentrations
# come from the anchors below.

T_GRID = np.array([0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96])

SMT_OH = pd.DataFrame({
    't_h':      T_GRID,
    'system':   [0.00, 0.65, 0.97, 1.00, 0.84, 0.42,
                 0.25, 0.13, 0.08, 0.04, 0.02, 0.00],
    'sheet':    [0.00, 0.59, 0.85, 0.81, 0.66, 0.34,
                 0.21, 0.10, 0.06, 0.03, 0.02, 0.00],
    'solution': [0.00, 0.06, 0.13, 0.20, 0.18, 0.08,
                 0.04, 0.02, 0.01, 0.00, 0.00, 0.00],
})

ADMP = pd.DataFrame({
    't_h':      T_GRID,
    'system':   [0.00, 0.18, 0.18, 0.26, 0.39, 0.77,
                 1.00, 0.76, 0.41, 0.18, 0.13, 0.06],
    'sheet':    [0.00, 0.11, 0.13, 0.16, 0.23, 0.50,
                 0.63, 0.47, 0.31, 0.14, 0.10, 0.05],
    'solution': [0.00, 0.07, 0.07, 0.10, 0.16, 0.27,
                 0.37, 0.29, 0.10, 0.04, 0.03, 0.01],
})

ADMP_OH = pd.DataFrame({
    't_h':      T_GRID,
    'system':   [0.00, 0.04, 0.13, 0.23, 0.34, 0.55,
                 0.85, 0.93, 1.00, 0.89, 0.68, 0.36],
    'sheet':    [0.00, 0.03, 0.05, 0.09, 0.14, 0.14,
                 0.21, 0.20, 0.18, 0.18, 0.15, 0.13],
    'solution': [0.00, 0.01, 0.08, 0.14, 0.20, 0.41,
                 0.64, 0.73, 0.82, 0.71, 0.53, 0.23],
})

p_AP = pd.DataFrame({
    't_h':      T_GRID,
    'system':   [0.00, 0.21, 0.55, 0.53, 1.00, 0.82,
                 0.45, 0.25, 0.17, 0.07, 0.04, 0.01],
    'sheet':    [0.00, 0.01, 0.02, 0.02, 0.03, 0.04,
                 0.04, 0.02, 0.02, 0.01, 0.01, 0.00],
    'solution': [0.00, 0.20, 0.55, 0.51, 0.96, 0.78,
                 0.42, 0.23, 0.16, 0.06, 0.03, 0.01],
})

PRODUCT_PROFILES = {
    'SMT-OH': SMT_OH,
    'p-AP': p_AP,
    'ADMP': ADMP,
    'ADMP-OH': ADMP_OH,
}

# Measured water-phase peak concentrations, used to put the normalised
# profiles on an absolute scale.
ANCHOR = {
    'ADMP': {'peak_mgL': 0.51, 't_peak': 36},
    'p-AP': {'peak_mgL': 0.10, 't_peak': 12},
}

# Molar masses, for products assumed to form equimolarly from a precursor
MW_pAP = 109.13            # 4-aminophenol                            [g/mol]
MW_SMT_OH = 294.33         # hydroxylated SMT                         [g/mol]


# -------- Toxicity data ----------------------------------------------------
# EC50 on S. obliquus [mg/L]
EC50_DEFAULT = {
    'SMT': 6.52,
    'ADMP': 1.72,
    'pAP': 0.89,
}

EC50_SMT_OH = 29.7         # hydroxylated SMT                          [mg/L]
EC50_ADMP_OH = 15.6        # hydroxylated ADMP                         [mg/L]

# EC50 assigned to the unresolved remainder [mg/L]:
#   conservative  the less toxic of the two resolved hydroxylated products
#   neutral       their geometric mean
#   corrected     the geometric mean divided by five, allowing for products
#                 more potent than those resolved
EC50_OTHER_DEFAULT = {
    'conservative': EC50_ADMP_OH,
    'neutral': np.sqrt(EC50_SMT_OH * EC50_ADMP_OH),
    'corrected': np.sqrt(EC50_SMT_OH * EC50_ADMP_OH) / 5.0,
}


# ============================================================================
# SECTION 2: SPECIATION TEMPLATES
# ============================================================================

def build_ratio_templates():
    """Product-to-parent water-phase concentration ratios R_j(xi).

    Referencing the products to the residual parent rather than to elapsed
    time makes the templates transferable between operating conditions: two
    conditions at the same xi have passed through the same extent of reaction,
    whatever their rate.

    Returns
    -------
    (R_ADMP, R_pAP, (xi_lo, xi_hi), (R_ADMP_last, R_pAP_last))
        Callables returning the ratio at a given xi. Below xi_lo they return
        zero; above xi_hi they hold the last calibrated value.
    """
    sol_ADMP = PRODUCT_PROFILES['ADMP'][['t_h', 'solution']].copy()
    sol_pAP = PRODUCT_PROFILES['p-AP'][['t_h', 'solution']].copy()

    P_max_ADMP = sol_ADMP.loc[sol_ADMP['t_h'] == ANCHOR['ADMP']['t_peak'],
                              'solution'].iloc[0]
    P_max_pAP = sol_pAP.loc[sol_pAP['t_h'] == ANCHOR['p-AP']['t_peak'],
                            'solution'].iloc[0]
    sol_ADMP['C_mgL'] = sol_ADMP['solution'] * (ANCHOR['ADMP']['peak_mgL']
                                                / P_max_ADMP)
    sol_pAP['C_mgL'] = sol_pAP['solution'] * (ANCHOR['p-AP']['peak_mgL']
                                              / P_max_pAP)

    C_w_SMT = JEM_SMT_DATA['water_SMT_mg'].values / V_BULK_PILOT
    C_at_grid = np.interp(sol_ADMP['t_h'].values,
                          JEM_SMT_DATA['t_h'].values, C_w_SMT)
    xi_at_grid = t_to_xi(sol_ADMP['t_h'].values)

    # Drop t = 0, and points where the parent is too depleted to give a
    # meaningful denominator
    mask = (sol_ADMP['t_h'].values > 0) & (C_at_grid > 1e-4)

    xi_v = xi_at_grid[mask]
    R_A = sol_ADMP['C_mgL'].values[mask] / C_at_grid[mask]
    R_P = sol_pAP['C_mgL'].values[mask] / C_at_grid[mask]

    order = np.argsort(xi_v)
    xi_v, R_A, R_P = xi_v[order], R_A[order], R_P[order]

    xi_lo, xi_hi = xi_v[0], xi_v[-1]

    def _make(interp, hold):
        def fn(xi):
            xi = np.atleast_1d(xi).astype(float)
            out = np.zeros_like(xi)
            in_range = (xi >= xi_lo) & (xi <= xi_hi)
            out[in_range] = interp(xi[in_range])
            out[xi > xi_hi] = hold
            return np.maximum(out, 0.0)
        return fn

    return (_make(PchipInterpolator(xi_v, R_A, extrapolate=False), R_A[-1]),
            _make(PchipInterpolator(xi_v, R_P, extrapolate=False), R_P[-1]),
            (xi_lo, xi_hi), (R_A[-1], R_P[-1]))


def build_system_templates():
    """System-total (water + sheet) product concentrations C_j(xi) [mg/L].

    Each profile is normalised to its own system-phase maximum, so multiplying
    by the peak concentration recovers the absolute trajectory. The ADMP and
    p-AP peaks follow from their water-phase anchors; SMT-OH is derived from
    p-AP assuming equimolar formation, and ADMP-OH from ADMP.

    Returns
    -------
    dict
        Product name -> callable in xi. Below the calibrated range the value
        is zero; above it, the last calibrated value is held.
    """
    scale_ADMP = ANCHOR['ADMP']['peak_mgL'] / 0.37   # ADMP solution at t=36 h
    scale_pAP = ANCHOR['p-AP']['peak_mgL'] / 0.96    # p-AP solution at t=12 h

    system_peak = {
        'ADMP': scale_ADMP,
        'p-AP': scale_pAP,
        # from p-AP, converted by molar mass, rescaled onto the SMT-OH profile
        'SMT-OH': scale_pAP * (MW_SMT_OH / MW_pAP) / 0.20,
        # from ADMP, rescaled onto the ADMP-OH profile
        'ADMP-OH': scale_ADMP / 0.82,
    }

    templates = {}
    for name, df in PRODUCT_PROFILES.items():
        C_sys = df['system'].values * system_peak[name]
        xi_at_t = t_to_xi(df['t_h'].values)

        # t = 0 is excluded from the interpolator and reinstated as a boundary
        mask = df['t_h'].values > 0
        xi_v, C_v = xi_at_t[mask], C_sys[mask]
        order = np.argsort(xi_v)
        xi_v, C_v = xi_v[order], C_v[order]

        def _make(interp, xi_lo, xi_hi, C_hi):
            def fn(xi):
                xi = np.atleast_1d(xi).astype(float)
                out = np.zeros_like(xi)
                in_range = (xi >= xi_lo) & (xi <= xi_hi)
                out[in_range] = interp(xi[in_range])
                out[xi > xi_hi] = C_hi
                return np.maximum(out, 0.0)
            return fn

        templates[name] = _make(
            PchipInterpolator(xi_v, C_v, extrapolate=False),
            xi_v[0], xi_v[-1], C_v[-1])

    return templates


# -------- TiO2-only sheets -------------------------------------------------
# ADMP water-phase concentration measured directly on TiO2-only sheets
TIO2_ADMP_MEAS = {
    't_UV_h': [0, 12, 24, 36, 48],
    'C_mgL': [0.0, 1.05, 1.45, 0.89, 0.44],
}


def build_tio2_templates():
    """Water-phase product concentrations on TiO2-only sheets [mg/L].

    ADMP is interpolated from measurement over the measured window and decays
    exponentially beyond it, at the rate implied by the last two points. The
    other products are obtained by scaling ADMP with the composite-system
    product ratios at matched xi.

    Returns
    -------
    dict
        Product name -> callable (t_UV_h, xi). ADMP ignores xi; the signature
        is kept uniform across the four products.
    """
    t_arr = np.array(TIO2_ADMP_MEAS['t_UV_h'], dtype=float)
    C_arr = np.array(TIO2_ADMP_MEAS['C_mgL'], dtype=float)
    ADMP_interp = PchipInterpolator(t_arr, C_arr, extrapolate=False)
    t_lo, t_hi = t_arr[0], t_arr[-1]
    C_at_t_hi = C_arr[-1]

    k_decay = np.log(C_arr[-2] / C_at_t_hi) / (t_hi - t_arr[-2])   # [1/h]

    ADMP_peak = C_arr.max()
    ADMP_peak_t = t_arr[C_arr.argmax()]

    def C_ADMP(t_UV, xi=None):
        t = np.atleast_1d(t_UV).astype(float)
        out = np.zeros_like(t)
        in_range = (t >= t_lo) & (t <= t_hi)
        out[in_range] = ADMP_interp(t[in_range])
        late = t > t_hi
        out[late] = C_at_t_hi * np.exp(-k_decay * (t[late] - t_hi))
        return np.maximum(out, 0.0)

    sys_tpl = build_system_templates()

    def _ratio_scaled(name):
        def fn(t_UV, xi):
            C_ADMP_t = C_ADMP(t_UV)
            C_ADMP_c = sys_tpl['ADMP'](xi)
            C_j_c = sys_tpl[name](xi)
            ratio = np.where(C_ADMP_c > 1e-9,
                             C_j_c / np.maximum(C_ADMP_c, 1e-12), 0.0)
            return np.maximum(C_ADMP_t * ratio, 0.0)
        return fn

    _C_ADMP_OH_ratio = _ratio_scaled('ADMP-OH')

    def C_ADMP_OH(t_UV, xi):
        # ADMP-OH forms from ADMP, so it cannot have accumulated before the
        # ADMP peak, and afterwards it is bounded by the ADMP consumed.
        C_ratio = _C_ADMP_OH_ratio(t_UV, xi)
        t = np.atleast_1d(t_UV).astype(float)
        upper = np.where(t < ADMP_peak_t, 0.0,
                         np.maximum(0.0, ADMP_peak - C_ADMP(t)))
        return np.maximum(np.minimum(C_ratio, upper), 0.0)

    return {
        'ADMP': C_ADMP,
        'p-AP': _ratio_scaled('p-AP'),
        'SMT-OH': _ratio_scaled('SMT-OH'),
        'ADMP-OH': C_ADMP_OH,
    }


# ============================================================================
# SECTION 3: TOXICITY
# ============================================================================

def _reaction_progress(df):
    """Add the reaction progress column xi = 1 - M_SMT(t) / M_SMT(0)."""
    df = df.sort_values(['#', 't(h)']).reset_index(drop=True)
    Ms0 = df.groupby('#')['Ms_SMT(mg)'].transform('first')
    df['xi'] = 1.0 - df['Ms_SMT(mg)'] / Ms0
    return df


def _add_toxic_units(df, EC50, EC50_OTHER):
    """Add toxic units and their concentration-addition totals."""
    df['TU_SMT'] = df['Cw_SMT(mg/L)'] / EC50['SMT']
    df['TU_ADMP'] = df['Cw_ADMP(mg/L)'] / EC50['ADMP']
    df['TU_pAP'] = df['Cw_pAP(mg/L)'] / EC50['pAP']

    resolved = df['TU_SMT'] + df['TU_ADMP'] + df['TU_pAP']
    for name, value in EC50_OTHER.items():
        df[f'TU_Other_{name}'] = df['Cw_Other(mg/L)'] / value
        df[f'TU_total_{name}'] = resolved + df[f'TU_Other_{name}']
    return df


def compute_toxicity(df, EC50=None, EC50_OTHER=None):
    """Predict toxicity for composite (TiO2/zeolite) sheets.

    ADMP and p-AP follow from the parent concentration through the measured
    ratio templates, then are capped at the simulated intermediate pool with
    their ratio preserved. Whatever the pool holds beyond them is the
    unresolved remainder.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format trajectory; see the module docstring for required columns.
    EC50, EC50_OTHER : dict, optional
        Override the default potencies and remainder scenarios.

    Returns
    -------
    pandas.DataFrame
        Input columns plus xi, speciated concentrations, a 'was_capped' flag,
        and toxic units per scenario.
    """
    if EC50 is None:
        EC50 = EC50_DEFAULT
    if EC50_OTHER is None:
        EC50_OTHER = EC50_OTHER_DEFAULT

    df = _reaction_progress(df.copy())
    R_ADMP, R_pAP, _, _ = build_ratio_templates()

    df['Cw_pool(mg/L)'] = df['Cw_I1(mg/L)'] + df['Cw_I2(mg/L)']

    Cs = df['Cw_SMT(mg/L)'].values
    pool = df['Cw_pool(mg/L)'].values
    raw_ADMP = R_ADMP(df['xi'].values) * Cs
    raw_pAP = R_pAP(df['xi'].values) * Cs
    raw_sum = raw_ADMP + raw_pAP

    # The ratio templates are calibrated on one condition; the simulated pool
    # is the mass actually available. Where the templates exceed it, rescale
    # both products by a common factor so their ratio is preserved.
    need_cap = (raw_sum > pool) & (pool > 1e-9)
    scale = np.ones_like(raw_sum)
    scale[need_cap] = pool[need_cap] / raw_sum[need_cap]

    df['Cw_ADMP(mg/L)'] = raw_ADMP * scale
    df['Cw_pAP(mg/L)'] = raw_pAP * scale
    df['Cw_Other(mg/L)'] = np.maximum(
        0.0, pool - df['Cw_ADMP(mg/L)'] - df['Cw_pAP(mg/L)'])
    df['was_capped'] = need_cap

    return _add_toxic_units(df, EC50, EC50_OTHER)


def compute_toxicity_tio2_only(df, EC50=None, EC50_OTHER=None):
    """Predict toxicity for TiO2-only sheets.

    All four products are resolved from their own templates, so the remainder
    is their hydroxylated fraction rather than an unmeasured residue; it is
    therefore also given a composition-weighted EC50 alongside the scenario
    values. Without a dark adsorption stage, UV-on time equals elapsed time.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format trajectory; see the module docstring for required columns.
    EC50, EC50_OTHER : dict, optional
        Override the default potencies and remainder scenarios.

    Returns
    -------
    pandas.DataFrame
        As `compute_toxicity`, plus the resolved hydroxylated products and a
        composition-weighted 'specific' scenario.
    """
    if EC50 is None:
        EC50 = EC50_DEFAULT
    if EC50_OTHER is None:
        EC50_OTHER = EC50_OTHER_DEFAULT

    df = _reaction_progress(df.copy())
    tpl = build_tio2_templates()

    df['Cw_pool(mg/L)'] = df['Cw_I1(mg/L)'] + df['Cw_I2(mg/L)']

    t_arr = df['t(h)'].values
    xi_arr = df['xi'].values
    df['Cw_ADMP(mg/L)'] = tpl['ADMP'](t_arr, xi_arr)
    df['Cw_pAP(mg/L)'] = tpl['p-AP'](t_arr, xi_arr)
    df['Cw_SMT_OH(mg/L)'] = tpl['SMT-OH'](t_arr, xi_arr)
    df['Cw_ADMP_OH(mg/L)'] = tpl['ADMP-OH'](t_arr, xi_arr)
    df['Cw_Other(mg/L)'] = df['Cw_SMT_OH(mg/L)'] + df['Cw_ADMP_OH(mg/L)']
    df['was_capped'] = False

    df = _add_toxic_units(df, EC50, EC50_OTHER)

    # Composition-weighted EC50 of the remainder, under concentration addition
    C_S = df['Cw_SMT_OH(mg/L)'].values
    C_A = df['Cw_ADMP_OH(mg/L)'].values
    C_O = df['Cw_Other(mg/L)'].values
    present = C_O > 1e-9
    C_safe = np.where(present, C_O, 1.0)
    inv = ((C_S / C_safe) / EC50_SMT_OH + (C_A / C_safe) / EC50_ADMP_OH)
    EC50_O = np.where(present & (inv > 0), 1.0 / np.where(inv > 0, inv, 1.0),
                      np.sqrt(EC50_SMT_OH * EC50_ADMP_OH))
    df['TU_Other_specific'] = C_O / EC50_O
    df['TU_total_specific'] = (df['TU_SMT'] + df['TU_ADMP'] + df['TU_pAP']
                               + df['TU_Other_specific'])
    return df


def TU_to_inhibition(TU, n=1.0):
    """Hill dose-response: growth inhibition as a fraction of the control.

    Returns a value in [0, 1); multiply by 100 for a percentage.
    """
    return TU ** n / (1 + TU ** n)


# ============================================================================
# SECTION 4: INPUT ADAPTER
# ============================================================================

def records_to_frame(results):
    """Build the long-format input table from simulator output.

    Parameters
    ----------
    results : dict
        {id: (condition_tuple, record)}, as returned by
        `raoc_simulator.run_conditions`.

    Returns
    -------
    pandas.DataFrame
        One row per condition and sample time.
    """
    rows = []
    for cid, (cond, rec) in results.items():
        _, rpm, uvi, T_C, AoV = cond
        for i, t_h in enumerate(rec['t_h']):
            rows.append({
                '#': cid, 't(h)': float(t_h),
                'rpm': rpm, 'UVI': uvi, 'T(degC)': T_C, 'A/V': AoV,
                'Ms_SMT(mg)': rec['Ms_P'][i],
                'Cw_SMT(mg/L)': rec['Cb_P'][i],
                'Cw_I1(mg/L)': rec['Cb_I1'][i],
                'Cw_I2(mg/L)': rec['Cb_I2'][i],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import raoc_simulator as sim

    df = compute_toxicity(records_to_frame(sim.run_conditions()))
    centre = df[df['#'] == 1]

    print(f"{'t(h)':>6} {'xi':>7} {'ADMP':>9} {'p-AP':>9} {'Other':>9} "
          f"{'TU_tot':>8} {'inhib':>7}")
    print(f"{'':>6} {'':>7} {'[mg/L]':>9} {'[mg/L]':>9} {'[mg/L]':>9} "
          f"{'':>8} {'':>7}   neutral scenario")
    print("-" * 62)
    for _, r in centre.iterrows():
        print(f"{r['t(h)']:6.2f} {r['xi']:7.3f} {r['Cw_ADMP(mg/L)']:9.4f} "
              f"{r['Cw_pAP(mg/L)']:9.4f} {r['Cw_Other(mg/L)']:9.4f} "
              f"{r['TU_total_neutral']:8.3f} "
              f"{TU_to_inhibition(r['TU_total_neutral']):7.3f}")
