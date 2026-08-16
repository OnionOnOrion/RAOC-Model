"""
RAOC Dimensionless Model
========================

Dimensionless form of the RAOC kinetic model for a parent compound (P) and two
intermediate pools (I1, I2) in a rotating drum photocatalytic contactor.

Physical constants, the Sherwood correlation for k_L and the film-thickness
correlation are read from the dimensional reference implementation
(`raoc_simulator.py`, path settable via the RAOC_DIM_PATH environment
variable). This module defines no parameter values of its own.


State vector (dimensionless)
----------------------------
    Cb_P,  Cf_P,  q_P        parent in bulk, film, adsorbed
    Cb_I1, Cf_I1, q_I1       intermediate I1 in bulk, film, adsorbed
    Cb_I2, Cf_I2             intermediate I2 in bulk, film (no adsorption)

Concentrations are scaled by C*, solid loadings by q*.


Reference scales
----------------
    C*        = C_0                    bulk concentration scale        [mg/L]
    q*        = q_m,P                  solid loading scale             [mg/g]
    omega_MT  = k_L A / V_bulk         mass-transfer frequency        [1/min]
    t_tilde   = omega_MT t             dimensionless time                 [-]

A is the total coated sheet area. The UV-exposed area fraction f_air enters the
reaction groups rather than omega_MT, because mass transfer acts over the full
rotation cycle while photoreaction acts only over the irradiated arc.


Pi-groups
---------
Geometric
    V_r       = V_film / V_bulk                      film fraction of liquid
    L_c       = w / V_film                    [g/L]  composite loading density
    gamma     = q* / C*                       [L/g]  loading-to-concentration
                                                     scale ratio

Kinetic
    Psi_ads,i = k_ads,i C* / omega_MT                adsorption vs MT rate
    Psi_des,i = k_des,i / omega_MT                   desorption vs MT rate
    Pi_aq     = f_air k_surf,aq / k_L                aqueous reaction velocity
                                                     vs MT velocity
    Pi_sol    = f_air (J_eff/60) A / (w q* omega_MT) solid-phase reaction
                                                     strength

Saturation
    K_LH,i~   = K_LH,i C*                            Langmuir half-saturation
    q_m,i~    = q_m,i / q*                           normalised capacity
    sigma_K~  = sigma_K / (q* rho_z)                 TiO2 site half-saturation

Intermediate chain
    phi       = M_P,tot / M_P,0                      parent retention, used as
                                                     the process coordinate
    Psi_P     = intrinsic photocatalytic strength [1/min], calibrated at
                runtime from the linear-decay window 0.5 < phi < 0.9; sets the
                long-time plateau of the intermediate decay rate
    mu_fast,i, mu_slow,i, n_i                        recalcitrance parameters
                                                     of the phi-driven decay
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

_DEFAULT_DIM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raoc_simulator.py")
_DIM_PATH = os.environ.get("RAOC_DIM_PATH", _DEFAULT_DIM_PATH)
if not os.path.isfile(_DIM_PATH):
    raise FileNotFoundError(
        f"dimensional reference implementation not found at {_DIM_PATH}; "
        "place raoc_simulator.py alongside this file or set RAOC_DIM_PATH")
_spec = importlib.util.spec_from_file_location("raoc_dim", _DIM_PATH)
_dim = importlib.util.module_from_spec(_spec)
sys.modules["raoc_dim"] = _dim
_spec.loader.exec_module(_dim)

F_AIR_DEFAULT = _dim.F_AIR_LAB


# ---------------------------------------------------------------------------
# 1. Reference scales and Pi-groups
# ---------------------------------------------------------------------------
def compute_scales(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=None):
    """Map one physical operating condition onto reference scales and Pi-groups.

    Parameters
    ----------
    rpm : float
        Drum rotation speed [1/min].
    T_C : float
        Bulk temperature [degC].
    I_mW : float
        UV index, normalised to the reference irradiance.
    A_cm2 : float
        Total coated sheet area [cm^2].
    V_bulk_L : float
        Bulk liquid volume [L].
    C0 : float
        Initial parent concentration [mg/L]; also the concentration scale C*.
    f_air : float, optional
        UV-exposed area fraction. Defaults to the half-submerged lab drum.

    Returns
    -------
    dict
        Reference scales, Pi-groups, and the dimensional quantities required by
        the Psi_P calibration.
    """
    if f_air is None:
        f_air = F_AIR_DEFAULT

    # Geometry and transport
    A_m2 = A_cm2 * 1e-4                              # coated area        [m^2]
    w_g = _dim.rho_z_areal * A_cm2 / 1000.0          # zeolite mass         [g]
    kL = _dim.compute_kL_app(rpm, T_C)               # MT coefficient     [m/s]
    delta = _dim.compute_film_thickness(rpm, T_C)    # film thickness       [m]
    V_film_L = A_m2 * delta * 1000.0                 # film volume          [L]
    omega = (A_m2 * kL * 60.0 * 1000.0) / V_bulk_L   # omega_MT         [1/min]

    # Temperature- and irradiance-corrected rate constants
    T_K = T_C + 273.15
    T_tilde = T_K / _dim.T_ref_K

    k_aq = (_dim.K_AQ_SURF
            * np.exp(-_dim.Ea_aq * 1000 / _dim.R_GAS * (1 / T_K - 1 / 293.15))
            * (I_mW / _dim.I_REF_UVI) ** _dim.ALPHA_AQ)          # [m/s]
    J_eff = (_dim.J_MAX_REF
             * np.exp(-_dim.Ea_J * 1000 / _dim.R_GAS * (1 / T_K - 1 / 293.15))
             * (I_mW / _dim.I_REF_UVI) ** _dim.ALPHA_SOL)        # [mg/(m^2 h)]
    sigma_K = _dim.sigma_K_of_irradiance(I_mW)                   # [mg/m^2]

    k_ads_P = _dim.k_ads_P_25 * np.exp(-_dim.beta_ads * (1 / T_tilde - 1))
    k_des_P = _dim.k_des_P_25 * np.exp(-_dim.beta_des * (1 / T_tilde - 1))
    k_ads_I1 = _dim.k_ads_I1_25 * np.exp(-_dim.beta_ads * (1 / T_tilde - 1))
    k_des_I1 = _dim.k_des_I1_25 * np.exp(-_dim.beta_des * (1 / T_tilde - 1))

    # Reference scales
    C_star = float(C0)
    q_star = float(_dim.qm_P)

    # Geometric groups
    V_r = V_film_L / V_bulk_L
    L_c = w_g / V_film_L
    gamma = q_star / C_star

    # Kinetic groups
    Psi_ads_P = (k_ads_P * C_star) / omega
    Psi_des_P = k_des_P / omega
    Psi_ads_I1 = (k_ads_I1 * C_star) / omega
    Psi_des_I1 = k_des_I1 / omega

    Pi_aq_int = k_aq / kL                 # without the exposed-arc correction
    Pi_aq = f_air * Pi_aq_int
    Pi_sol = f_air * (J_eff / 60.0 * A_m2) / (w_g * q_star * omega)

    # Saturation groups
    sigma_K_tilde = sigma_K / (q_star * _dim.rho_z_g_per_m2)
    K_LH_P_t = _dim.K_LH_P * C_star
    K_LH_I1_t = _dim.K_LH_I1 * C_star
    K_LH_I2_t = _dim.K_LH_I2 * C_star
    qm_P_t = _dim.qm_P / q_star
    qm_I1_t = _dim.qm_I1_sheet / q_star

    # Psi_P fallback: steady-state upper bound on the parent decay rate, used
    # until the runtime calibration locks a value.
    sigma_P_max = _dim.qm_P * _dim.rho_z_g_per_m2
    sat_P_steady = sigma_P_max / (sigma_P_max + sigma_K)
    R_solP_steady = f_air * (J_eff / 60.0) * A_m2 * sat_P_steady      # [mg/min]
    denom_0 = 1.0 + _dim.K_LH_P * C0
    R_aqP_init = f_air * k_aq * 60.0 * A_m2 * C0 * 1000.0 / denom_0   # [mg/min]
    Psi_P_fallback = (R_solP_steady + R_aqP_init) / (C0 * V_bulk_L)   # [1/min]

    return dict(
        # dimensional quantities
        A_m2=A_m2, w_g=w_g, V_film_L=V_film_L, V_bulk_L=V_bulk_L,
        kL=kL, delta=delta, omega=omega,
        k_aq=k_aq, J_eff=J_eff, sigma_K=sigma_K,
        k_ads_P=k_ads_P, k_des_P=k_des_P,
        k_ads_I1=k_ads_I1, k_des_I1=k_des_I1,
        # reference scales
        C_star=C_star, q_star=q_star,
        # Pi-groups
        V_r=V_r, L_c=L_c, gamma=gamma,
        Psi_ads_P=Psi_ads_P, Psi_des_P=Psi_des_P,
        Psi_ads_I1=Psi_ads_I1, Psi_des_I1=Psi_des_I1,
        Pi_aq=Pi_aq, Pi_aq_int=Pi_aq_int, Pi_sol=Pi_sol,
        sigma_K_tilde=sigma_K_tilde,
        K_LH_P_t=K_LH_P_t, K_LH_I1_t=K_LH_I1_t, K_LH_I2_t=K_LH_I2_t,
        qm_P_t=qm_P_t, qm_I1_t=qm_I1_t,
        Psi_P_fallback=Psi_P_fallback,
        # operating condition
        rpm=rpm, T_C=T_C, I_mW=I_mW, C0=C0, A_cm2=A_cm2, f_air=f_air,
    )


# ---------------------------------------------------------------------------
# 2. Dimensionless right-hand side
# ---------------------------------------------------------------------------
def build_rhs(s: dict):
    """Assemble the dimensionless right-hand side for the groups in `s`.

    Returns
    -------
    (rhs, psi_cache) : callable, dict
        `psi_cache` is mutated during integration and holds the calibrated
        Psi_P [1/min].
    """
    V_r, L_c, gamma = s["V_r"], s["L_c"], s["gamma"]
    Psi_a_P, Psi_d_P = s["Psi_ads_P"], s["Psi_des_P"]
    Psi_a_I1, Psi_d_I1 = s["Psi_ads_I1"], s["Psi_des_I1"]
    Pi_aq, Pi_sol = s["Pi_aq"], s["Pi_sol"]
    sK = s["sigma_K_tilde"]
    K_P, K_I1, K_I2 = s["K_LH_P_t"], s["K_LH_I1_t"], s["K_LH_I2_t"]
    qmP_t, qmI1_t = s["qm_P_t"], s["qm_I1_t"]

    A_m2, w_g = s["A_m2"], s["w_g"]
    V_b, V_f = s["V_bulk_L"], s["V_film_L"]
    C_star, q_star = s["C_star"], s["q_star"]
    omega, k_aq, J_eff = s["omega"], s["k_aq"], s["J_eff"]
    C0_dim, f_air = s["C0"], s["f_air"]

    mu_f_I1, mu_s_I1, n_I1 = _dim.mu_fast_I1, _dim.mu_slow_I1, _dim.n_I1
    mu_f_I2, mu_s_I2, n_I2 = _dim.mu_fast_I2, _dim.mu_slow_I2, _dim.n_I2
    Y_P_I1 = _dim.yield_P_to_I1
    Y_I1_I2 = _dim.yield_I1_to_I2

    _psi_cache = {
        "Psi_P": s.get("Psi_P_override", s["Psi_P_fallback"]),
        "samples": [],
        "locked": s.get("Psi_P_override") is not None,
    }

    def rhs(t_tilde, y):
        Cb_P, Cf_P, qP, Cb_I1, Cf_I1, qI1, Cb_I2, Cf_I2 = [
            max(yi, 0.0) for yi in y]

        # Langmuir-Hinshelwood competition on the TiO2 surface
        denom = 1.0 + K_P * Cf_P + K_I1 * Cf_I1 + K_I2 * Cf_I2

        # Parent site coverage; I1 competes for the same sites
        sat_P = qP / (qP + qI1 * (K_I1 / K_P) + sK + 1e-30)

        # Parent
        dCb_P = -(Cb_P - Cf_P)
        dCf_P = ((1.0 / V_r) * (Cb_P - Cf_P)
                 - L_c * gamma * (Psi_a_P * (qmP_t - qP) * Cf_P
                                  - Psi_d_P * qP)
                 - (Pi_aq / V_r) * Cf_P / denom)
        dqP = (Psi_a_P * (qmP_t - qP) * Cf_P
               - Psi_d_P * qP
               - Pi_sol * sat_P)

        # Driver for the intermediate chain, evaluated on a system-mass basis
        M_P_tot = (Cb_P * V_b + Cf_P * V_f) * C_star + qP * w_g * q_star  # [mg]
        R_aqP_dim = f_air * k_aq * 60.0 * A_m2 * (Cf_P * C_star) * 1000.0 / denom
        R_solP_dim = f_air * (J_eff / 60.0) * A_m2 * sat_P
        Rate_P_tot = R_aqP_dim + R_solP_dim
        k_P_eff_now = Rate_P_tot / M_P_tot if M_P_tot > 0.01 else 0.0  # [1/min]

        # Process coordinate
        M_P_0 = C0_dim * V_b
        phi = float(np.clip(M_P_tot / M_P_0 if M_P_0 > 0 else 0.0, 0.0, 1.0))

        # Calibrate Psi_P from the linear-decay window
        if (not _psi_cache["locked"]) and 0.50 < phi < 0.90 and M_P_tot > 0.1:
            _psi_cache["samples"].append(k_P_eff_now)
            if len(_psi_cache["samples"]) > 50:
                _psi_cache["Psi_P"] = float(np.median(_psi_cache["samples"]))
                _psi_cache["locked"] = True
        Psi_P = _psi_cache["Psi_P"]

        # k_P_eff while parent is present; Psi_P once it is depleted
        phi_blend_mid, phi_blend_w = 0.01, 0.004
        w_psi = 1.0 / (1.0 + np.exp((phi - phi_blend_mid) / phi_blend_w))
        k_drive = (1.0 - w_psi) * k_P_eff_now + w_psi * Psi_P   # [1/min]

        kd_I1 = (k_drive * (mu_s_I1 + (mu_f_I1 - mu_s_I1) * phi ** n_I1)) / omega
        kd_I2 = (k_drive * (mu_s_I2 + (mu_f_I2 - mu_s_I2) * phi ** n_I2)) / omega

        # Intermediates react out of the total aqueous pool; bulk and film are
        # in pseudo-equilibrium, so sinks are split by mass share
        M_aq_I1 = Cb_I1 + Cf_I1 * V_r
        M_aq_I2 = Cb_I2 + Cf_I2 * V_r
        R_I1 = kd_I1 * M_aq_I1 / denom
        R_I2 = kd_I2 * M_aq_I2 / denom

        if M_aq_I1 > 1e-12:
            fb_I1 = Cb_I1 / M_aq_I1
            ff_I1 = 1.0 - fb_I1
        else:
            fb_I1, ff_I1 = 1.0, 0.0
        if M_aq_I2 > 1e-12:
            fb_I2 = Cb_I2 / M_aq_I2
            ff_I2 = 1.0 - fb_I2
        else:
            fb_I2, ff_I2 = 1.0, 0.0

        # I1 appears where the parent reacted
        prod_I1_film = Y_P_I1 * (Pi_aq / V_r) * Cf_P / denom
        prod_I1_solid = Y_P_I1 * Pi_sol * sat_P

        dCb_I1 = -(Cb_I1 - Cf_I1) - R_I1 * fb_I1
        dCf_I1 = ((1.0 / V_r) * (Cb_I1 - Cf_I1)
                  - L_c * gamma * (Psi_a_I1 * (qmI1_t - qI1) * Cf_I1
                                   - Psi_d_I1 * qI1)
                  + prod_I1_film
                  - R_I1 * ff_I1 / V_r)
        dqI1 = (Psi_a_I1 * (qmI1_t - qI1) * Cf_I1
                - Psi_d_I1 * qI1
                + prod_I1_solid)

        # I2 appears where the I1 reacted
        dCb_I2 = -(Cb_I2 - Cf_I2) + Y_I1_I2 * R_I1 * fb_I1 - R_I2 * fb_I2
        dCf_I2 = ((1.0 / V_r) * (Cb_I2 - Cf_I2)
                  + Y_I1_I2 * R_I1 * ff_I1 / V_r
                  - R_I2 * ff_I2 / V_r)

        return [dCb_P, dCf_P, dqP, dCb_I1, dCf_I1, dqI1, dCb_I2, dCf_I2]

    return rhs, _psi_cache


# ---------------------------------------------------------------------------
# 3. Integration
# ---------------------------------------------------------------------------
def simulate(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=None,
             t_max_min=720.0, n_pts=601, atol=1e-11, rtol=1e-9,
             Psi_P_override=None):
    """Integrate the dimensionless system in t_tilde.

    Parameters
    ----------
    t_max_min : float
        Integration horizon in physical minutes.
    Psi_P_override : float, optional
        Use this Psi_P [1/min] instead of the runtime calibration.

    Returns
    -------
    (df, s) : pandas.DataFrame, dict
        `df` holds both the dimensionless trajectory (`t_tilde`, `*_t` columns)
        and its dimensional rescaling. `s` is the group dictionary, with the
        final Psi_P cache under 'psi_cache_final'.
    """
    s = compute_scales(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=f_air)
    if Psi_P_override is not None:
        s["Psi_P_override"] = float(Psi_P_override)

    rhs, psi_cache = build_rhs(s)

    t_end_tilde = t_max_min * s["omega"]
    t_tilde = np.linspace(0.0, t_end_tilde, n_pts)
    y0 = np.zeros(8)
    y0[0] = 1.0                       # bulk and film both start at C0,
    y0[1] = 1.0                       # matching the dimensional initial condition

    sol = solve_ivp(rhs, (0.0, t_end_tilde), y0, t_eval=t_tilde,
                    method="LSODA", atol=atol, rtol=rtol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    Cs, qs = s["C_star"], s["q_star"]
    df = pd.DataFrame({
        "t_tilde": sol.t,
        "t_min": sol.t / s["omega"],
        "Cb_P": sol.y[0] * Cs, "Cf_P": sol.y[1] * Cs, "q_P": sol.y[2] * qs,
        "Cb_I1": sol.y[3] * Cs, "Cf_I1": sol.y[4] * Cs, "q_I1": sol.y[5] * qs,
        "Cb_I2": sol.y[6] * Cs, "Cf_I2": sol.y[7] * Cs,
        "Cb_P_t": sol.y[0], "Cf_P_t": sol.y[1], "q_P_t": sol.y[2],
        "Cb_I1_t": sol.y[3], "Cf_I1_t": sol.y[4], "q_I1_t": sol.y[5],
        "Cb_I2_t": sol.y[6], "Cf_I2_t": sol.y[7],
    })
    s["psi_cache_final"] = dict(psi_cache)
    return df, s


def prelock_Psi_P(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=None,
                  t_max_min=720.0, atol=1e-11, rtol=1e-9):
    """Return the Psi_P [1/min] obtained from the dimensional implementation.

    The calibration takes the median of k_P_eff over whichever trajectory
    points the adaptive integrator visits inside 0.5 < phi < 0.9, so the two
    implementations lock slightly different values. Passing this value through
    `Psi_P_override` removes that dependence.
    """
    kwargs = {} if f_air is None else {"f_air": f_air}
    rhs_d, geom = _dim.build_simulator(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0,
                                       **kwargs)
    y0 = np.zeros(8)
    y0[0] = y0[1] = C0
    solve_ivp(rhs_d, (0.0, t_max_min), y0, method="LSODA", atol=atol, rtol=rtol)
    return geom["psi_cache"]["Psi_P"]


# ---------------------------------------------------------------------------
# 4. Equivalence check against the dimensional implementation
# ---------------------------------------------------------------------------
def verify_against_dimensional(rpm=20.0, T_C=25.0, I_mW=1.0, A_cm2=800.0,
                               V_bulk_L=2.0, C0=10.0, f_air=None,
                               t_max_min=720.0, prelock=True):
    """Integrate both implementations on a common grid and report deviations.

    Returns
    -------
    (out, report, s) : pandas.DataFrame, dict, dict
        `report` maps each state variable to (max|delta|, RMS, max relative
        deviation), plus 'Psi_P_locked'.
    """
    kwargs = {} if f_air is None else {"f_air": f_air}
    rhs_d, geom = _dim.build_simulator(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0,
                                       **kwargs)

    df_grid, _ = simulate(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=f_air,
                          t_max_min=t_max_min)

    y0 = np.zeros(8)
    y0[0] = y0[1] = C0
    sol = solve_ivp(rhs_d, (0.0, t_max_min), y0,
                    t_eval=df_grid["t_min"].values,
                    method="LSODA", atol=1e-11, rtol=1e-9)
    if not sol.success:
        raise RuntimeError(f"dimensional integration failed: {sol.message}")
    Psi_P_locked = geom["psi_cache"]["Psi_P"]

    df_nd, s = simulate(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=f_air,
                        t_max_min=t_max_min,
                        Psi_P_override=Psi_P_locked if prelock else None)

    idx = {"Cb_P": 0, "Cf_P": 1, "q_P": 2, "Cb_I1": 3, "Cf_I1": 4,
           "q_I1": 5, "Cb_I2": 6, "Cf_I2": 7}
    out = pd.DataFrame({"t_min": df_nd["t_min"].values})
    for name, j in idx.items():
        out[f"{name}_nd"] = df_nd[name].values
        out[f"{name}_dim"] = sol.y[j]

    def _err(a, b):
        d = np.asarray(a) - np.asarray(b)
        scale = np.max(np.abs(b)) + 1e-30
        return (float(np.max(np.abs(d))),
                float(np.sqrt(np.mean(d ** 2))),
                float(np.max(np.abs(d)) / scale))

    report = {k: _err(out[f"{k}_nd"], out[f"{k}_dim"]) for k in idx}
    report["Psi_P_locked"] = Psi_P_locked
    return out, report, s


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def print_groups(s):
    """Print the Pi-group inventory for one operating condition."""
    line = "=" * 68
    print(line)
    print(f" Pi-groups | rpm={s['rpm']}, T={s['T_C']} degC, UVI={s['I_mW']}, "
          f"C0={s['C0']} mg/L")
    print(f"           | A={s['A_cm2']} cm2, V={s['V_bulk_L']} L, "
          f"f_air={s['f_air']}")
    print(line)
    print(f"  omega_MT       = {s['omega']:.6f} 1/min   "
          f"(1/omega_MT = {1/s['omega']:.2f} min)")
    print(f"  V_r            = {s['V_r']:.4e}")
    print(f"  L_c            = {s['L_c']:.4e}  g/L")
    print(f"  gamma          = {s['gamma']:.4e}  L/g")
    print(f"  Psi_ads,P      = {s['Psi_ads_P']:.4e}")
    print(f"  Psi_des,P      = {s['Psi_des_P']:.4e}")
    print(f"  Psi_ads,I1     = {s['Psi_ads_I1']:.4e}")
    print(f"  Psi_des,I1     = {s['Psi_des_I1']:.4e}")
    print(f"  Pi_aq          = {s['Pi_aq']:.4e}   "
          f"(intrinsic {s['Pi_aq_int']:.4e}, x f_air)")
    print(f"  Pi_sol         = {s['Pi_sol']:.4e}")
    print(f"  sigma_K~       = {s['sigma_K_tilde']:.4e}")
    print(f"  K_LH,P~        = {s['K_LH_P_t']:.4e}")
    print(f"  q_m,I1~        = {s['qm_I1_t']:.4e}")
    print(f"  Psi_P fallback = {s['Psi_P_fallback']:.4e} 1/min")
    print(line)


if __name__ == "__main__":
    out, report, s = verify_against_dimensional()
    print_groups(s)
    print("\nDimensionless vs. dimensional implementation, 0-720 min:")
    print("-" * 68)
    for key in [k for k in report if k != "Psi_P_locked"]:
        mx, rms, rel = report[key]
        print(f"  {key:6s}: max|d| = {mx:.3e}   RMS = {rms:.3e}   "
              f"max-rel = {rel:.2e}")
    print(f"\n  Psi_P (locked) = {report['Psi_P_locked']:.4e} 1/min")
