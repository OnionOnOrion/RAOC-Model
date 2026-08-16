"""
RAOC Dimensional Simulator
==========================

Kinetic model of sulfamethazine (SMT) degradation in a rotating advanced
oxidation contactor (RAOC): a horizontal drum coated with TiO2/zeolite
composite sheets, half-submerged in the bulk liquid and irradiated over the
exposed arc.

A fixed point on the sheet alternates between a submerged phase, where solute
is exchanged with the bulk by convective mass transfer, and an exposed phase,
where the liquid film it carries is irradiated and photocatalytic degradation
proceeds. The model resolves three phases (bulk water, liquid film, adsorbed on
zeolite) for the parent compound and a two-pool intermediate chain.

Species
-------
    P    parent (SMT); partitions between bulk, film and zeolite
    I1   hydrophobic intermediates; same three-phase partitioning
    I2   hydrophilic intermediates; aqueous only, no zeolite affinity

State vector (8 elements)
-------------------------
    [Cb_P, Cf_P, q_P, Cb_I1, Cf_I1, q_I1, Cb_I2, Cf_I2]

    C in mg/L, q in mg/g, t in min.

Protocol
--------
Adsorption and irradiation start together at t = 0. Bulk and film are
initialised at C0; nothing is adsorbed.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================================
# SECTION 1: PARAMETERS
# ============================================================================

# -------- Mass transfer: Sherwood correlation from BA dissolution ----------
d_lab = 0.12               # drum diameter                                 [m]
R_lab = 0.06               # drum radius                                   [m]
L_lab = 0.21               # drum length                                   [m]
Bn_lab = 2 * 0.01 / d_lab  # baffle number, n_b h_b / d                    [-]
A_lam = 4.676              # Sh prefactor, laminar (Re <= Re_crit)         [-]
alpha_lam = 0.50           # Sh Re-exponent, laminar                       [-]
A_turb = 1.189             # Sh prefactor, turbulent                       [-]
alpha_turb = 2.0 / 3.0     # Sh Re-exponent, turbulent                     [-]
gamma_Bn = 0.30            # Sh baffle exponent                            [-]
Re_crit = 4031             # laminar/turbulent transition                  [-]
gamma_f = 0.09             # film renewal fraction per cycle               [-]
Delta_h_lab = 100e-6       # film thickness scale                          [m]
c_sheet = 2.52             # Hasan film coefficient for the composite      [-]
g_grav = 9.81              # gravitational acceleration                [m/s^2]


def nu_T(T_C):
    """Water kinematic viscosity [m^2/s] at T_C [degC]."""
    return np.exp(np.interp(T_C, [15, 25, 35],
                            np.log([1.139e-6, 0.893e-6, 0.723e-6])))


def D_BA_T(T_C):
    """Benzoic acid diffusivity [m^2/s] at T_C [degC]."""
    return np.exp(np.interp(T_C, [15, 25, 35],
                            np.log([0.74e-9, 1.00e-9, 1.34e-9])))


MW_BA = 122.12             # benzoic acid molar mass                  [g/mol]
MW_SMT = 278.33            # sulfamethazine molar mass                [g/mol]
# Wilke-Chang scaling of k_L from the BA tracer to SMT
D_scale_BA_to_SMT = ((MW_BA / MW_SMT) ** (1 / 3)) ** (2 / 3)


# -------- Adsorption on zeolite (Langmuir) ---------------------------------
# The model runs on the kinetic constants; the equilibrium constant and
# adsorption enthalpy are retained for reference.
qm_P = 10.47               # parent capacity on the sheet              [mg/g]
KL_P_25 = 1.50             # Langmuir constant at 25 degC              [L/mg]
dH_ads_P = 24.4            # adsorption enthalpy, endothermic        [kJ/mol]
k_ads_P_25 = 0.00874       # adsorption rate constant at 25 degC  [L/(mg min)]
k_des_P_25 = 0.00619       # desorption rate constant at 25 degC      [1/min]
beta_ads = 13.03           # E_a/(R T_ref) for adsorption                 [-]
beta_des = 3.19            # E_a/(R T_ref) for desorption                 [-]
T_ref_K = 298.15           # reference temperature                        [K]

# I1 parameters converted from powder measurements: scaled by the sheet
# utilisation factor observed for the parent, and from mg-C to mg-SMT.
sheet_factor = 10.47 / 149.8        # sheet/powder capacity ratio          [-]
C_frac = 12 * 12 / 278.33           # carbon mass fraction of SMT          [-]
qm_I1_sheet = 21.5 * sheet_factor / C_frac   # I1 capacity            [mg/g]
KL_I1_sheet = 0.29 * C_frac                  # I1 Langmuir constant   [L/mg]
k_ads_I1_25 = 1.87e-2 * C_frac               # I1 adsorption    [L/(mg min)]
k_des_I1_25 = 6.44e-2                        # I1 desorption         [1/min]


# -------- Reaction: parent ------------------------------------------------
SIGMA_K = 125.27           # L-H half-saturation coverage at I_ref  [mg/m^2]
J_MAX_REF = 101.327        # intrinsic max solid-phase flux
                           # at UVI = 1, per exposed area        [mg/(m^2 h)]
K_AQ_SURF = 1.58089e-5     # intrinsic aqueous rate constant,
                           # per exposed area                          [m/s]
Ea_aq = 28.0               # aqueous reaction activation energy     [kJ/mol]
ALPHA_AQ = 0.59            # aqueous UV-intensity exponent                [-]
Ea_J = 8.0                 # solid-phase activation energy          [kJ/mol]
R_GAS = 8.314              # gas constant                        [J/(mol K)]
I_REF_UVI = 1.0            # reference UV index                           [-]

# J_eff and sigma_K both derive from the same photogenerated hole steady
# state, so they share the irradiance exponent ALPHA_SOL. Its value lies
# between the trapping-limited (1) and recombination-limited (0.5) limits.
ALPHA_SOL = 0.87           # solid-phase irradiance exponent              [-]


def sigma_K_of_irradiance(uvi, sigma_K_ref=None, alpha_K=None, I_ref=None):
    """L-H half-saturation coverage sigma_K(I) [mg/m^2].

    sigma_K = sigma_K_ref (I/I_ref)^alpha_K, with alpha_K = ALPHA_SOL by
    default. Pass alpha_K explicitly for sensitivity studies (alpha_K = 0
    recovers a constant sigma_K).
    """
    if sigma_K_ref is None:
        sigma_K_ref = SIGMA_K
    if alpha_K is None:
        alpha_K = ALPHA_SOL
    if I_ref is None:
        I_ref = I_REF_UVI

    if alpha_K == 0.0:
        return sigma_K_ref
    return sigma_K_ref * (uvi / I_ref) ** alpha_K


# -------- Reaction: intermediates ------------------------------------------
# The intermediate decay rate is scaled by the parent retention phi, which
# tracks how far the organic pool has matured: fresh intermediates oxidise
# quickly (mu_fast), terminal residues slowly (mu_slow).
mu_fast_I1 = 5.34          # I1 rate multiplier, fresh pool                [-]
mu_slow_I1 = 0.253         # I1 rate multiplier, refractory pool           [-]
n_I1 = 1.87                # I1 phi-exponent                               [-]
mu_fast_I2 = 1.39          # I2 rate multiplier, fresh pool                [-]
mu_slow_I2 = 0.167         # I2 rate multiplier, refractory pool           [-]
n_I2 = 1.85                # I2 phi-exponent                               [-]
yield_P_to_I1 = 1.00       # no direct mineralisation of the parent        [-]
yield_I1_to_I2 = 1.00      # every I1 reacted becomes I2                   [-]


# -------- Langmuir-Hinshelwood competition on TiO2 -------------------------
K_LH_P = 0.012             # parent                                    [L/mg]
K_LH_I1 = 5.3e-3           # I1                                        [L/mg]
K_LH_I2 = 1.1e-2           # I2                                        [L/mg]


# -------- Material and geometry --------------------------------------------
rho_z_areal = 11.2         # zeolite areal density                 [mg/cm^2]
rho_z_g_per_m2 = 112       # zeolite areal density                   [g/m^2]
F_AIR_LAB = 0.55           # UV-exposed area fraction, half-submerged
                           # lab drum; the pilot drum uses 0.60           [-]


# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def compute_kL_app(rpm, T_C):
    """Apparent mass transfer coefficient for SMT [m/s].

    Sums a submerged convective contribution, from the Sherwood correlation
    calibrated by benzoic acid dissolution and rescaled to SMT by diffusivity,
    and a film-renewal contribution proportional to rotation speed.
    """
    nu = nu_T(T_C)
    DBA = D_BA_T(T_C)
    Re = (rpm / 60) * d_lab ** 2 / nu
    Sc = nu / DBA

    if Re <= Re_crit:
        Sh = A_lam * Re ** alpha_lam * Sc ** (1 / 3) * Bn_lab ** gamma_Bn
    else:
        Sh = A_turb * Re ** alpha_turb * Sc ** (1 / 3) * Bn_lab ** gamma_Bn

    kL_submerged = Sh * DBA / d_lab * D_scale_BA_to_SMT
    kL_film = gamma_f * Delta_h_lab * (rpm / 60)
    return kL_submerged + kL_film


def compute_film_thickness(rpm, T_C):
    """Liquid film thickness on the rotating sheet [m], Hasan form."""
    nu = nu_T(T_C)
    N = rpm / 60
    return c_sheet * np.sqrt(2 * nu * N * R_lab / g_grav)


# ============================================================================
# SECTION 3: FORWARD MODEL
# ============================================================================

def build_simulator(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0, f_air=F_AIR_LAB):
    """Build the ODE right-hand side for one operating condition.

    Parameters
    ----------
    rpm : float
        Drum rotation speed [1/min].
    T_C : float
        Bulk temperature [degC].
    I_mW : float
        UV index, normalised to I_REF_UVI.
    A_cm2 : float
        Total coated sheet area [cm^2].
    V_bulk_L : float
        Bulk liquid volume [L].
    C0 : float
        Initial parent concentration [mg/L].
    f_air : float
        UV-exposed area fraction.

    Returns
    -------
    (rhs, geometry) : callable, dict
        `geometry` caches derived quantities and the Psi_P calibration cache.
    """
    A_m2 = A_cm2 * 1e-4                     # coated area               [m^2]
    w_g = rho_z_areal * A_cm2 / 1000        # zeolite mass on sheet       [g]
    kL = compute_kL_app(rpm, T_C)           # MT coefficient            [m/s]
    delta = compute_film_thickness(rpm, T_C)  # film thickness            [m]
    V_film_L = A_m2 * delta * 1000          # film volume                 [L]
    Lc = w_g / V_film_L                     # loading on a film basis   [g/L]

    T_K = T_C + 273.15
    T_tilde = T_K / T_ref_K

    # Reaction constants at this temperature and irradiance. The solid-phase
    # flux and sigma_K share ALPHA_SOL; the aqueous channel has its own
    # exponent.
    k_aq_T = (K_AQ_SURF * np.exp(-Ea_aq * 1000 / R_GAS * (1 / T_K - 1 / 293.15))
              * (I_mW / I_REF_UVI) ** ALPHA_AQ)
    J_eff_T = (J_MAX_REF
               * np.exp(-Ea_J * 1000 / R_GAS * (1 / T_K - 1 / 293.15))
               * (I_mW / I_REF_UVI) ** ALPHA_SOL)
    sigma_K_T = sigma_K_of_irradiance(I_mW)

    k_ads_P_T = k_ads_P_25 * np.exp(-beta_ads * (1 / T_tilde - 1))
    k_des_P_T = k_des_P_25 * np.exp(-beta_des * (1 / T_tilde - 1))
    k_ads_I1_T = k_ads_I1_25 * np.exp(-beta_ads * (1 / T_tilde - 1))
    k_des_I1_T = k_des_I1_25 * np.exp(-beta_des * (1 / T_tilde - 1))

    # Psi_P is the intrinsic photocatalytic strength of the reactor: set by
    # irradiance, temperature and geometry, independent of how much parent
    # remains. It drives the intermediate chain once the parent is gone. The
    # value below is a steady-state upper bound, used until the runtime
    # calibration locks a value from the linear-decay window.
    sigma_P_max = qm_P * rho_z_g_per_m2
    sat_P_steady = sigma_P_max / (sigma_P_max + sigma_K_T)
    R_solP_steady = f_air * (J_eff_T / 60) * A_m2 * sat_P_steady     # [mg/min]
    denom_0 = 1 + K_LH_P * C0
    R_aqP_init = f_air * k_aq_T * 60 * A_m2 * C0 * 1000 / denom_0    # [mg/min]
    Psi_P_fallback = (R_solP_steady + R_aqP_init) / (C0 * V_bulk_L)  # [1/min]

    _psi_cache = {
        'Psi_P': Psi_P_fallback,
        'samples': [],
        'locked': False,
    }

    def rhs(t, y):
        Cb_P, Cf_P, q_P, Cb_I1, Cf_I1, q_I1, Cb_I2, Cf_I2 = [
            max(yi, 0) for yi in y]

        # Mass transfer between bulk and film
        flux_P = A_m2 * kL * 60 * 1000 * (Cb_P - Cf_P)               # [mg/min]
        flux_I1 = A_m2 * kL * 60 * 1000 * (Cb_I1 - Cf_I1)
        flux_I2 = A_m2 * kL * 60 * 1000 * (Cb_I2 - Cf_I2)

        # Adsorption on zeolite
        ads_P = k_ads_P_T * (qm_P - q_P) * Cf_P - k_des_P_T * q_P
        ads_I1 = k_ads_I1_T * (qm_I1_sheet - q_I1) * Cf_I1 - k_des_I1_T * q_I1

        # Langmuir-Hinshelwood competition on the TiO2 surface
        denom = 1 + K_LH_P * Cf_P + K_LH_I1 * Cf_I1 + K_LH_I2 * Cf_I2

        # Aqueous parent reaction, area-based
        R_aqP = f_air * k_aq_T * 60 * A_m2 * Cf_P * 1000 / denom      # [mg/min]

        # Solid-phase parent reaction; I1 competes for the same sites
        sigma_P = q_P * rho_z_g_per_m2                                # [mg/m^2]
        sigma_I1 = q_I1 * rho_z_g_per_m2
        sat_P = sigma_P / (sigma_P + sigma_I1 * (K_LH_I1 / K_LH_P) + sigma_K_T)
        R_solP = f_air * (J_eff_T / 60) * A_m2 * sat_P                # [mg/min]
        qP_dep = R_solP / w_g                                     # [mg/(g min)]

        # I1 appears where the parent reacted
        prod_I1_solP = R_solP * yield_P_to_I1
        prod_I1_aqP = R_aqP * yield_P_to_I1

        # Process coordinate: parent retention
        M_P_tot = Cb_P * V_bulk_L + Cf_P * V_film_L + q_P * w_g          # [mg]
        Rate_P_total = R_aqP + R_solP
        k_P_eff_now = Rate_P_total / M_P_tot if M_P_tot > 0.01 else 0.0
        M_P_0 = C0 * V_bulk_L
        phi = np.clip(M_P_tot / M_P_0 if M_P_0 > 0 else 0.0, 0.0, 1.0)

        # Calibrate Psi_P on the linear-decay window, away from the initial
        # adsorption transient and from the 0/0 noise at parent depletion.
        if (not _psi_cache['locked']) and 0.50 < phi < 0.90 and M_P_tot > 0.1:
            _psi_cache['samples'].append(k_P_eff_now)
            if len(_psi_cache['samples']) > 50:
                _psi_cache['Psi_P'] = float(np.median(_psi_cache['samples']))
                _psi_cache['locked'] = True
        Psi_P = _psi_cache['Psi_P']

        # Driver for the intermediate decay: the instantaneous parent decay
        # rate while parent is present, blending into Psi_P once it is not.
        # k_P_eff = Rate/M is numerically 0/0 as M -> 0, which would freeze the
        # intermediates; Psi_P sets a small nonzero long-time floor instead.
        phi_blend_mid = 0.01
        phi_blend_w = 0.004
        w_psi = 1.0 / (1.0 + np.exp((phi - phi_blend_mid) / phi_blend_w))
        k_drive = (1.0 - w_psi) * k_P_eff_now + w_psi * Psi_P

        k_dec_I1 = k_drive * (mu_slow_I1 + (mu_fast_I1 - mu_slow_I1) * phi ** n_I1)
        k_dec_I2 = k_drive * (mu_slow_I2 + (mu_fast_I2 - mu_slow_I2) * phi ** n_I2)

        # Intermediates react out of the total aqueous pool: mass transfer is
        # fast enough that bulk and film stay in pseudo-equilibrium, so the
        # sink is distributed between them by mass share.
        M_aq_I1 = Cb_I1 * V_bulk_L + Cf_I1 * V_film_L
        M_aq_I2 = Cb_I2 * V_bulk_L + Cf_I2 * V_film_L
        R_aqI1_total = k_dec_I1 * M_aq_I1 / denom                     # [mg/min]
        R_aqI2_total = k_dec_I2 * M_aq_I2 / denom

        if M_aq_I1 > 1e-12:
            f_b_I1 = (Cb_I1 * V_bulk_L) / M_aq_I1
            f_f_I1 = 1.0 - f_b_I1
        else:
            f_b_I1, f_f_I1 = 1.0, 0.0
        if M_aq_I2 > 1e-12:
            f_b_I2 = (Cb_I2 * V_bulk_L) / M_aq_I2
            f_f_I2 = 1.0 - f_b_I2
        else:
            f_b_I2, f_f_I2 = 1.0, 0.0

        # I2 appears where the I1 reacted
        prod_I2_total = R_aqI1_total * yield_I1_to_I2
        prod_I2_to_bulk = prod_I2_total * f_b_I1
        prod_I2_to_film = prod_I2_total * f_f_I1

        return [
            # parent
            -flux_P / V_bulk_L,
            flux_P / V_film_L - Lc * ads_P - R_aqP / V_film_L,
            ads_P - qP_dep,
            # I1
            -flux_I1 / V_bulk_L - R_aqI1_total * f_b_I1 / V_bulk_L,
            (flux_I1 / V_film_L - Lc * ads_I1
             + prod_I1_aqP / V_film_L - R_aqI1_total * f_f_I1 / V_film_L),
            ads_I1 + prod_I1_solP / w_g,
            # I2
            (-flux_I2 / V_bulk_L
             + prod_I2_to_bulk / V_bulk_L - R_aqI2_total * f_b_I2 / V_bulk_L),
            (flux_I2 / V_film_L
             + prod_I2_to_film / V_film_L - R_aqI2_total * f_f_I2 / V_film_L),
        ]

    geometry = dict(A_m2=A_m2, w_g=w_g, V_film_L=V_film_L, Lc=Lc,
                    kL=kL, delta=delta, k_aq_T=k_aq_T, J_eff_T=J_eff_T,
                    sigma_K_T=sigma_K_T, Psi_P_fallback=Psi_P_fallback,
                    psi_cache=_psi_cache)
    return rhs, geometry


# ============================================================================
# SECTION 4: RUN ONE CONDITION
# ============================================================================

def simulate_condition(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0,
                       sample_h=None, t_max_min=720, f_air=F_AIR_LAB):
    """Integrate one operating condition and report phase-resolved masses.

    Parameters
    ----------
    sample_h : array-like, optional
        Output times [h]. Defaults to a ten-point schedule over 12 h.
    t_max_min : float
        Integration horizon [min]; must cover max(sample_h).

    Returns
    -------
    dict
        Concentrations and loadings at the sample times, the corresponding
        phase masses [mg], first-order rate diagnostics [1/h], and the peak of
        the composite-phase parent inventory.
    """
    if sample_h is None:
        sample_h = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12])
    sample_h = np.asarray(sample_h, dtype=float)
    if sample_h.max() * 60 > t_max_min:
        raise ValueError("sample_h extends beyond t_max_min")

    rhs, geom = build_simulator(rpm, T_C, I_mW, A_cm2, V_bulk_L, C0,
                                f_air=f_air)

    # Bulk and film start at C0; nothing adsorbed, no intermediates
    y0 = [C0, C0, 0, 0, 0, 0, 0, 0]
    sol = solve_ivp(rhs, [0, t_max_min], y0,
                    method='LSODA', rtol=1e-9, atol=1e-11,
                    dense_output=True, max_step=1.0)
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")

    Y = sol.sol(sample_h * 60)
    Cb_P, Cf_P, q_P, Cb_I1, Cf_I1, q_I1, Cb_I2, Cf_I2 = Y

    V_film_L = geom['V_film_L']
    w_g = geom['w_g']

    rec = {
        't_h': sample_h,
        'Cb_P': Cb_P, 'Cf_P': Cf_P, 'q_P': q_P,
        'Cb_I1': Cb_I1, 'Cf_I1': Cf_I1, 'q_I1': q_I1,
        'Cb_I2': Cb_I2, 'Cf_I2': Cf_I2,
        # phase masses [mg]
        'Mw_P': Cb_P * V_bulk_L,
        'Mfilm_P': Cf_P * V_film_L,
        'Mc_P': Cf_P * V_film_L + q_P * w_g,
        'Maq_P': Cb_P * V_bulk_L + Cf_P * V_film_L,
        'Msolid_P': q_P * w_g,
        'Mw_I1': Cb_I1 * V_bulk_L,
        'Mfilm_I1': Cf_I1 * V_film_L,
        'Mc_I1': Cf_I1 * V_film_L + q_I1 * w_g,
        'Maq_I1': Cb_I1 * V_bulk_L + Cf_I1 * V_film_L,
        'Msolid_I1': q_I1 * w_g,
        'Mw_I2': Cb_I2 * V_bulk_L,
        'Mfilm_I2': Cf_I2 * V_film_L,
        'Maq_I2': Cb_I2 * V_bulk_L + Cf_I2 * V_film_L,
    }
    rec['Ms_P'] = rec['Maq_P'] + rec['Msolid_P']
    rec['Ms_I1'] = rec['Maq_I1'] + rec['Msolid_I1']
    rec['Ms_I2'] = rec['Maq_I2']

    # First-order rate over 0-4 h, fitted to total system parent mass
    t_dense = np.linspace(0, 240, 200)
    Y_dense = sol.sol(t_dense)
    Ms_P_d = Y_dense[0] * V_bulk_L + Y_dense[1] * V_film_L + Y_dense[2] * w_g
    valid = Ms_P_d > 0.01 * Ms_P_d[0]
    if valid.sum() > 5:
        rec['k_obs_0_4h'] = np.polyfit(t_dense[valid],
                                       -np.log(Ms_P_d[valid] / Ms_P_d[0]),
                                       1)[0] * 60
    else:
        rec['k_obs_0_4h'] = float('nan')

    # Initial rate over the first 30 min
    t30 = np.linspace(0, 30, 50)
    Y30 = sol.sol(t30)
    Ms30 = Y30[0] * V_bulk_L + Y30[1] * V_film_L + Y30[2] * w_g
    rec['k_obs_init'] = -(np.log(Ms30[-1]) - np.log(Ms30[0])) / 30 * 60

    # Peak of the composite-phase parent inventory
    t_fine = np.linspace(0, t_max_min, 1000)
    Y_fine = sol.sol(t_fine)
    Mc_P_fine = Y_fine[1] * V_film_L + Y_fine[2] * w_g
    i_peak = int(np.argmax(Mc_P_fine))
    rec['Mc_peak_t_min'] = t_fine[i_peak]
    rec['Mc_peak_value'] = Mc_P_fine[i_peak]

    rec['geometry'] = geom
    return rec


# ============================================================================
# SECTION 5: EXPERIMENTAL DESIGN
# ============================================================================

# (id, rpm, UVI, T [degC], A/V [cm^2/L]) at V = 2 L, C0 = 10 mg/L
CONDITIONS = [
    (1, 20, 1.00, 25, 400),    # centre point
    (2, 40, 1.00, 25, 400),    # high rpm
    (3, 5, 1.00, 25, 400),     # low rpm
    (4, 20, 0.50, 25, 400),    # half irradiance
    (5, 20, 2.00, 25, 400),    # double irradiance
    (6, 20, 0.25, 25, 400),    # quarter irradiance
    (7, 10, 1.00, 15, 400),    # low temperature
    (8, 10, 1.00, 35, 400),    # high temperature
    (9, 20, 1.00, 25, 100),    # quarter area loading
    (10, 5, 2.00, 25, 400),    # low rpm, double irradiance
    (11, 20, 2.00, 15, 400),   # double irradiance, low temperature
    (12, 20, 0.50, 35, 400),   # half irradiance, high temperature
    (13, 20, 1.00, 25, 200),   # half area loading
    (14, 20, 1.00, 25, 300),   # three-quarter area loading
]


def run_conditions(conditions=None, C0=10.0, V_bulk_L=2.0,
                   t_max_min=720, sample_h=None, f_air=F_AIR_LAB):
    """Run a set of operating conditions.

    Parameters
    ----------
    conditions : list of tuple, optional
        (id, rpm, UVI, T_C, A/V) tuples. Defaults to CONDITIONS.

    Returns
    -------
    dict
        {id: (condition_tuple, record)}, with records as returned by
        `simulate_condition`.
    """
    if conditions is None:
        conditions = CONDITIONS

    results = {}
    for cond in conditions:
        cid, rpm, uvi, T_C, AoV = cond
        A_cm2 = AoV * V_bulk_L
        results[cid] = (cond, simulate_condition(
            rpm, T_C, uvi, A_cm2, V_bulk_L, C0,
            sample_h=sample_h, t_max_min=t_max_min, f_air=f_air))
    return results


if __name__ == "__main__":
    results = run_conditions()
    print(f"{'#':>3} {'rpm':>4} {'UVI':>5} {'T':>4} {'A/V':>5} "
          f"{'k_init':>9} {'k_0-4h':>9}   [1/h]")
    print("-" * 52)
    for cid, (cond, rec) in results.items():
        _, rpm, uvi, T_C, AoV = cond
        print(f"{cid:3d} {rpm:4d} {uvi:5.2f} {T_C:4d} {AoV:5d} "
              f"{rec['k_obs_init']:9.4f} {rec['k_obs_0_4h']:9.4f}")
