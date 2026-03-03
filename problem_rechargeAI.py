import numpy as np
import csdl_alpha as csdl
import time

from csdl_alpha.experimental import PySimulator
from modopt import CSDLAlphaProblem, TrustConstr, SLSQP

import funlib

# ---------- smooth helpers ----------
def smooth_positive(z, eps=1e-3, sqrt_fn=csdl.sqrt):
    """Provides a continuously differentiable max(z,0) function which provides [0, inf] output with any z (only positive)
    
    Args:
        z (float): Number
        eps (float): Smoothness factor (Higher = Smoother, less accurate)
        sqrt_fun (function): CSDL square-root
    Returns:
        out (float): Approximation of max(z,0)
    """
    # C1 approximation of max(z, 0)
    return 0.5 * (z + sqrt_fn(z * z + eps * eps))

# ---------- A) Parameters ----------
def get_problem_params():
    h = 12000.0
    return dict(
        # https://www.spectrolab.com/DataSheets/Panel/panels.pdf
        sigma_panel = 3,                # Area density of solar panels (kg/m^2)

        # https://www.airbusdefenceandspacenetherlands.nl/dsstuff/uploads/2019/11/SWdatasheet062020web.pdf
        P_area = 250.0,                 # Power per area (W/m^2)

        # https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/#3.4
        E_spec = 250.0 * 3600.0,        # Energy density of battery (J/kg)

        rho = funlib.isa_density(h),    # Density at given altitude (kg/m^3)
        g = 9.80665,                    # Gravity (m/s^2)
        R_loiter = 5000.0,              # Loiter circle radius (m)
        CD0 = 0.03,                     # Zero-lift drag coefficient
        e = 0.80,                       # Oswald efficiency factor
        CL_max = 1.4,                   # Maximum lift coefficient
        stall_SF = 1.2,                 # Stall safety factor
        eta_prop = 0.80,                # Propeller efficiency
        f_day = 0.50,                   # Fraction of daylight during the day
        eta_mppt = 0.98,                # Transformer efficiency
        eta_batt = 0.95,                # Battery efficiency
        m_payload = 30.0,               # Payload mass (kg)
        a_struct = 1.2,                 # Structural mass for wing area (kg/m^2)
        b_struct = 0.04,                # Structural mass for wingspan (kg/m)
        WS_max = 250.0,                 # Maximum wing loading (kg/m^2)
        b_max = 40.0,                   # Maximum wingspan (m)
        P_max = 12000.0,                # Maximum power draw (W)

        # Keep optimization in finite-endurance regime (W). If this is too restrictive,
        # lower it, but keep it > 0 to avoid unbounded endurance.
        P_batt_avg_min = 200.0,           # Minimum average battery power (W)
    )

# ---------- B) Geometry ----------
def add_geometry(S, AR, sqrt_fn=csdl.sqrt):
    """Calculates wingspan with aspect ratio and wing area
    
    Args:
        S (float): Wing area (m^2)
        AR (float): Aspect ratio
        sqrt_fn (function): CSDL square-root
    
    Returns:
        out (float): Wingspan (m)
    """
    b = sqrt_fn(AR * S)
    return b

# ---------- C) Mass model ----------
def add_mass_model(S, AR, m_batt, f_panel, params, sqrt_fn=csdl.sqrt):
    """Mass model for aircraft
    Args:
        S (float): Wing area (m^2)
        AR (float): Aspect ratio
        m_batt (float): Mass of battery (kg)
        f_panel (float): Fraction of wing that is solar panel
        params (dict): Model constants
        sqrt_fn (function): CSDL square-root

    Returns:
        out (tuple):\n
            out[0]: Total weight (N)\n
            out[1]: Total mass (kg)\n
            out[2]: Solar panel area (m^2)
    """
    g = params["g"] # Gravity
    m_payload = params["m_payload"] # Payload mass
    sigma_panel = params["sigma_panel"] # Area density of solar panels
    a_struct = params["a_struct"] #
    b_struct = params["b_struct"] #

    A_panel = f_panel * S # Area of solar panels
    m_solar = sigma_panel * A_panel # Mass of solar panels
    m_struct = a_struct * S + b_struct * sqrt_fn(AR * S) # Mass of wing structure

    m_total = m_payload + m_batt + m_solar + m_struct # Total mass
    W = m_total * g # Total weight
    
    return W, m_total, A_panel

# ---------- D) Aero + loiter ----------
def add_aero_loiter(S, AR, V, W, params, sqrt_fn=csdl.sqrt):
    """Aerodynamic model for aircraft
    Args:
        S (float): Wing area (m^2)
        AR (float): Aspect ratio
        V (float): Aircraft velocity (m/s)
        W (float): Weight of aircraft (N)
        params (dict): Model constants
        sqrt_fn (function): CSDL square-root

    Returns:
        out (tuple):\n
            out[0]: Lift coefficient\n
            out[1]: Drag coefficient\n
            out[2]: Drag (N)\n
            out[3]: Load factor
    """
    rho = params["rho"] # Density of air
    g = params["g"] # Gravity
    R_loiter = params["R_loiter"] # Loiter circle radius (m)
    CD0 = params["CD0"] # Zero-lift drag coefficient
    e = params["e"] # Oswald efficiency factor

    q = 0.5 * rho * V**2 # Dynamic Pressure
    n = sqrt_fn(1.0 + (V**2 / (g * R_loiter))**2) # Load Factor
    L_req = n * W # Lift required
    CL = L_req / (q * S) # Lift coefficient

    k = 1.0 / (np.pi * e * AR) # Induced drag factor
    CD = CD0 + k * CL**2 # Total drag coefficient
    D = q * S * CD # Total drag

    return CL, CD, D, n

# ---------- E) Power + energy + endurance ----------
def add_power_energy_endurance(S, V, m_batt, A_panel, D, params, positive_fn=smooth_positive):
    """
    Args:
        S (float): Wing area (m^2)
        V (float): Velocity of aircraft (m/s)
        m_batt (float): Mass of batteries (kg)
        D (float): Total drag (N)
        params (dict): Model constants
        positive_fn (function):

    Returns:
        out (tuple):\n
            out[0]: Required power (W)\n
            out[1]: Power from solar (W)\n
            out[2]: Average battery power use over 1 day (W)\n
            out[3]: Endurance time (s)
    """
    eta_prop = params["eta_prop"] # Propeller efficiency
    P_area = params["P_area"] # Power per area
    eta_mppt = params["eta_mppt"] # Transformer efficiency
    f_day = params["f_day"] # Fraction of day in sunlight
    eta_batt = params["eta_batt"] # Battery efficiency
    E_spec = params["E_spec"] # Energy density of battery

    P_req = (D * V) / eta_prop # Required propeller power
    P_solar = P_area * eta_mppt * A_panel # Power generated by solar

    E_avail = eta_batt * E_spec * m_batt # Total energy capacity

    # Signed battery power draw:
    # >0 discharge, <0 charge. Day can recharge when solar exceeds required power.
    P_batt_day = P_req - P_solar
    P_batt_avg_cycle = f_day * P_batt_day + (1.0 - f_day) * P_req

    # Endurance must use battery depletion rate (not charging rate).
    # If net is near/under 0, endurance is effectively unbounded.
    P_batt_deplete = positive_fn(P_batt_avg_cycle, eps=1e-3) + 1e-3
    T_endurance = E_avail / P_batt_deplete

    return P_req, P_solar, P_batt_avg_cycle, T_endurance

# ---------- F) Constraints + objective ----------
def add_constraints_objective(S, b, W, CL, P_req, P_batt_avg_cycle, T_endurance, params):
    """Defines objective and constraints for CSDL

    Args:
        S (float): Wing area (m^2)
        b (float): Wingspan (m)
        W (float): Weight of aircraft (N)
        CL (float): Lift coefficient
        P_req (float): Required power (W)
        P_batt_avg_cycle (float): Average battery power use over 1 day (W)
        T_endurance (float): Endurance time (s)
        params (dict): Model constants
    """
    CL_max = params["CL_max"] # Maximum lift coefficient
    stall_SF = params["stall_SF"] # Stall safety factor
    WS_max = params["WS_max"] # Maximum wing loading
    b_max = params["b_max"] # Maximum wingspan
    P_max = params["P_max"] # Maximum power draw
    P_batt_avg_min = params["P_batt_avg_min"] # Minimum average battery power

    # Objective
    (-T_endurance).set_as_objective() # Maximize endurance

    # Constraints (all <= 0 form)
    (CL - (CL_max / stall_SF)).set_as_constraint(upper=0.0) # Lift coefficient constraint with stall safety factor
    ((W / S) - WS_max).set_as_constraint(upper=0.0) # Wing loading constraint
    (b - b_max).set_as_constraint(upper=0.0) # Wingspan constraint
    (P_req - P_max).set_as_constraint(upper=0.0) # Power constraint

    # Prevent near-infinite endurance / division singularity by requiring positive net battery depletion over the day-night cycle.
    # P_batt_avg_cycle >= P_batt_avg_min
    (P_batt_avg_min - P_batt_avg_cycle).set_as_constraint(upper=0.0)

# ---------- 1) CSDL builder ----------
def build_recorder(use_solar=True):
    params = get_problem_params()

    rec = csdl.Recorder()
    rec.start()

    # Design variables
    S = csdl.Variable(name="S", value=30.0)
    AR = csdl.Variable(name="AR", value=22.0)
    V = csdl.Variable(name="V", value=30.0)
    m_batt = csdl.Variable(name="m_batt", value=60.0)
    f_panel = csdl.Variable(name="f_panel", value=0.6)

    S.set_as_design_variable(lower=5.0, upper=50.0)
    AR.set_as_design_variable(lower=10.0, upper=30.0)
    V.set_as_design_variable(lower=15.0, upper=45.0)
    m_batt.set_as_design_variable(lower=5.0, upper=200.0)
    f_panel.set_as_design_variable(lower=0.0, upper=1.0)

    # Build model via functions
    b = add_geometry(S, AR)
    W, m_total, A_panel = add_mass_model(S, AR, m_batt, f_panel, params)
    CL, CD, D, n = add_aero_loiter(S, AR, V, W, params)
    P_req, P_solar, P_batt_avg_cycle, T_endurance = add_power_energy_endurance(
        S, V, m_batt, A_panel, D, params
    )

    # Name variables
    b.add_name("wingspan")
    W.add_name("weight")
    CL.add_name("CL_loiter")
    P_req.add_name("P_req")
    P_solar.add_name("P_solar")
    P_batt_avg_cycle.add_name("P_batt_avg_cycle")
    T_endurance.add_name("T_endurance_s")

    add_constraints_objective(S, b, W, CL, P_req, P_batt_avg_cycle, T_endurance, params)

    rec.stop()
    return rec

# ---------- 2) Create Simulator ----------
def make_problem(problem_name="solar_uav"):
    rec = build_recorder(use_solar=True)
    sim = PySimulator(rec)
    return CSDLAlphaProblem(problem_name=problem_name, simulator=sim), sim

# ---------- 3) Solving Problem ----------
def run_opt(problem):
    optimizer = TrustConstr(problem, solver_options={"maxiter": 800, "barrier_tol": 1e-8})
    # optimizer = SLSQP(problem, solver_options={"maxiter": 500, "ftol": 1e-8})

    t0 = time.perf_counter()
    optimizer.solve()
    elapsed = time.perf_counter() - t0
    optimizer.print_results()

    results = getattr(optimizer, "results", {})
    if not isinstance(results, dict):
        results = {}

    return results, elapsed

# ---------- 4) Deriving optimized values ----------
def derive_outputs_from_design(x, params):
    S, AR, V, m_batt, f_panel = np.asarray(x).astype(float).ravel()

    wingspan = add_geometry(S, AR, sqrt_fn=np.sqrt)
    weight, _, A_panel = add_mass_model(S, AR, m_batt, f_panel, params)
    CL_loiter, _, D, _ = add_aero_loiter(S, AR, V, weight, params, sqrt_fn=np.sqrt)
    smooth_positive_np = lambda z, eps=1e-3: smooth_positive(z, eps=eps, sqrt_fn=np.sqrt)
    P_req, P_solar, P_batt_avg_cycle, T_endurance_s = add_power_energy_endurance(
        S, V, m_batt, A_panel, D, params, positive_fn=smooth_positive_np
    )

    return {
        "S": float(S),
        "AR": float(AR),
        "V": float(V),
        "m_batt": float(m_batt),
        "f_panel": float(f_panel),
        "wingspan": float(wingspan),
        "weight": float(weight),
        "CL_loiter": float(CL_loiter),
        "P_req": float(P_req),
        "P_solar": float(P_solar),
        "P_batt_avg_cycle": float(P_batt_avg_cycle),
        "T_endurance_s": float(T_endurance_s),
        "T_endurance_h": float(T_endurance_s/3600)
    }

# ---------- 5) Print Solutions ----------
def print_solution_summary(sim, results):
    print("\nOptimized values:")

    x = results.get("x", None)
    if x is None:
        print("  unavailable (solver did not return design vector 'x')")
        return

    vals = derive_outputs_from_design(x, get_problem_params())
    for key in (
        "S",
        "AR",
        "V",
        "m_batt",
        "f_panel",
        "wingspan",
        "weight",
        "CL_loiter",
        "P_req",
        "P_solar",
        "P_batt_avg_cycle",
        "T_endurance_s",
        "T_endurance_h"
    ):
        print(f"  {key}: {vals[key]:.6g}")

# ---------- Script ----------
if __name__ == "__main__":
    problem, sim = make_problem(problem_name="solar_uav_slsqp")
    results, _ = run_opt(problem)
    try:
        sim.run()
    except Exception:
        pass
    print_solution_summary(sim, results)
