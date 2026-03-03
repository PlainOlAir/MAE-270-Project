import numpy as np
import csdl_alpha as csdl
import time

from csdl_alpha.experimental import PySimulator
from modopt import CSDLAlphaProblem, TrustConstr, SLSQP

# ---------- ISA density ----------
def isa_density(h_m: float) -> float:
    g0 = 9.80665
    R = 287.05287
    T0 = 288.15
    p0 = 101325.0
    L = -0.0065
    h_trop = 11000.0

    if h_m <= h_trop:
        T = T0 + L * h_m
        p = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        T_trop = T0 + L * h_trop
        p_trop = p0 * (T_trop / T0) ** (-g0 / (L * R))
        T = T_trop
        p = p_trop * np.exp(-g0 * (h_m - h_trop) / (R * T))

    return float(p / (R * T))

# ---------- A) Parameters ----------
def get_problem_params():
    h = 12000.0
    return dict(
        rho=isa_density(h),
        g=9.81,
        R_loiter=5000.0,
        CD0=0.03,
        e=0.80,
        CL_max=1.4,
        stall_SF=1.2,
        eta_prop=0.80,
        f_day=0.50,
        P_area=250.0,           # https://www.airbusdefenceandspacenetherlands.nl/dsstuff/uploads/2019/11/SWdatasheet062020web.pdf
        eta_mppt=0.98,
        eta_batt=0.95,
        E_spec=250.0 * 3600.0,  # J/kg https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/#3.4
        m_payload=30.0,
        sigma_panel=4,          # https://www.spectrolab.com/DataSheets/Panel/panels.pdf
        a_struct=1.2,
        b_struct=0.04,
        WS_max=250.0,
        b_max=40.0,
        P_max=12000.0,
    )

# ---------- B) Geometry ----------
def add_geometry(S, AR, sqrt_fn=csdl.sqrt):
    b = sqrt_fn(AR * S)         # span
    return b

# ---------- C) Mass model ----------
def add_mass_model(S, AR, m_batt, f_panel, params):
    g = params["g"]
    m_payload = params["m_payload"]
    sigma_panel = params["sigma_panel"]
    a_struct = params["a_struct"]
    b_struct = params["b_struct"]

    A_panel = f_panel * S
    m_solar = sigma_panel * A_panel
    m_struct = a_struct * S + b_struct * (AR * S)

    m_total = m_payload + m_batt + m_solar + m_struct
    W = m_total * g
    return W, m_total, A_panel

# ---------- D) Aero + loiter ----------
def add_aero_loiter(S, AR, V, W, params, sqrt_fn=csdl.sqrt):
    rho = params["rho"]
    g = params["g"]
    R_loiter = params["R_loiter"]
    CD0 = params["CD0"]
    e = params["e"]

    q = 0.5 * rho * V**2        # Dynamic Pressure 
    n = sqrt_fn(1.0 + (V**2 / (g * R_loiter))**2)     # Load Factor
    L_req = n * W       # Lift required
    CL = L_req / (q * S)        # Lift coefficient

    # Drag
    k = 1.0 / (np.pi * e * AR)
    CD = CD0 + k * CL**2
    D = q * S * CD
    return CL, CD, D, n

# ---------- E) Power + energy + endurance ----------
def add_power_energy_endurance(S, V, m_batt, A_panel, D, params):
    eta_prop = params["eta_prop"]
    P_area = params["P_area"]
    eta_mppt = params["eta_mppt"]
    f_day = params["f_day"]
    eta_batt = params["eta_batt"]
    E_spec = params["E_spec"]

    P_req = (D * V) / eta_prop
    P_solar = P_area * eta_mppt * A_panel

    E_avail = eta_batt * E_spec * m_batt

    # day/night average battery draw (can be negative during day to represent charging)
    P_day = P_req - P_solar
    P_batt_avg = f_day * P_day + (1.0 - f_day) * P_req

    T_endurance = E_avail / P_batt_avg
    return P_req, P_solar, T_endurance

# ---------- F) Constraints + objective ----------
def add_constraints_objective(S, b, W, CL, P_req, T_endurance, params):
    CL_max = params["CL_max"]
    stall_SF = params["stall_SF"]
    WS_max = params["WS_max"]
    b_max = params["b_max"]
    P_max = params["P_max"]

    # objective
    (-T_endurance).set_as_objective()

    # constraints (all <= 0 form)
    (CL - (CL_max / stall_SF)).set_as_constraint(upper=0.0)
    ((W / S) - WS_max).set_as_constraint(upper=0.0)
    (b - b_max).set_as_constraint(upper=0.0)
    (P_req - P_max).set_as_constraint(upper=0.0)

# ---------- 1) CSDL builder ----------
def build_recorder(use_solar=True):
    params = get_problem_params()

    rec = csdl.Recorder()
    rec.start()

    # design variables
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

    # build model via functions
    b = add_geometry(S, AR)
    W, m_total, A_panel = add_mass_model(S, AR, m_batt, f_panel, params)
    CL, CD, D, n = add_aero_loiter(S, AR, V, W, params)
    P_req, P_solar, T_endurance = add_power_energy_endurance(S, V, m_batt, A_panel, D, params)

    # optional names (useful for debugging/plots later)
    b.add_name("wingspan")
    W.add_name("weight")
    CL.add_name("CL_loiter")
    P_req.add_name("P_req")
    P_solar.add_name("P_solar")
    T_endurance.add_name("T_endurance_s")

    add_constraints_objective(S, b, W, CL, P_req, T_endurance, params)

    rec.stop()
    return rec

# ---------- 2) Create Simulator ----------
def make_problem(problem_name="solar_uav"):
    rec = build_recorder(use_solar=True)
    sim = PySimulator(rec)
    return CSDLAlphaProblem(problem_name=problem_name, simulator=sim), sim

# ---------- 3) Solving Problem ----------
def run_opt(problem):
    optimizer = TrustConstr(problem, solver_options={"maxiter": 800, "gtol": 1e-8})
    # optimizer = SLSQP(problem, solver_options={"maxiter": 100, "ftol": 1e-7})

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
    P_req, P_solar, T_endurance_s = add_power_energy_endurance(S, V, m_batt, A_panel, D, params)

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
