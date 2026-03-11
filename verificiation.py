import problem as pb
import numpy as np
import funlib

b = pb.add_geometry(60,10,np.sqrt)
# W, m_total, A_panel = pb.add_mass_model(S, AR, m_batt, f_panel, params)
# CL, CD, D, n = pb.add_aero_loiter(S, AR, V, W, params)
# P_req, P_solar, P_batt_avg_cycle, T_endurance = pb.add_power_energy_endurance(S, V, m_batt, A_panel, D, params)

print(b)
print(funlib.isa_density(8000))
print(funlib.isa_density_equations(8000))