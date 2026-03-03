"""
Module providing commonly used functions
"""

import numpy as np

def isa_density(h_m: float) -> float:
    """Returns density of air (kg/m^3) in accordance to the International Standard Atmosphere (ISA)

    Args:
        h_m (float): Altitude above sea-level in [0, 105000] meters

    Returns:
        out (float): Density of air in kg/m^3

    Raises:
        ValueError: If altitude is outside of allowable range
    """

    hBreak = [0,       11000,  20000,  32000,  47000,  51000,   71000,  84852,  90000,  105000]
    LBreak = [-0.0065, 0,      0.001,  0.0028, 0,      -0.0028, -0.002, 0,      0.004,  0]
    TBreak = [288.15,  216.65, 216.65, 228.65, 270.65, 270.65,  214.65, 186.95, 186.95, 246.95]

    if h_m < 0 or h_m > 105000:
        raise ValueError("Altitude must be [0, 105000] meters.")

    g0 = 9.80665
    R = 287.05287
    T0 = 288.15
    p0 = 101325.0

    T = T0
    p = p0

    for i in range(9):

        if h_m < hBreak[i+1]:
            T = TBreak[i] + LBreak[i] * (h_m - hBreak[i])

            if LBreak[i] == 0:
                p = p * np.exp(-g0 * (h_m - hBreak[i]) / (R * TBreak[i]))
            else:
                p = p * (T / TBreak[i]) ** (-g0 / (LBreak[i] * R))

            break

        else:
            if LBreak[i] == 0:
                p = p * np.exp(-g0 * (hBreak[i+1] - hBreak[i]) / (R * TBreak[i]))
            else:
                p = p * (TBreak[i+1] / TBreak[i]) ** (-g0 / (LBreak[i] * R))
        

    return float(p / (R * T))
