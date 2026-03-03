"""
Module providing commonly used functions
"""

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
    tBreak = [288.15,  216.65, 216.65, 228.65, 270.65, 270.65,  214.65, 186.95, 186.95, 246.95]

    if h_m < 0 or h_m > 105000:
        raise ValueError("Altitude must be [0, 105000] meters.")

    g0 = 9.80665
    R = 287.05287
    T0 = 288.15
    p0 = 101325.0

    for i in range(9):
        if h_m < hBreak[i+1]:
            L = LBreak[i]
            T = tBreak[i] + L * (h_m - hBreak[i])
            break

    p = p0 * (T / T0) ** (-g0 / (L * R))

    return float(p / (R * T))

    g0 = 9.80665
    # R = 287.05287
    # T0 = 288.15
    # p0 = 101325.0
    # L = -0.0065
    # h_trop = 11000.0
    # 
    # if h_m <= h_trop:
    #     T = T0 + L * h_m
    #     p = p0 * (T / T0) ** (-g0 / (L * R))
    # else:
    #     T_trop = T0 + L * h_trop
    #     p_trop = p0 * (T_trop / T0) ** (-g0 / (L * R))
    #     T = T_trop
    #     p = p_trop * np.exp(-g0 * (h_m - h_trop) / (R * T))
    # 
    # return float(p / (R * T))
