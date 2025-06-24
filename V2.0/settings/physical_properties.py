'''
physical properties of the system - n-Butyl acetate and water Henschke 1995
'''


# parameter definition
g = 9.81 # gravity constant m/s^2
Ha = 1e-20 # Hamaker constant J


def set_physical_properties(exp="ye"):
    global rho_o, rho_w, eta_o, eta_w, delta_rho, sigma, r_v
    if (exp == "ye"):
        rho_o = 819.72
        rho_w = 998.29
        eta_o = 9.73e-3
        eta_w = 1.012e-3
        delta_rho = abs(rho_w-rho_o)
        sigma = 0.036
        r_v = 0.01224
    elif(exp == "niba1"):
        rho_o = 832.5
        rho_w = 1002.7
        eta_o = 8.73e-3
        eta_w = 1.02e-3
        delta_rho = abs(rho_w-rho_o)
        sigma = 0.00822
        r_v = 0.05073
    elif(exp == "niba2"):
        rho_o = 825.4
        rho_w = 1000
        eta_o = 6.04e-3
        eta_w = 0.82e-3
        delta_rho = abs(rho_w-rho_o)
        sigma = 0.00822
        r_v = 0.03329
    elif(exp == "niba3"):
        rho_o = 818.2
        rho_w = 996.5
        eta_o = 4.363e-3
        eta_w = 0.68e-3
        delta_rho = abs(rho_w-rho_o)
        sigma = 0.00822
        r_v = 0.02182
    elif(exp == "niba4"):
        rho_o = 810.9
        rho_w = 992.3
        eta_o = 3.23e-3
        eta_w = 0.56e-3
        delta_rho = abs(rho_w-rho_o)
        sigma = 0.00822
        r_v = 0.02014
    else:
        rho_o = 883 # density of organic phase kg/m3
        rho_w = 998 # density of water phase kg/m3
        eta_o = 0.775e-3 # viscosity of organic phase Pa*s
        eta_w = 1.012e-3 # viscosity of water phase Pa*s
        delta_rho = abs(rho_w-rho_o) # density difference in kg/m3
        sigma = 0.013 # surface tension N/m
        r_v = 0.0383 # asymetric film drainage parameter
        print("Physical properties from n-Butyl acetate and water Henschke 1995 taken.")