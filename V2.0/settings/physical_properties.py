'''
physical properties of the system - n-Butyl acetate and water Henschke 1995
'''

# parameter definition
g = 9.81 # gravity constant m/s^2
Ha = 1e-20 # Hamaker constant J
# property data 
rho_o = 883 # density of organic phase kg/m3
rho_w = 998 # density of water phase kg/m3
eta_o = 0.775e-3 # viscosity of organic phase Pa*s
eta_w = 1.012e-3 # viscosity of water phase Pa*s
delta_rho = abs(rho_w-rho_o) # density difference in kg/m3
sigma = 0.013 # surface tension N/m
r_v = 0.0383 # asymetric film drainage parameter
