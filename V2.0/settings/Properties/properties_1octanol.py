'''
Properties of 1-octanol in 100mM NaCl water
from own measurement at AVT.FVT
Song Zhai
2024-09-01
'''


# property data
RHO_O = 825.3 # density of organic phase kg/m3
RHO_W = 1000 # density of water phase kg/m3
ETA_O = 6.043e-3 # viscosity of organic phase Pa*s
ETA_W = 0.82869e-3 # viscosity of water phase Pa*s
DELTA_RHO = abs(RHO_W-RHO_O) # density difference in kg/m3
R_IG = 8.314 # ideal gas constant J/mol*K
RHO_G = 1.2 # density of gas kg/m3
M_G = 28.97e-3 # molar mass of gas kg/mol
SIGMA = 0.00822 # surface tension N/m
R_V = 0.0215 # asymetric film drainage parameter