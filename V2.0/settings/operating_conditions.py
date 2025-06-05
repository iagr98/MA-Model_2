'''
operating conditions for the simulation
'''

# genereal settings
name_of_simulation = 'test' # name of the simulation
do_plot = False # plot the results of the simulation

def set_operating_conditions(phi_0=1e-3, dV_ges=1000, eps_0=0.5):
    global Sauterdiameter_in, Q_in, phasefraction_in, Q_w_out, Q_o_out
    Sauterdiameter_in = phi_0       # Sauter diameter of the droplets in the inflow
    Q_in = dV_ges / 3.6 * 1e-6      # Inflow rate m^3/s
    phasefraction_in = eps_0        # Phase fraction of organic phase in the inflow
    Q_w_out = Q_in/2                # Outflow rate of the water phase m^3/s
    Q_o_out = Q_in/2                # Outflow rate of the organic phase m^3/s