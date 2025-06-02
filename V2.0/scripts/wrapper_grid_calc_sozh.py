'''
wrapper function to calculate different operating points for a given grid
'''

import numpy as np
import pandas as pd

import joblib

import test_script_multiprocess as tsm

N_CPU = 20

# pool = Pool(N_CPU)

y_0 = [0.10, 0.20, 0.12] # initial conditions for h_w, h_l, h_dp 
t_end = 500 # s

# create grid of operating points
V_in_min = 0.5/3600
V_in_max = 1.5/3600
n_V_in = 3
phase_frac_in_min = 0.5
phase_frac_in_max = 0.5
n_phase_frac_in = 1
Sauter_in_min = 0.734e-3
Sauter_in_max = 0.734e-3
n_sauter = 1
eta_w_min = 0.001
eta_w_max = 0.030
n_eta_w = 5
rho_w_min = 1000
rho_w_max = 1100
n_rho_w = 3

r_v = 0.0383

# create grid
X1 = np.linspace(V_in_min, V_in_max, n_V_in)
# X2 = np.linspace(phase_frac_in_min, phase_frac_in_max, n_phase_frac_in)
X2 = np.linspace(eta_w_min, eta_w_max, n_eta_w)
# X3 = np.linspace(Sauter_in_min, Sauter_in_max, n_sauter)
X3 = np.linspace(rho_w_min, rho_w_max, n_rho_w)

# prepare iterable for multiprocessing
it = []
for i in range(n_V_in):
    for j in range(n_eta_w):
        for k in range(n_rho_w):
            q_w_out = X1[i]*(1-phase_frac_in_min)
            q_o_out = X1[i]*(phase_frac_in_min)
            name = 'Exp_Sim_Vin_' + str(X1[i]*3600) + 'm3h-1_eta_w_' + str(X2[j]) + '_Pas_rho_w_' + str(X3[k]) + 'kg_m-3' 
            op = (X1[i], phase_frac_in_min, Sauter_in_min, q_w_out, q_o_out)
            init = (y_0[0], y_0[1], y_0[2])
            physical_prop_aq = (X2[j], X3[k])
            it.append((op, init, r_v, name, True, t_end, physical_prop_aq))
            
# safe it as Dataframe and export as csv
df = pd.DataFrame(it, columns=['OperatingConditions', 'InitialConditions', 'r_v', 'name', 'safe_plot', 't_end', 'physical_prop_aq'])
df.to_csv('2024_06_12_input_gridcalc.csv')

# run main_script for each operating point
joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(tsm.run_mainscript)(i) for i in it)
