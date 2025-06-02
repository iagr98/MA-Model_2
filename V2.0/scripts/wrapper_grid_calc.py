'''
wrapper function to calculate different operating points for a given grid
'''

import numpy as np
import pandas as pd
import joblib

import test_script_multiprocess as tsm

N_CPU = 3

# pool = Pool(N_CPU)

y_0 = [0.10, 0.20, 0.12] # initial conditions for h_w, h_l, h_dp 
t_end = 10 # s

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

r_v = 0.0383

# create grid
X1 = np.linspace(V_in_min, V_in_max, n_V_in)
X2 = np.linspace(phase_frac_in_min, phase_frac_in_max, n_phase_frac_in)
X3 = np.linspace(Sauter_in_min, Sauter_in_max, n_sauter)

# prepare iterable for multiprocessing pool.starmap
it = []
for i in range(n_V_in):
    for j in range(n_phase_frac_in):
        for k in range(n_sauter):
            q_w_out = X1[i]*(1-X2[j])
            q_o_out = X1[i]*(X2[j])
            name = 'Exp_Sim_Vin_' + str(X1[i]*3600) + 'm3h-1_phasefrac_' + str(X2[j]) + '_Sauterin_' + str(X3[k]*1e6) + 'um'
            op = (X1[i], X2[j], X3[k], q_w_out, q_o_out)
            init = (y_0[0], y_0[1], y_0[2])
            it.append((op, init, r_v, name, True, t_end))
            
# safe it as Dataframe and export as csv
df = pd.DataFrame(it, columns=['OperatingConditions', 'InitialConditions', 'r_v', 'name', 'safe_plot', 't_end'])
df.to_csv('2024_05_31_input_gridcalc.csv')

# run main_script for each operating point
joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(tsm.run_mainscript)(i) for i in it)
