'''
wrapper function to calculate flooding volume flow rates for a given set of operating points
'''

import numpy as np
import pandas as pd
import joblib

import test_script_multiprocess_scan as tsm

N_CPU = 16

y_0 = [0.10, 0.20, 0.13] # initial conditions for h_w, h_l, h_dp 
t_end = 500 # s

sigma = 8.22e-3 # N/m
# list of input parameters
d32 = [905,950,738,865,708,635,723,673, 700,880,633,608,628,585,668,630] # um
Q_in_min = [500] # l/h
Q_in_max = [2000]
dQ_in = [100]
phase_frac = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5, 0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
T = [20,20,30,30,40,40,50,50, 20,20,30,30,40,40,50,50,] # C
# load physical properties from csv
physical_properties = pd.read_csv('physical_prop_t-dependent_1octanol.csv')
r_v = [0.06276,0.06276,0.03143,0.03143,0.0176,0.0176,0.01612,0.01612, 0.06276,0.06276,0.03143,0.03143,0.0176,0.0176,0.01612,0.01612] # endoscope
# h_dp_flood = [0.115,0.115,0.131,0.125,0.136,0.136,0.134,0.137, 0.134,0.134,0.132,0.129,0.136,0.134,0.132,0.131]
# h_dp_flood = [0.133 for i in range(len(r_v))] # m

# prepare iterable for multiprocessing
it = []
# iterate over all Q_ins 
for i in range(len(d32)):
    for Q_in in range(Q_in_min[0], Q_in_max[0]+dQ_in[0], dQ_in[0]):
        q_w_out = Q_in*(1-phase_frac[i])/1000/3600
        q_o_out = Q_in*(phase_frac[i])/1000/3600
        rho_w = np.interp(T[i], physical_properties['T'], physical_properties['rho_w'])
        eta_w = np.interp(T[i], physical_properties['T'], physical_properties['eta_w'])
        rho_o = np.interp(T[i], physical_properties['T'], physical_properties['rho_o'])
        eta_o = np.interp(T[i], physical_properties['T'], physical_properties['eta_o'])
        name = str(Q_in) + '_Qin_L_h_' + str(T[i]) + 'C' + '_' + str(phase_frac[i]) + 'phasefrac' + '_' + str(d32[i]) + 'um'
        op = (Q_in/1000/3600, phase_frac[i], d32[i]/1e6, q_w_out, q_o_out)
        init = (y_0[0], y_0[1], y_0[2])
        pp = (rho_w, eta_w, rho_o, eta_o, sigma)
        it.append((pp, op, init, r_v[i], name, t_end))
    
# safe it as Dataframe and export as csv
df = pd.DataFrame(it, columns=['PhysicalProperties', 'OperatingConditions', 'InitialConditions', 'r_v', 'name', 't_end'])
df.to_csv('2024_11_27_input_T-dependentflooding_endoscope.csv')

# run main_script for each operating point
joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(tsm.run_mainscript)(i) for i in it)
