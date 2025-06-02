'''
Wrapper for LHS sampling and executing test_script_multiprocess.py
'''

# import main_script
from scipy.stats import qmc
import joblib
import pandas as pd

import test_script_multiprocess as tsm


N_CPU = 20

t_end = 20 # s
safe_plot = False
Q_w_in = 3/3600 # m^3/s
phase_fraction = 0.4
r_v = 0.0383
R = 0.1 # radius of the separator in m

# define parameter ranges
d_32_min = 1.0e-3*0.9
d_32_max = 1.0e-3*1.1
Q_w_out_min = Q_w_in*(1-phase_fraction)*0.9
Q_w_out_max = Q_w_in*(1-phase_fraction)*1.1
h_w_init_min = 0.10*0.9
h_w_init_max = 0.10*1.1
h_dp_init_min = 0.12*0.9
h_dp_init_max = 0.12*1.1
r_v_min = r_v*0.9
r_v_max = r_v*1.1
Q_w_in_min = Q_w_in*0.9
Q_w_in_max = Q_w_in*1.1

l_bounds = [d_32_min, Q_w_out_min, h_w_init_min, h_dp_init_min, Q_w_in_min, r_v_min]
u_bounds = [d_32_max, Q_w_out_max, h_w_init_max, h_dp_init_max , Q_w_in_max, r_v_max]
n = int(0.2e3)
sampler = qmc.LatinHypercube(d=6) 
sample = sampler.random(n=n)
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
print(qmc.discrepancy(sample_scaled))

# prepare input list
it = []
for i in range(len(sample_scaled)):
    y0 = (sample_scaled[i,2], 2*R, sample_scaled[i,3])
    # constant liq. height
    # Q_o_out = Q_w_in - sample_scaled[i,1]
    Q_o_out = sample_scaled[i,4] - sample_scaled[i,1]
    name = 'DN200_L1.8m_No' + str(i)
    op = (sample_scaled[i,4], phase_fraction, sample_scaled[i,0], sample_scaled[i,1], Q_o_out)
    it.append((op, y0, sample_scaled[i,-1], name, True, t_end))

# safe it as Dataframe and export as csv
df = pd.DataFrame(it, columns=['OperatingConditions', 'InitialConditions', 'r_v', 'name', 'safe_plot', 't_end'])
df.to_csv('input_LHS.csv')

# run main_script for each operating point
joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(tsm.run_mainscript)(i) for i in it)