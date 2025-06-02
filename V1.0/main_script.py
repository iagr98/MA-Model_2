'''
 Model of Backi et al. 2018
 Modifications:
 2 phase: l-l
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

import constants
import fun
import plot_fun

import sys
from datetime import datetime
old_stdout = sys.stdout

## define simulation settings
# name of this simulation to find logfile and plots
name_of_simulation = 'DN200_L1.8m_Vin_4.0_phase_fraction_0.33_SauterIn_0.5'
do_plot = True
try:
    os.mkdir('results')
except:
    pass
try:
    os.mkdir('results_csv')
except:
    pass
# track computation time
start_time = datetime.now()
# path with current date and time
path = datetime.now().strftime('results' + str("\\") + '%Y_%m_%d_%H_%M_' + name_of_simulation)
suffix = path + '%Y_%m_%d_%H_%M_' + name_of_simulation
# create folder where plots and logfiles are saved 
try:
    os.mkdir(path)
except:
    pass

# decide if logfile should be printed
print_logfile = True

if print_logfile:
    # log file with current date and time from subfolder
    log_file = open(path + '\logfile.log',"w")
    sys.stdout = log_file
    
# initializing simulation
print('----------------------------------------')  
print('Separator model calculation')
print('----------------------------------------')  

# simulation settings
t_end = 100 # end time in s
time_span = [0, t_end] # time span in s

# initial conditions
y_0 = np.zeros(3) # initial conditions
y_0[0] = 0.10 # initial water height in m
y_0[2] = 0.12 # initial dense-packed zone height in m
y_0[1] = 0.20 # initial liquid height in m
# check if initial conditions are physical
if y_0[1] > 2*constants.R:
    raise ValueError('Initial liquid height is larger than Separator diameter!')
if y_0[2] > y_0[1] or y_0[2] < y_0[0]:
    raise ValueError('Initial dense-packed zone height is larger than liquid height or smaller than water height!')
if y_0[0] < 0:
    raise ValueError('Initial water height is negative!')
# operational conditions
Q_IN = 4.0/3600 # total flow rate inlet m3/s
EPSILON_IN = 0.33 # phase fraction of organic phase -
# boundary conditions / inlet conditions
d_32 = 0.3e-3 # diameter of a droplet m at separator inlet
d_max = 2.5*d_32 # maximum diameter of droplets in m
# initialize boundary conditions
hold_up_calc, n_in, d_bins, N_in_total = fun.initialize_boundary_conditions(EPSILON_IN, d_32, d_max, path, plot=do_plot) 
# inlet flow rates of aq. and org. phases
q_w_in = Q_IN
q_w_out = Q_IN*(1-hold_up_calc)
q_o_out = Q_IN*(hold_up_calc)

# print constants with units
print('Constants: number of segments in separator' , constants.N_S, '-')
print('Constants: number of droplet classes' , constants.N_D, '-')
print('Constants: total flow rate inlet' , Q_IN*3600, 'm3/h')
print('Constants: phase fraction at inlet' , EPSILON_IN, '-')
print('Constants: radius of separator' , constants.R, 'm')
print('Constants: length of separator' , constants.L, 'm')
print('Constants: density of organic phase' , constants.RHO_O, 'kg/m3')
print('Constants: density of water phase' , constants.RHO_W, 'kg/m3')
print('Constants: viscosity of organic phase' , constants.ETA_O, 'Pa*s')
print('Constants: viscosity of water phase' , constants.ETA_W, 'Pa*s')
print('Constants: density difference' , constants.DELTA_RHO, 'kg/m3')
print('Constants: interfacial tension' , constants.SIGMA, 'N/m')
print('Constants: asymetric film drainage parameter' , constants.R_V, '-')
print('----------------------------------------')
# print initial conditions y_0
print('Initial conditions water height: ', y_0[0], 'm')
print('Initial conditions liquid height: ', y_0[1], 'm')
print('Initial conditions dense-packed zone: ', y_0[2], 'm')
print('----------------------------------------')
# print boundary conditions with units
print('Boundary conditions sauter mean diameter: ', d_32*1e6, 'um')
print('Boundary conditions maximum diameter: ', d_max*1e6, 'um')
# print boundary conditions with units
print('Boundary conditions total flow rate liquid inlet: ', (q_w_in)*3600, 'm3/h')
print('Boundary conditions aq. phase flow rate inlet: ', q_w_in*3600, 'm3/h')
print('Boundary conditions aq. phase flow rate outlet: ', q_w_out*3600, 'm3/h')
print('Boundary conditions org. phase flow rate outlet: ', q_o_out*3600, 'm3/h')
print('Boundary conditions holdup in aq. phase: ', hold_up_calc, '-')
print('Boundary conditions droplet classes: ', d_bins*1e6, 'um')
print('Boundary conditions total number of droplets: ', N_in_total, '-')
print('Boundary conditions number of droplets: ', n_in, '-')

## generate inputs for solve_ivp
u = (q_w_in, q_w_out, q_o_out, n_in, EPSILON_IN)
p = (d_bins,)
inputs = (u, p)
solver_options = {'rtol': 1e-12, 'atol': 1e-12}
# define events representing unphysical states
def flooded_densepackedzone(t, y, u, p):
    import constants
    # flooded separator with dense-packed phase
    return y[2] + 1e-3 - y[1]
flooded_densepackedzone.terminal = True
def flooded_separator(t, y, u, p):
    import constants
    # flooded separator with liquid phase
    return y[1] - 2*constants.R - 1e-3
flooded_separator.terminal = True
def empty_aq_phase(t, y, u, p):
    import constants
    # empty separator with aq. phase
    return y[0] - 1e-3
empty_aq_phase.terminal = True

events_physics = (flooded_densepackedzone, flooded_separator, empty_aq_phase)

print('Solving ivp...')
sol = solve_ivp(fun.dy_dt_RHS, time_span, y_0, args=inputs, dense_output=True, \
    method='RK45', events=events_physics, options=solver_options, t_eval=np.linspace(0, t_end, 10*t_end+1))
print(sol)

# plot solution of solve_ivp up to first occuring event
if sol.status == 0:
    t_range = np.linspace(0, sol.t[-1], int(10*sol.t[-1]+1))
if sol.status == 1: # evenet occured  
    t_range = np.linspace(0, sol.t_events[0][0], int(10*sol.t_events[0][0]+1))
    t_end = sol.t_events[0][0]
if sol.status == -1: # integration failed and terminate script
    print('Integration failed!')
    sys.exit()
    
z = sol.sol(t_range)
if do_plot:
    fig, ax = plt.subplots()
    ax.plot(t_range, z.T[:,0]*1e3, label='Water')   
    ax.plot(t_range, z.T[:,1]*1e3, label='Liquid')
    ax.plot(t_range, z.T[:,2]*1e3, label='Dense-packed zone')
    ax.set_ylim([0,2.1*constants.R*1e3])
    ax.set_xlim([0, t_end])
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Liquid heights / mm')
    ax.set_title('Two-phase l-l separator DN' + str(2*constants.R*1e3) + ', L=' + str(constants.L) + 'm')
    ax.legend()
    # save plot
    fig.savefig(path + '\liquid_heights.png', dpi=1000)
    fig.savefig(path + '\liquid_heights.eps', dpi=1000)
    fig.savefig(path + '\liquid_heights.svg', dpi=1000)

# outputs
print(sol.t_events)

print('----------------------------------------')
print('Start calculation of algebraic variables')
print('----------------------------------------')

# dimension of time vector
N_t = len(t_range)

# declaration of algebraic variables
dy_dt = np.zeros((N_t, 3))
# coalescence rate
dV_s = np.zeros((N_t, constants.N_S, constants.N_D))
dV_si = np.zeros((N_t, constants.N_S))
dV_s_tot = np.zeros(N_t)
epsilon = np.zeros((N_t, constants.N_S + 1))
dV_c = np.zeros((N_t, constants.N_S))
d32_dp = np.zeros((N_t, constants.N_S))
d32_out = np.zeros(N_t)
dV_w_dp = np.zeros((N_t, constants.N_S))
tau = np.zeros((N_t, constants.N_S))
q_w = np.zeros((N_t, constants.N_S+1))
q_dp = np.zeros((N_t, constants.N_S+1))
# declaration of variables entering a segment and for each class
n = np.zeros((N_t, constants.N_S + 1, constants.N_D))
pos = np.zeros((N_t, constants.N_S + 1, constants.N_D))

# calculation of algebraic variables
# loop over time steps
for t in range(N_t):
    results = fun.calculate_separator(z.T[t,:], u, p)
    # allocate results
    # dy_dt[t,:] = results[0]
    # q_w[t,:] = results[1]
    epsilon[t,:] = results[2]
    q_dp[t,:] = results[4]
    n[t,:,:] = results[5]
    # pos[t,:,:] = results[6]
    d32_dp[t,:] = results[7]
    # tau[t,:] = results[8]
    dV_c[t,:] = results[9]
    dV_w_dp[t,:] = results[10]
    # dV_s[t,:,:] = results[11]
    dV_si[t,:] = results[12]
    dV_s_tot[t] = np.sum(dV_si[t,:])
    d32_out[t] = fun.get_sauter_mean_diameter(n[t,-1,:], d_bins)
dV_w_dp_tot = np.sum(dV_w_dp, axis=1)

L_dp = constants.L
# find length where q_dp is zero
for i in range(constants.N_S):
        if q_dp[-1,1+i] == 0:
            L_dp = i*constants.L/constants.N_S
            print('q_dp is zero at length: ', L_dp, 'm')
            break
# print height of dense-packed zone at end of simulation
print('Height of dense-packed zone at end of simulation: ', (z.T[-1,2]-z.T[-1,0])*1e3, 'mm')

## save results as dataframe and export 
df = pd.DataFrame(data=z.T, columns=['Water height', 'Liquid height', 'Dense-packed zone height']) # 
# add time vector at first column
df.insert(0, 'Time', t_range)
df['Coalescence rate'] = np.sum(dV_c, axis=1)
df['Water feed to dense-packed zone'] = dV_w_dp_tot
df['Hold up at outlet'] = epsilon[:,-1]
df['Sedimentation rate'] = dV_s_tot
df['Sauter mean diameter outlet'] = d32_out
# add control inputs
df['Aq flow rate inlet'] = q_w_in*np.ones(N_t)
df['Aq flow rate outlet'] = q_w_out*np.ones(N_t)
df['Organic flow rate outlet'] = q_o_out*np.ones(N_t)
df['Hold up at inlet'] = EPSILON_IN*np.ones(N_t)
df['Sauter mean diameter inlet'] = d_32*np.ones(N_t)
df.to_csv(path + '\\results.csv')
#save in separate folder for csv files
path_csv = datetime.now().strftime('results_csv' + str("\\") + '%Y_%m_%d_%H_%M_' + name_of_simulation)
df.to_csv(path_csv+'.csv')

## plotting
if do_plot:
    plot_fun.plot_holdup_outlet(t_range, epsilon, t_end, path)
    plot_fun.plot_number_distribution_outlet(t_range, n, d_bins, d_max, t_end, path)
    plot_fun.plot_flowrates(t_range, dV_c, dV_s_tot, dV_w_dp, t_end, path)
    plot_fun.plot_d32_over_time(t_range, d32_dp, d_max, t_end, path)
    plot_fun.plot_holdup_over_length(t_range, epsilon, t_end, path)
    plot_fun.plot_qdp(q_dp = q_dp, t_range = t_range, path = path)

# track computation time
end_time = datetime.now()
print('----------------------------------------')
print('Computation time: ', end_time - start_time)
print('----------------------------------------')

## save results: L_dp, height of dense-packed zone, sol.status, t_end as dataframe and export
df2 = pd.DataFrame() 
df2['L_dp'] = [L_dp]
df2['Height of dense-packed zone'] = [(z.T[-1,2]-z.T[-1,0])]
df2['sol.status'] = [sol.status]
df2['t_end'] = [t_end]
df2['CompTime'] = [end_time - start_time]
df2.to_csv(path + '\\results_secondary.csv')
df2.to_csv(path_csv+'_secondary.csv')

if print_logfile:
    # print log outputs
    sys.stdout = sys.__stdout__
    log_file.close()



