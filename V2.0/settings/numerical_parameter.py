'''
This file contains the numerical parameters for the simulation. The parameters are:
eps: epsilon for numerical stabilit
N_s: number of segments
N_d: number of droplet classes
t_end time of the simulation s
'''

eps = 1e-12

def set_numerical_parameter(N_s_i=200, N_d_i=50, t_end_i=200):
    global N_s, N_d, t_end
    N_s = N_s_i
    N_d = N_d_i
    t_end = t_end_i


