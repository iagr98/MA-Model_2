'''
Testing model 2
'''

import separator_model as sm
import settings.geometry_parameter as gp
import settings.model_parameter as mp
import settings.physical_properties as pp
import settings.numerical_parameter as np
import settings.operating_conditions as op
import settings.initial_conditions as ic

def init_sim(exp, phi_0, dV_ges, eps_0, N_s, N_d):
    gp.set_geometry_parameter(exp)
    pp.set_physical_properties(exp)
    np.set_numerical_parameter(N_s, N_d)
    op.set_operating_conditions(phi_0, dV_ges, eps_0)
    ic.set_initial_conditions(exp)
    return sm.SeparatorModel(pp, gp, np, mp, op, ic)

def run_sim(exp="ye", phi_0=610e-6, dV_ges=240, eps_0=0.2, N_s=200, N_d=50):
    Sim = init_sim(exp, phi_0, dV_ges, eps_0, N_s, N_d)
    Sim.solve_ODE(h_w_const=True, h_l_const=True, print_logfile=True, report=True, calc_algebraic=True)
    return Sim

if __name__ == "__main__":

    exp = "ye"
    phi_0 = 610e-6
    dV_ges = 240
    eps_0 = 0.2

    Sim = run_sim(exp, phi_0, dV_ges, eps_0)
