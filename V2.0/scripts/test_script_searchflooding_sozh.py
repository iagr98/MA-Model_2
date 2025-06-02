'''
Testing to search for flooding point (total volume flow rate)
'''

import separator_model as sm
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import minimize
import pandas as pd

import settings.geometry_parameter as gp_init
import settings.model_parameter as mp_init
import settings.numerical_parameter as np_init
import settings.operating_conditions as op_init
import settings.initial_conditions as ic_init
import settings.physical_properties as pp_init



def calc_flooding_point(input_tuple):
    pp, op_new, ic, r_v, name_of_simulation, t_end, h_dp_flood = input_tuple
    # load model with initial values from settings/*.py
    model = sm.SeparatorModel(pp_init, gp_init, np_init, mp_init, op_init, ic_init)
    # initialize model
    model.update_physical_properties(pp)
    model.update_operating_conditions(op_new)
    model.update_initial_conditions(ic)
    model.update_r_v(r_v)
    model.update_plot_settings(False)
    model.update_name_of_simulation(name_of_simulation)
    model.update_t_end(t_end)
    
    if 1==1:
        model.find_flooding_flowrate(h_dp_flood)
        total_volume_flow_flood = model.Q_in_flood
    
    if 1==0:
        def fun_calc_height(total_volume_flow, h_flood):
            """Calculate the heights of the separator for a given total volume flow rate.

            Args:
                total_volume_flow float: Total volume flow rate entering the separator
                h_flood float: Height of the DPZ + water of the separator at the flooding point
            Returns:
                h list: List of heights of the separator
            """
            # get current operating point
            operating_point = model.get_operating_conditions()
            # update total volume flow rate
            lst = list(operating_point)
            lst[0] = total_volume_flow
            lst[-2] = total_volume_flow*(1-operating_point[1])
            lst[-1] = total_volume_flow*operating_point[1]
            operating_point_new = tuple(lst)
            model.update_operating_conditions(operating_point_new)
            # solve ODE
            model.solve_ODE(h_w_const=True, report=False, print_logfile=False,calc_algebraic=False)
            # get heights
            h_dp = model.h_dp
            res = h_dp - h_flood
            return res**2
        
        # total_volume_flow_flood = fsolve(fun_calc_height, total_volume_flow_0, args=(h_dp_flood), maxfev=20, xtol=1e-4)
        # total_volume_flow_flood = brentq(fun_calc_height, 0.4/3600, 2.5/3600, args=(h_dp_flood), xtol=1e-4, maxiter=10)
        total_volume_flow_flood = minimize(fun_calc_height, 1.5/3600, args=(h_dp_flood), method='Nelder-Mead', tol=1e-6, options={'maxiter': 2, 'disp': True})
    
    
    # save total_volume_flow_flood as csv file in subfolder results\name_of_simulation
    pd.DataFrame([total_volume_flow_flood]).to_csv('results/' + name_of_simulation + '/total_volume_flow_flood.csv')
    
    return total_volume_flow_flood
