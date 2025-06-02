'''
Testing
'''

import separator_model as sm
import settings.geometry_parameter as gp
import settings.model_parameter as mp
import settings.physical_properties as pp
import settings.numerical_parameter as np

import settings.operating_conditions as op
import settings.initial_conditions as ic


def run_mainscript(input_tuple):
    # unpack input tuple
    operating_point, initial_condition, r_v, name_of_simulation, do_plot, t_end = input_tuple
    # load model with initial values from settings/*.py
    model = sm.SeparatorModel(pp, gp, np, mp, op, ic)
    # update settings with input values
    model.update_operating_conditions(operating_point)
    model.update_initial_conditions(initial_condition)
    model.update_name_of_simulation(name_of_simulation)
    model.update_plot_settings(do_plot)
    model.update_t_end(t_end)
    model.update_r_v(r_v)

    # solve ODE
    model.solve_ODE()