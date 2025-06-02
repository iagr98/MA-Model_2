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

# load model with initial values from settings/*.py
model = sm.SeparatorModel(pp, gp, np, mp, op, ic)

def run_mainscript(input_tuple):
    # unpack input tuple
    pp, op_new, ic, r_v, name_of_simulation, t_end = input_tuple

    # update settings with input values
    model.update_physical_properties(pp)
    model.update_operating_conditions(op_new)
    model.update_initial_conditions(ic)
    model.update_r_v(r_v)
    model.update_plot_settings(False)
    model.update_name_of_simulation(name_of_simulation)
    model.update_t_end(t_end)

    # solve ODE
    model.solve_ODE()