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

model = sm.SeparatorModel(pp, gp, np, mp, op, ic)
# model.Q_in = 0.5/3600
# model.d32_in = 368e-6
# model.h_w_initial = 0.055
# model.h_l_initial = 0.15
# model.h_dp_initial = 0.0726
model.solve_ODE(h_w_const=True, calc_algebraic=True)
import numpy as np
efficiency = 1 - np.sum(model.n_ivan[-1,-1,:]) / np.sum(model.n_in)
print('Extraction efficiency at outlet= ',efficiency)