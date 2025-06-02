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

model.solve_ODE()