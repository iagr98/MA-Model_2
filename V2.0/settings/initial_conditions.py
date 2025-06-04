'''
Initial conditions for the simulation
h_w: initial height of the water phase in the separator m
h_dp: initial height of the dense-packed zone m
h_l: initial height of the light phase m
'''

def set_initial_conditions(exp="ye"):
    global h_w, h_dp, h_l
    if (exp=="ye"):
        h_w = 0.055
        h_dp = 0.095
        h_l = 0.15
    elif((exp == "niba1") or (exp == "niba2") or (exp == "niba2") or (exp == "niba4")):
        h_w = 0.1
        h_dp = 0.13
        h_l = 0.2
    else:
        h_w = 0.1
        h_dp = 0.2
        h_l = 0.3
        print("Test does not belong to either niba or ye. h_w=0.1m, h_dp=0.2m and h_l=0.3m")
