'''
This file contains the geometry parameters of the separator.
'''
# Geometry data

d_stirrer = 0.1 # diameter of stirrer m

def set_geometry_parameter(exp="ye"):
    global r, l
    if (exp == "ye"):
        r = 0.075 # radius of separator m
        l = 0.56 # length of separator m
    elif((exp == "niba1") or (exp == "niba2") or (exp == "niba2") or (exp == "niba4")):
        r = 0.1
        l = 1.0
    else:
        r = 0.15
        l = 1.0
        print("Test does not belong to either niba or ye. r=0.15m and l=1.0m selected")