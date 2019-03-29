#! /usr/bin/python3
import matplotlib

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np
import time
import os

def write_in_file(rad, RA_cen, Dec_cen, e, PA_c, bg_c, ng, Rcirc_c):
    f = open("mock.inp", "w")
    f.write("NFW\n" + str(rad) + "\n" + str(RA_cen) + " " + str(Dec_cen) + "\n" + str(e) + " " + str(PA_c) + "\n" + str(bg_c) + "\n" + str(ng) + "\n" + str(Rcirc_c))
    f.close()

log_radius = [-2.5, -2, -1.5, -1]
radius = [(10**i)*60 for i in log_radius]
RA = [0, 10, 20, 30]
Dec = [0, 20, 30, 40]
ell = [0, 0.4, 0.7, 0.95]
PA = [0, 10, 20, 30]
bg = [i/3600. for i in [0, 3000, 5000, 5000]]
Rcirc = 8

#ngal = [20, 40, 80, 160, 320, 640, 1280]
ngal = 1280
for i in [1]: #range(len(RA)):
    write_in_file(radius[i], RA[i], Dec[i], ell[i], PA[i], bg[i], ngal, Rcirc)
    os.system("mock.py < mock.inp")

#ell = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#for i in ell:
#    write_in_file(radius, RA, Dec, i, PA, bg, 1280, Rcirc)
#    os.system("mock.py < mock.inp")

#PA = np.arange(0, 180, 10)
#for i in PA:
#    write_in_file(radius, RA, Dec, 0, i, bg, 1280, Rcirc)
#    os.system("mock.py < mock.inp")

#radius = 10**np.arange(-2.2, 0, 0.2)
#for i in radius:
#    write_in_file(i, RA, Dec, 0, 0, bg, 1280, Rcirc)
#    os.system("mock.py < mock.inp")

#bg = np.arange(0, 1280, 20)
#for i in bg:
#    write_in_file(radius, RA, Dec, 0, 0, i, 1280, Rcirc)
#    os.system("mock.py < mock.inp")
