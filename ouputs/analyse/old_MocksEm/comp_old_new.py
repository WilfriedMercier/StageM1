import matplotlib

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np
import time

def plot_one_graph(datax, datay, error_y, xlabel, ylabel, plot_label, symbol, title):
    plt.errorbar(datax, datay, yerr=error_y, label=plot_label, marker=symbol, linestyle='None')
    plt.xlabel(xlabel, fontsize=size_text)
    plt.ylabel(ylabel, fontsize=size_text)
    plt.title(title, fontsize=size_text)

def plot_many_graphs(datax, datay, error_y, xlabels, ylabels, plot_labels, symbols, title, export):
    for i in range(len(xlabels)):
        plot_one_graph(datax[i], datay[i], error_y[i], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)

    plt.legend()
    plt.xscale('log')
    plt.grid(linestyle='dashed')

    dir_ext = "comp_noell_nobg/"
    #dir_ext = "comp_ell_bg/"
    plt.savefig(dir_ext + export)
    plt.show()

names = ["1280_bfgs_noellnobg.output", "160_de_noellnobg.output", "20_lbb_noellnobg.output", "320_nm_noellnobg.output", "40_tnc_noellnobg.output", "80_bfgs_noellnobg.output",
         "1280_de_noellnobg.output", "160_lbb_noellnobg.output", "20_nm_noellnobg.output", "320_tnc_noellnobg.output", "640_bfgs_noellnobg.output", "80_de_noellnobg.output",
         "1280_lbb_noellnobg.output", "160_nm_noellnobg.output", "20_tnc_noellnobg.output", "40_bfgs_noellnobg.output", "640_de_noellnobg.output", "80_lbb_noellnobg.output",
         "1280_nm_noellnobg.output", "160_tnc_noellnobg.output", "320_bfgs_noellnobg.output", "40_de_noellnobg.output", "640_lbb_noellnobg.output", "80_nm_noellnobg.output",
         "1280_tnc_noellnobg.output", "20_bfgs_noellnobg.output", "320_de_noellnobg.output", "40_lbb_noellnobg.output", "640_nm_noellnobg.output", "80_tnc_noellnobg.output",
         "160_bfgs_noellnobg.output", "20_de_noellnobg.output", "320_lbb_noellnobg.output", "40_nm_noellnobg.output", "640_tnc_noellnobg.output"]
names1 = names

#names = ["1280_bfgs_ellbg.output", "160_de_ellbg.output", "20_lbb_ellbg.output", "320_nm_ellbg.output", "40_tnc_ellbg.output", "80_bfgs_ellbg.output",
#         "1280_de_ellbg.output", "160_lbb_ellbg.output", "20_nm_ellbg.output", "320_tnc_ellbg.output", "640_bfgs_ellbg.output", "80_de_ellbg.output",
#         "1280_lbb_ellbg.output", "160_nm_ellbg.output", "20_tnc_ellbg.output", "40_bfgs_ellbg.output", "640_de_ellbg.output", "80_lbb_ellbg.output",
#         "1280_nm_ellbg.output", "160_tnc_ellbg.output", "320_bfgs_ellbg.output", "40_de_ellbg.output", "640_lbb_ellbg.output", "80_nm_ellbg.output",
#         "1280_tnc_ellbg.output", "20_bfgs_ellbg.output", "320_de_ellbg.output", "40_lbb_ellbg.output", "640_nm_ellbg.output", "80_tnc_ellbg.output",
#         "160_bfgs_ellbg.output", "20_de_ellbg.output", "320_lbb_ellbg.output", "40_nm_ellbg.output", "640_tnc_ellbg.output"]

#names1 = ["1280_bfgs_elliptical_analytical.output", "160_de_elliptical_analytical.output", "20_lbb_elliptical_analytical.output", "320_nm_elliptical_analytical.output", "40_tnc_elliptical_analytical.output", "80_bfgs_elliptical_analytical.output",
#         "1280_de_elliptical_analytical.output", "160_lbb_elliptical_analytical.output", "20_nm_elliptical_analytical.output", "320_tnc_elliptical_analytical.output", "640_bfgs_elliptical_analytical.output", "80_de_elliptical_analytical.output",
#         "1280_lbb_elliptical_analytical.output", "160_nm_elliptical_analytical.output", "20_tnc_elliptical_analytical.output", "40_bfgs_elliptical_analytical.output", "640_de_elliptical_analytical.output", "80_lbb_elliptical_analytical.output",
#         "1280_nm_elliptical_analytical.output", "160_tnc_elliptical_analytical.output", "320_bfgs_elliptical_analytical.output", "40_de_elliptical_analytical.output", "640_lbb_elliptical_analytical.output", "80_nm_elliptical_analytical.output",
#         "1280_tnc_elliptical_analytical.output", "20_bfgs_elliptical_analytical.output", "320_de_elliptical_analytical.output", "40_lbb_elliptical_analytical.output", "640_nm_elliptical_analytical.output", "80_tnc_elliptical_analytical.output",
#         "160_bfgs_elliptical_analytical.output", "20_de_elliptical_analytical.output", "320_lbb_elliptical_analytical.output", "40_nm_elliptical_analytical.output", "640_tnc_elliptical_analytical.output"]


nbgal = [1280, 160, 20, 320, 40, 80, 1280, 160, 20, 320, 640, 80, 1280, 160, 20, 40, 640, 80, 1280, 160, 320, 40, 640, 80, 1280, 20, 320, 40, 640, 80, 160, 20, 320, 40, 640]
meth  = ["bfgs", "de", "lbb", "nm", "tnc"]*7

#Exact values
true_log_scale = -2.08

size_text      = 14


dir_new = "../noell_nobg/"
dir_old = "../old//noell_nobg/"
#dir_new = "../ell_bg/"
#dir_old = "../old/ell_bg/"

#time
start = time.clock()

scale_old_medianguess_bias = []
scale_old_medianfound_bias = []
disp_scale_oldguess = []
disp_scale_old = []

scale_new_medianguess_bias = []
scale_new_medianfound_bias = []
disp_scale_newguess = []
disp_scale_new = []

galaxies = []

for j in range(1,len(meth)//7):
    for i in range(j, len(names), 5):
        log_scale_old_radius, lsr_old_guess = np.genfromtxt(dir_old + names1[i], usecols=(12, 22), unpack=True)
        N = len(log_scale_old_radius)

        scale_old_medianguess_bias.append(np.median(lsr_old_guess) - true_log_scale)
        scale_old_medianfound_bias.append(np.median(log_scale_old_radius) - true_log_scale)
        disp_scale_oldguess.append( np.abs(np.std(lsr_old_guess))/np.sqrt(2.*N) )
        disp_scale_old.append( np.abs(np.std(log_scale_old_radius))/np.sqrt(2.*N) )

        log_scale_new_radius, lsr_new_guess = np.genfromtxt(dir_new + names[i], usecols=(12, 22), unpack=True)
        N = len(log_scale_new_radius)

        scale_new_medianguess_bias.append(np.median(lsr_new_guess) - true_log_scale)
        scale_new_medianfound_bias.append(np.median(log_scale_new_radius) - true_log_scale)
        disp_scale_newguess.append( np.abs(np.std(lsr_new_guess))/np.sqrt(2.*N) )
        disp_scale_new.append( np.abs(np.std(log_scale_new_radius))/np.sqrt(2.*N) )

        galaxies.append(nbgal[i])

    step = 0.05
    x_data = np.array([[i*(1-2*step) for i in galaxies], [i*(1-step) for i in galaxies], [i*(1+step) for i in galaxies], [ i*(1+2*step) for i in galaxies]])

    labels_x = ["", "", "", "Number of galaxies"]
    labels_plots = ["Median guess : old", "Median found : old", "Median guess : new", "Median found : new"]
    symbols = [".", "*", "x", "d"]
    #title_label = meth[j].upper() + " method (no background, no ellipticity)"
    title_label = meth[j].upper() + " method (background + ellipticity)"

    y_data = np.array([scale_old_medianguess_bias, scale_old_medianfound_bias, scale_new_medianguess_bias, scale_new_medianfound_bias])
    disp = [disp_scale_oldguess, disp_scale_old, disp_scale_newguess, disp_scale_new]
    labels_y =  ["", "", "", r'$\mathrm{log_{10} (r_{-2} / r_{-2}^{true})}$']
    plot_many_graphs(x_data, y_data, disp, labels_x, labels_y, labels_plots, symbols, title_label, meth[j] + "_comp_oldnew.pdf")

    scale_old_medianguess_bias = []
    scale_old_medianfound_bias = []
    disp_scale_oldguess        = []
    disp_scale_old             = []

    scale_new_medianguess_bias = []
    scale_new_medianfound_bias = []
    disp_scale_newguess        = []
    disp_scale_new             = []

    galaxies                   = []

print("Time taken : " + str(time.clock() - start) + " s")


















