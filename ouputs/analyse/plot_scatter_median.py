import matplotlib
import time

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(which='both', direction='in')

def plot_one_graph(datax, datay, xlabel, ylabel, plot_label, symbol, title):
    plt.plot(datax, datay, label=plot_label, marker=symbol, linestyle='None')
    ax.set_xlabel(xlabel, fontsize=size_text)
    ax.set_ylabel(ylabel, fontsize=size_text)
    ax.set_title(title, fontsize=size_text)

def plot_many_graphs(datax, datay, xlabels, ylabels, plot_labels, symbols, title, export):
    for i in range(len(datax)):
        if i < 2:
            plot_one_graph(datax[i], datay[i], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)
        else:
            plot_one_graph(datax[i], datay[i], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)
    plt.legend()
    plt.xscale('log')
    plt.grid(linestyle='dashed')

    #ext_dir = "scatter_ell_bg/"
    #ext_dir = "scatter_ell_nobg/"
    #ext_dir = "scatter_noell_bg/"
    ext_dir = "scatter_noell_nobg/"
    plt.savefig(ext_dir + export)
    plt.show()


# MAIN #

#Exact values
true_log_scale = -2.08
size_text      = 14

#names = ["1280_mediansep_elliptical.output", "160_mediansep_elliptical.output", "20_mediansep_elliptical.output", "320_mediansep_elliptical.output", "40_mediansep_elliptical.output", "640_mediansep_elliptical.output", "80_mediansep_elliptical.output"]
#names = ["1280_medsep_ellnobg.output", "160_medsep_ellnobg.output", "20_medsep_ellnobg.output", "320_medsep_ellnobg.output", "40_medsep_ellnobg.output", "640_medsep_ellnobg.output", "80_medsep_ellnobg.output"]
#names = ["1280_medsep_noellbg.output", "160_medsep_noellbg.output", "20_medsep_noellbg.output", "320_medsep_noellbg.output", "40_medsep_noellbg.output", "640_medsep_noellbg.output", "80_medsep_noellbg.output"]
names = ["1280_medsep_noellnobg.output", "160_medsep_noellnobg.output", "20_medsep_noellnobg.output", "320_medsep_noellnobg.output", "40_medsep_noellnobg.output", "640_medsep_noellnobg.output", "80_medsep_noellnobg.output"]

nbgal = [1280, 160, 20, 320, 40, 640, 80]

#time
start = time.clock()

#plots
disp_scaleguess = []
disp_scale      = []
galaxies        = []

#directory = "../old/ell_bg/"
#directory = "../old/ell_nobg/"
#directory = "../old/noell_bg/"
directory = "../old/noell_nobg/"
for i in range(0, len(names), 1):
    log_scale_radius = np.genfromtxt(directory + names[i], usecols=(12), unpack=True)

    disp_scale.append(np.std(log_scale_radius))
    galaxies.append(nbgal[i])

step = 0.01
x_data = np.array([[i*(1+step) for i in galaxies]])
labels_x = ["Number of galaxies"]
labels_plots = ["Found"]
symbols = ["x"]

#title_labels = ["Median separation method (ellipticity + background)"]
#title_labels = ["Median separation method (ellipticity, no background)"]
#title_labels = ["Median separation method (no ellipticity + background)"]
title_labels = ["Median separation method (no ellipticity, no background)"]

labels_y =  [r'$\mathrm{\sigma[log_{10} (r_{-2} / r_{-2}^{true})]}$']
y_data = np.array([disp_scale])
plot_many_graphs(x_data, y_data, labels_x, labels_y, labels_plots, symbols, title_labels[0], "medsep_scat.pdf")

print("Time taken : " + str(time.clock() - start) + " s")

