import matplotlib
import time

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np

def plot_one_graph(datax, datay, xlabel, ylabel, plot_label, symbol, title):
    plt.plot(datax, datay, label=plot_label, marker=symbol, linestyle='None')
    plt.xlabel(xlabel, fontsize=size_text)
    plt.ylabel(ylabel, fontsize=size_text)
    plt.title(title, fontsize=size_text)

def plot_many_graphs(datax, datay, xlabels, ylabels, plot_labels, symbols, title, export):
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', direction='in')
    for i in range(len(datax)):
        plot_one_graph(datax[i], datay[i], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)

    plt.legend(prop={'size': size_text-6}, loc='upper right')
    plt.xscale('log')
    plt.grid(linestyle='dashed')

    dir_ext = "comp_meth_scatter/"
    plt.savefig(dir_ext + export)
    plt.show()


# MAIN #

#Exact values
true_log_scale = -2.08
true_ell       = 0.5
true_PA        = 50
true_bg        = 20

size_text      = 14

###input data filenames###
names = ["1280_bfgs_ellbg.output", "160_de_ellbg.output", "20_lbb_ellbg.output", "320_nm_ellbg.output", "40_tnc_ellbg.output", "80_bfgs_ellbg.output",
         "1280_de_ellbg.output", "160_lbb_ellbg.output", "20_nm_ellbg.output", "320_tnc_ellbg.output", "640_bfgs_ellbg.output", "80_de_ellbg.output",
         "1280_lbb_ellbg.output", "160_nm_ellbg.output", "20_tnc_ellbg.output", "40_bfgs_ellbg.output", "640_de_ellbg.output", "80_lbb_ellbg.output",
         "1280_nm_ellbg.output", "160_tnc_ellbg.output", "320_bfgs_ellbg.output", "40_de_ellbg.output", "640_lbb_ellbg.output", "80_nm_ellbg.output",
         "1280_tnc_ellbg.output", "20_bfgs_ellbg.output", "320_de_ellbg.output", "40_lbb_ellbg.output", "640_nm_ellbg.output", "80_tnc_ellbg.output",
         "160_bfgs_ellbg.output", "20_de_ellbg.output", "320_lbb_ellbg.output", "40_nm_ellbg.output", "640_tnc_ellbg.output"]

names1 = ["1280_bfgs_ellnobg.output", "160_de_ellnobg.output", "20_lbb_ellnobg.output", "320_nm_ellnobg.output", "40_tnc_ellnobg.output", "80_bfgs_ellnobg.output",
         "1280_de_ellnobg.output", "160_lbb_ellnobg.output", "20_nm_ellnobg.output", "320_tnc_ellnobg.output", "640_bfgs_ellnobg.output", "80_de_ellnobg.output",
         "1280_lbb_ellnobg.output", "160_nm_ellnobg.output", "20_tnc_ellnobg.output", "40_bfgs_ellnobg.output", "640_de_ellnobg.output", "80_lbb_ellnobg.output",
         "1280_nm_ellnobg.output", "160_tnc_ellnobg.output", "320_bfgs_ellnobg.output", "40_de_ellnobg.output", "640_lbb_ellnobg.output", "80_nm_ellnobg.output",
         "1280_tnc_ellnobg.output", "20_bfgs_ellnobg.output", "320_de_ellnobg.output", "40_lbb_ellnobg.output", "640_nm_ellnobg.output", "80_tnc_ellnobg.output",
         "160_bfgs_ellnobg.output", "20_de_ellnobg.output", "320_lbb_ellnobg.output", "40_nm_ellnobg.output", "640_tnc_ellnobg.output"]

names2 = ["1280_bfgs_noellbg.output", "160_de_noellbg.output", "20_lbb_noellbg.output", "320_nm_noellbg.output", "40_tnc_noellbg.output", "80_bfgs_noellbg.output",
         "1280_de_noellbg.output", "160_lbb_noellbg.output", "20_nm_noellbg.output", "320_tnc_noellbg.output", "640_bfgs_noellbg.output", "80_de_noellbg.output",
         "1280_lbb_noellbg.output", "160_nm_noellbg.output", "20_tnc_noellbg.output", "40_bfgs_noellbg.output", "640_de_noellbg.output", "80_lbb_noellbg.output",
         "1280_nm_noellbg.output", "160_tnc_noellbg.output", "320_bfgs_noellbg.output", "40_de_noellbg.output", "640_lbb_noellbg.output", "80_nm_noellbg.output",
         "1280_tnc_noellbg.output", "20_bfgs_noellbg.output", "320_de_noellbg.output", "40_lbb_noellbg.output", "640_nm_noellbg.output", "80_tnc_noellbg.output",
         "160_bfgs_noellbg.output", "20_de_noellbg.output", "320_lbb_noellbg.output", "40_nm_noellbg.output", "640_tnc_noellbg.output"]

names3 = ["1280_bfgs_noellnobg.output", "160_de_noellnobg.output", "20_lbb_noellnobg.output", "320_nm_noellnobg.output", "40_tnc_noellnobg.output", "80_bfgs_noellnobg.output",
         "1280_de_noellnobg.output", "160_lbb_noellnobg.output", "20_nm_noellnobg.output", "320_tnc_noellnobg.output", "640_bfgs_noellnobg.output", "80_de_noellnobg.output",
         "1280_lbb_noellnobg.output", "160_nm_noellnobg.output", "20_tnc_noellnobg.output", "40_bfgs_noellnobg.output", "640_de_noellnobg.output", "80_lbb_noellnobg.output",
         "1280_nm_noellnobg.output", "160_tnc_noellnobg.output", "320_bfgs_noellnobg.output", "40_de_noellnobg.output", "640_lbb_noellnobg.output", "80_nm_noellnobg.output",
         "1280_tnc_noellnobg.output", "20_bfgs_noellnobg.output", "320_de_noellnobg.output", "40_lbb_noellnobg.output", "640_nm_noellnobg.output", "80_tnc_noellnobg.output",
         "160_bfgs_noellnobg.output", "20_de_noellnobg.output", "320_lbb_noellnobg.output", "40_nm_noellnobg.output", "640_tnc_noellnobg.output"]

names4 = ["1280_bfgs_ellbgcen_10000.output", "160_de_ellbgcen_10000.output", "20_lbb_ellbgcen_10000.output", "320_nm_ellbgcen_10000.output", "40_tnc_ellbgcen_10000.output", "80_bfgs_ellbgcen_10000.output",
         "1280_de_ellbgcen_10000.output", "160_lbb_ellbgcen_10000.output", "20_nm_ellbgcen_10000.output", "320_tnc_ellbgcen_10000.output", "640_bfgs_ellbgcen_10000.output", "80_de_ellbgcen_10000.output",
         "1280_lbb_ellbgcen_10000.output", "160_nm_ellbgcen_10000.output", "20_tnc_ellbgcen_10000.output", "40_bfgs_ellbgcen_10000.output", "640_de_ellbgcen_10000.output", "80_lbb_ellbgcen_10000.output",
         "1280_nm_ellbgcen_10000.output", "160_tnc_ellbgcen_10000.output", "320_bfgs_ellbgcen_10000.output", "40_de_ellbgcen_10000.output", "640_lbb_ellbgcen_10000.output", "80_nm_ellbgcen_10000.output",
         "1280_tnc_ellbgcen_10000.output", "20_bfgs_ellbgcen_10000.output", "320_de_ellbgcen_10000.output", "40_lbb_ellbgcen_10000.output", "640_nm_ellbgcen_10000.output", "80_tnc_ellbgcen_10000.output",
         "160_bfgs_ellbgcen_10000.output", "20_de_ellbgcen_10000.output", "320_lbb_ellbgcen_10000.output", "40_nm_ellbgcen_10000.output", "640_tnc_ellbgcen_10000.output"]

names5 = ["1280_medsep_ellnobg.output", "160_medsep_ellnobg.output", "20_medsep_ellnobg.output", "320_medsep_ellnobg.output", "40_medsep_ellnobg.output", "640_medsep_ellnobg.output", "80_medsep_ellnobg.output"]

nbgal    = [1280, 160, 20, 320, 40, 80, 1280, 160, 20, 320, 640, 80, 1280, 160, 20, 40, 640, 80, 1280, 160, 320, 40, 640, 80, 1280, 20, 320, 40, 640, 80, 160, 20, 320, 40, 640]
meth     = ["bfgs", "de", "lbb", "nm", "tnc"]*7
galaxies = []

#time
start = time.clock()

scale_scatter, ell_scatter, PA_scatter, bg_scatter     = [], [], [], []
scale_scatter1, ell_scatter1, PA_scatter1, bg_scatter1 = [], [], [], []
scale_scatter2, ell_scatter2, PA_scatter2, bg_scatter2 = [], [], [], []
scale_scatter3, ell_scatter3, PA_scatter3, bg_scatter3 = [], [], [], []
scale_scatter4, ell_scatter4, PA_scatter4, bg_scatter4 = [], [], [], []
scale_scatter5                                         = []

scale_guess_scatter, ell_guess_scatter, PA_guess_scatter, bg_guess_scatter     = [], [], [], []
scale_guess_scatter1, ell_guess_scatter1, PA_guess_scatter1, bg_guess_scatter1 = [], [], [], []
scale_guess_scatter2, ell_guess_scatter2, PA_guess_scatter2, bg_guess_scatter2 = [], [], [], []
scale_guess_scatter3, ell_guess_scatter3, PA_guess_scatter3, bg_guess_scatter3 = [], [], [], []
scale_guess_scatter4, ell_guess_scatter4, PA_guess_scatter4, bg_guess_scatter4 = [], [], [], []

galaxies  = []
galaxies2 = []

directory = "../ell_bg/"
directory1 = "../ell_nobg/"
directory2 = "../noell_bg/"
directory3 = "../noell_nobg/"
directory4 = "../ell_bg_cen10000/"
directory5 = "../medsep/"

#median separation
for i in range(len(names5)):
    log_scale_radius = np.genfromtxt(directory5 + names5[i], usecols=(12,), unpack=True)
    N = len(log_scale_radius)

    scale_scatter5.append(np.median(log_scale_radius) - true_log_scale)

    galaxies2.append(nbgal[i+12])
    log_scale_radius = []

for j in range(1,len(meth)//7):
    for i in range(j, len(names), 5):
		#ellipticity and background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory + names[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_scatter.append(np.std(log_scale_radius))
        ell_scatter.append(np.std(ell))
        PA_scatter.append(np.std(PA))
        bg_scatter.append(np.std(log_bg))

        scale_guess_scatter.append(np.std(lsr_guess))
        ell_guess_scatter.append(np.std(ell_guess))
        PA_guess_scatter.append(np.std(PA_guess))
        bg_guess_scatter.append(np.std(log_bg_guess))

		#ellipticity no background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory1 + names1[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_scatter1.append(np.std(log_scale_radius))
        ell_scatter1.append(np.std(ell))
        PA_scatter1.append(np.std(PA))
        bg_scatter1.append(np.std(log_bg))

        scale_guess_scatter1.append(np.std(log_scale_radius))
        ell_guess_scatter1.append(np.std(ell))
        PA_guess_scatter1.append(np.std(PA))
        bg_guess_scatter1.append(np.std(log_bg))

        #no ellipticity + background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory2 + names2[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_scatter2.append(np.std(log_scale_radius))
        ell_scatter2.append(np.std(ell))
        PA_scatter2.append(np.std(PA))
        bg_scatter2.append(np.std(log_bg))

        scale_guess_scatter2.append(np.std(lsr_guess))
        ell_guess_scatter2.append(np.std(ell_guess))
        PA_guess_scatter2.append(np.std(PA_guess))
        bg_guess_scatter2.append(np.std(log_bg_guess))

        #no ellipticity and no background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory3 + names3[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_scatter3.append(np.std(log_scale_radius))
        ell_scatter3.append(np.std(ell))
        PA_scatter3.append(np.std(PA))
        bg_scatter3.append(np.std(log_bg))

        scale_guess_scatter3.append(np.std(lsr_guess))
        ell_guess_scatter3.append(np.std(ell_guess))
        PA_guess_scatter3.append(np.std(PA_guess))
        bg_guess_scatter3.append(np.std(log_bg_guess))

        #ellipticity + background + centering + Monte-Carlo
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory4 + names4[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_scatter4.append(np.std(log_scale_radius))
        ell_scatter4.append(np.std(ell))
        PA_scatter4.append(np.std(PA))
        bg_scatter4.append(np.std(log_bg))

        scale_guess_scatter4.append(np.std(lsr_guess))
        ell_guess_scatter4.append(np.std(ell_guess))
        PA_guess_scatter4.append(np.std(PA_guess))
        bg_guess_scatter4.append(np.std(log_bg_guess))

        galaxies.append(nbgal[i])

    step = 0.05
    x_data = np.array([[i*(1- 2.5*step) for i in galaxies], [i*(1-2*step) for i in galaxies], [i*(1-1.5*step) for i in galaxies], [i*(1-step) for i in galaxies], [ i*(1-0.5*step) for i in galaxies], [i*(1+0.5*step) for i in galaxies], [i*(1+step) for i in galaxies], [i*(1+1.5*step) for i in galaxies], [ i*(1+2*step) for i in galaxies], [i*(1+2.5*step) for i in galaxies]])
    x_data2 = np.array([[i*(1- 2.5*step) for i in galaxies2], [i*(1-2*step) for i in galaxies], [i*(1-1.5*step) for i in galaxies], [i*(1-step) for i in galaxies], [ i*(1-0.5*step) for i in galaxies], [i for i in galaxies], [i*(1+0.5*step) for i in galaxies], [i*(1+step) for i in galaxies], [ i*(1+1.5*step) for i in galaxies], [i*(1+2*step) for i in galaxies], [i*(1+2.5*step) for i in galaxies]])

    labels_x = ["", "", "", "", "", "", "", "", "", "Number of galaxies"]
    labels_x2 = ["", "", "", "", "", "", "", "", "", "", "Number of galaxies"]
    labels_plots = ["found : No bg, no ell", "found : no bg, ell", "found : no ell, bg", "found : ell + bg", "found : ell + bg + cen(MC)", " guess : No bg, no ell", "guess : no bg, ell", "guess : no ell, bg", "guess : ell + bg", "guess : ell + bg + cen(MC)"]
    labels_plots2 = ["found : median sep", "found : No bg, no ell", "found : no bg, ell", "found : no ell, bg", "found : ell + bg", "found : ell + bg + cen(MC)", " guess : No bg, no ell", "guess : no bg, ell", "guess : no ell, bg", "guess : ell + bg", "guess : ell + bg + cen(MC)"]
    symbols = [".", "*", "x", "d", "s", 'o', "v", '^', "<", ">"]
    symbols2 = ["+", ".", "*", "x", "d", "s", 'o', "v", '^', "<", ">"]

    y_data = np.array([scale_scatter5, scale_scatter3, scale_scatter1, scale_scatter2, scale_scatter, scale_scatter4, scale_guess_scatter3, scale_guess_scatter1, scale_guess_scatter2, scale_guess_scatter, scale_guess_scatter4])
    labels_y =  ["", "", "", "", "", "", "",  "", "", "", r'$\mathrm{\sigma[log_{10} (r_{-2} / r_{-2}^{true})]}$']
    plot_many_graphs(x_data2, y_data, labels_x2, labels_y, labels_plots2, symbols2, meth[j].upper() + " method", meth[j] + "_comp_scale_radius_scat.pdf")

    y_data = []
    y_data = np.array([ell_scatter3, ell_scatter1, ell_scatter2, ell_scatter, ell_scatter4, ell_guess_scatter3, ell_guess_scatter1, ell_guess_scatter2, ell_guess_scatter, ell_guess_scatter4])
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{\sigma[e - e^{true}]}$']
    plot_many_graphs(x_data, y_data, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_ell_scat.pdf")

    y_data = []
    y_data = np.array([PA_scatter3, PA_scatter1, PA_scatter2, PA_scatter, PA_scatter4, PA_guess_scatter3, PA_guess_scatter1, PA_guess_scatter2, PA_guess_scatter, PA_guess_scatter4])
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{\sigma[PA - PA^{true}]}$' + " (deg)"]
    plot_many_graphs(x_data, y_data, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_PA_scat.pdf")

    y_data = []
    y_data = np.array([bg_scatter3, bg_scatter1, bg_scatter2, bg_scatter, bg_scatter4, bg_guess_scatter3, bg_guess_scatter1, bg_guess_scatter2, bg_guess_scatter, bg_guess_scatter4])
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{\sigma[N_{bg} - N_{bg}^{true}]}$']
    plot_many_graphs(x_data, y_data, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_bg_scat.pdf")

    scale_scatter, ell_scatter, PA_scatter, bg_scatter     = [], [], [], []
    scale_scatter1, ell_scatter1, PA_scatter1, bg_scatter1 = [], [], [], []
    scale_scatter2, ell_scatter2, PA_scatter2, bg_scatter2 = [], [], [], []
    scale_scatter3, ell_scatter3, PA_scatter3, bg_scatter3 = [], [], [], []
    scale_scatter4, ell_scatter4, PA_scatter4, bg_scatter4 = [], [], [], []

    scale_guess_scatter, ell_guess_scatter, PA_guess_scatter, bg_guess_scatter     = [], [], [], []
    scale_guess_scatter1, ell_guess_scatter1, PA_guess_scatter1, bg_guess_scatter1 = [], [], [], []
    scale_guess_scatter2, ell_guess_scatter2, PA_guess_scatter2, bg_guess_scatter2 = [], [], [], []
    scale_guess_scatter3, ell_guess_scatter3, PA_guess_scatter3, bg_guess_scatter3 = [], [], [], []
    scale_guess_scatter4, ell_guess_scatter4, PA_guess_scatter4, bg_guess_scatter4 = [], [], [], []

    galaxies = []

print("Time taken : " + str(time.clock() - start) + " s")
















