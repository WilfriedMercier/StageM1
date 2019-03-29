import matplotlib
import time

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np

def plot_one_graph(datax, datay, error_y, xlabel, ylabel, plot_label, symbol, title):
    plt.errorbar(datax, datay, yerr=error_y, label=plot_label, marker=symbol, linestyle='None')
    plt.xlabel(xlabel, fontsize=size_text)
    plt.ylabel(ylabel, fontsize=size_text)
    plt.title(title, fontsize=size_text)

def plot_many_graphs(datax, datay, error_y, xlabels, ylabels, plot_labels, symbols, title, export):
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', direction='in')
    for i in range(len(datax)):
        plot_one_graph(datax[i], datay[i], error_y[i], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)

    plt.legend(prop={'size': size_text-6}, loc='upper right')
    plt.xscale('log')
    plt.grid(linestyle='dashed')

    dir_ext = "comp_meth/"
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

scale_medianfound_bias, ell_medianfound_bias, PA_medianfound_bias, bg_medianfound_bias     = [], [], [], []
scale_medianfound_bias1, ell_medianfound_bias1, PA_medianfound_bias1, bg_medianfound_bias1 = [], [], [], []
scale_medianfound_bias2, ell_medianfound_bias2, PA_medianfound_bias2, bg_medianfound_bias2 = [], [], [], []
scale_medianfound_bias3, ell_medianfound_bias3, PA_medianfound_bias3, bg_medianfound_bias3 = [], [], [], []
scale_medianfound_bias4, ell_medianfound_bias4, PA_medianfound_bias4, bg_medianfound_bias4 = [], [], [], []
scale_medianfound_bias5                                                                    = []


disp_scale, disp_ell, disp_PA, disp_bg = [], [], [], []
disp_scale1, disp_ell1, disp_PA1, disp_bg1 = [], [], [], []
disp_scale2, disp_ell2, disp_PA2, disp_bg2 = [], [], [], []
disp_scale3, disp_ell3, disp_PA3, disp_bg3 = [], [], [], []
disp_scale4, disp_ell4, disp_PA4, disp_bg4 = [], [], [], []
disp_scale5                                = []

scale_medianguess_bias, ell_medianguess_bias, PA_medianguess_bias, bg_medianguess_bias     = [], [], [], []
scale_medianguess_bias1, ell_medianguess_bias1, PA_medianguess_bias1, bg_medianguess_bias1 = [], [], [], []
scale_medianguess_bias2, ell_medianguess_bias2, PA_medianguess_bias2, bg_medianguess_bias2 = [], [], [], []
scale_medianguess_bias3, ell_medianguess_bias3, PA_medianguess_bias3, bg_medianguess_bias3 = [], [], [], []
scale_medianguess_bias4, ell_medianguess_bias4, PA_medianguess_bias4, bg_medianguess_bias4 = [], [], [], []

dispguess_scale, dispguess_ell, dispguess_PA, dispguess_bg	   = [], [], [], []
dispguess_scale1, dispguess_ell1, dispguess_PA1, dispguess_bg1 = [], [], [], []
dispguess_scale2, dispguess_ell2, dispguess_PA2, dispguess_bg2 = [], [], [], []
dispguess_scale3, dispguess_ell3, dispguess_PA3, dispguess_bg3 = [], [], [], []
dispguess_scale4, dispguess_ell4, dispguess_PA4, dispguess_bg4 = [], [], [], []

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

    scale_medianfound_bias5.append(np.median(log_scale_radius) - true_log_scale)
    disp_scale5.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )

    galaxies2.append(nbgal[i+12])
    log_scale_radius = []



for j in range(1,len(meth)//7):
    for i in range(j, len(names), 5):
		#ellipticity and background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory + names[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_medianfound_bias.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias.append(np.median(ell) - true_ell)
        PA_medianfound_bias.append(np.median(PA) - true_PA)
        bg_medianfound_bias.append(np.median(log_bg) - np.log10(true_bg))

        disp_scale.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        scale_medianguess_bias.append(np.median(lsr_guess) - true_log_scale)
        ell_medianguess_bias.append(np.median(ell_guess) - true_ell)
        PA_medianguess_bias.append(np.median(PA_guess) - true_PA)
        bg_medianguess_bias.append(np.median(log_bg_guess) - np.log10(true_bg))

        dispguess_scale.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        dispguess_ell.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        dispguess_PA.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        dispguess_bg.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

		#ellipticity no background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory1 + names1[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_medianfound_bias1.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias1.append(np.median(ell) - true_ell)
        PA_medianfound_bias1.append(np.median(PA) - true_PA)
        bg_medianfound_bias1.append(np.median(log_bg) - np.log10(true_bg))

        disp_scale1.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell1.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA1.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg1.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        scale_medianguess_bias1.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianguess_bias1.append(np.median(ell) - true_ell)
        PA_medianguess_bias1.append(np.median(PA) - true_PA)
        bg_medianguess_bias1.append(np.median(log_bg) - np.log10(true_bg))

        dispguess_scale1.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        dispguess_ell1.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        dispguess_PA1.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        dispguess_bg1.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

        #no ellipticity + background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory2 + names2[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_medianfound_bias2.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias2.append(np.median(ell) - true_ell)
        PA_medianfound_bias2.append(np.median(PA) - true_PA)
        bg_medianfound_bias2.append(np.median(log_bg) - np.log10(true_bg))

        disp_scale2.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell2.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA2.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg2.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        scale_medianguess_bias2.append(np.median(lsr_guess) - true_log_scale)
        ell_medianguess_bias2.append(np.median(ell_guess) - true_ell)
        PA_medianguess_bias2.append(np.median(PA_guess) - true_PA)
        bg_medianguess_bias2.append(np.median(log_bg_guess) - np.log10(true_bg))

        dispguess_scale2.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        dispguess_ell2.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        dispguess_PA2.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        dispguess_bg2.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

        #no ellipticity and no background
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory3 + names3[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_medianfound_bias3.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias3.append(np.median(ell) - true_ell)
        PA_medianfound_bias3.append(np.median(PA) - true_PA)
        bg_medianfound_bias3.append(np.median(log_bg) - np.log10(true_bg))

        disp_scale3.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell3.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA3.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg3.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        scale_medianguess_bias3.append(np.median(lsr_guess) - true_log_scale)
        ell_medianguess_bias3.append(np.median(ell_guess) - true_ell)
        PA_medianguess_bias3.append(np.median(PA_guess) - true_PA)
        bg_medianguess_bias3.append(np.median(log_bg_guess) - np.log10(true_bg))

        dispguess_scale3.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        dispguess_ell3.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        dispguess_PA3.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        dispguess_bg3.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

        #ellipticity + background + centering + Monte-Carlo
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory4 + names4[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)
        N = len(log_scale_radius)

        scale_medianfound_bias4.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias4.append(np.median(ell) - true_ell)
        PA_medianfound_bias4.append(np.median(PA) - true_PA)
        bg_medianfound_bias4.append(np.median(log_bg) - np.log10(true_bg))

        disp_scale4.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell4.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA4.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg4.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        scale_medianguess_bias4.append(np.median(lsr_guess) - true_log_scale)
        ell_medianguess_bias4.append(np.median(ell_guess) - true_ell)
        PA_medianguess_bias4.append(np.median(PA_guess) - true_PA)
        bg_medianguess_bias4.append(np.median(log_bg_guess) - np.log10(true_bg))

        dispguess_scale4.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        dispguess_ell4.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        dispguess_PA4.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        dispguess_bg4.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

        galaxies.append(nbgal[i])

        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = [], [], [], [], [], [], [], []

    step = 0.05
    x_data = np.array([[i*(1- 2.5*step) for i in galaxies], [i*(1-2*step) for i in galaxies], [i*(1-1.5*step) for i in galaxies], [i*(1-step) for i in galaxies], [ i*(1-0.5*step) for i in galaxies], [i*(1+0.5*step) for i in galaxies], [i*(1+step) for i in galaxies], [i*(1+1.5*step) for i in galaxies], [ i*(1+2*step) for i in galaxies], [i*(1+2.5*step) for i in galaxies]])
    x_data2 = np.array([[i*(1- 2.5*step) for i in galaxies2], [i*(1-2*step) for i in galaxies], [i*(1-1.5*step) for i in galaxies], [i*(1-step) for i in galaxies], [ i*(1-0.5*step) for i in galaxies], [i for i in galaxies], [i*(1+0.5*step) for i in galaxies], [i*(1+step) for i in galaxies], [ i*(1+1.5*step) for i in galaxies], [i*(1+2*step) for i in galaxies], [i*(1+2.5*step) for i in galaxies]])


    labels_x  = ["", "", "", "", "", "", "", "", "", "Number of galaxies"]
    labels_x2 = ["", "", "", "", "", "", "", "", "", "", "Number of galaxies"]
    labels_plots  = ["found : No bg, no ell", "found : no bg, ell", "found : no ell, bg", "found : ell + bg", "found : ell + bg + cen(MC)", " guess : No bg, no ell", "guess : no bg, ell", "guess : no ell, bg", "guess : ell + bg", "guess : ell + bg + cen(MC)"]
    labels_plots2 = ["found : median sep", "found : No bg, no ell", "found : no bg, ell", "found : no ell, bg", "found : ell + bg", "found : ell + bg + cen(MC)", " guess : No bg, no ell", "guess : no bg, ell", "guess : no ell, bg", "guess : ell + bg", "guess : ell + bg + cen(MC)"]
    symbols = [".", "*", "x", "d", "s", 'o', "v", '^', "<", ">"]
    symbols2 = ["+", ".", "*", "x", "d", "s", 'o', "v", '^', "<", ">"]

    y_data = np.array([scale_medianfound_bias5, scale_medianfound_bias3, scale_medianfound_bias1, scale_medianfound_bias2, scale_medianfound_bias, scale_medianfound_bias4, scale_medianguess_bias3, scale_medianguess_bias1, scale_medianguess_bias2, scale_medianguess_bias, scale_medianguess_bias4])
    disp = [disp_scale5, disp_scale3, disp_scale1, disp_scale2, disp_scale, disp_scale4, dispguess_scale3, dispguess_scale1, dispguess_scale2, dispguess_scale, dispguess_scale4]
    labels_y =  ["", "", "", "", "", "", "", "", "", "", r'$\mathrm{log_{10} (r_{-2} / r_{-2}^{true})}$']

    print(disp_scale5)
    plot_many_graphs(x_data2, y_data, disp, labels_x2, labels_y, labels_plots2, symbols2, meth[j].upper() + " method", meth[j] + "_comp_scale_radius.pdf")

    y_data = []
    disp   = []
    y_data = np.array([ell_medianfound_bias3, ell_medianfound_bias1, ell_medianfound_bias2, ell_medianfound_bias, ell_medianfound_bias4, ell_medianguess_bias3, ell_medianguess_bias1, ell_medianguess_bias2, ell_medianguess_bias, ell_medianguess_bias4])
    disp = [disp_ell3, disp_ell1, disp_ell2, disp_ell, disp_ell4, dispguess_ell3, dispguess_ell1, dispguess_ell2, dispguess_ell, dispguess_ell4]
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{e - e^{true}}$']
    plot_many_graphs(x_data, y_data, disp, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_ell.pdf")

    y_data = []
    disp   = []
    y_data = np.array([PA_medianfound_bias3, PA_medianfound_bias1, PA_medianfound_bias2, PA_medianfound_bias, PA_medianfound_bias4, PA_medianguess_bias3, PA_medianguess_bias1, PA_medianguess_bias2, PA_medianguess_bias, PA_medianguess_bias4])
    disp = [disp_PA3, disp_PA1, disp_PA2, disp_PA, disp_PA4, dispguess_PA3, dispguess_PA1, dispguess_PA2, dispguess_PA, dispguess_PA4]
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{PA - PA^{true}}$' + " (deg)"]
    plot_many_graphs(x_data, y_data, disp, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_PA.pdf")

    y_data = []
    disp   = []
    y_data = np.array([bg_medianfound_bias3, bg_medianfound_bias1, bg_medianfound_bias2, bg_medianfound_bias, bg_medianfound_bias4, bg_medianguess_bias3, bg_medianguess_bias1, bg_medianguess_bias2, bg_medianguess_bias, bg_medianguess_bias4])
    disp = [disp_bg3, disp_bg1, disp_bg2, disp_bg, disp_bg4, dispguess_bg3, dispguess_bg1, dispguess_bg2, dispguess_bg, dispguess_bg4]
    labels_y =  ["", "", "", "", "", "", "", "", "", r'$\mathrm{log_{10} (N_{bg} / N_{bg}^{true})}$']
    plot_many_graphs(x_data, y_data, disp, labels_x, labels_y, labels_plots, symbols, meth[j].upper() + " method", meth[j] + "_comp_bg.pdf")

    scale_medianfound_bias, ell_medianfound_bias, PA_medianfound_bias, bg_medianfound_bias     = [], [], [], []
    scale_medianfound_bias1, ell_medianfound_bias1, PA_medianfound_bias1, bg_medianfound_bias1 = [], [], [], []
    scale_medianfound_bias2, ell_medianfound_bias2, PA_medianfound_bias2, bg_medianfound_bias2 = [], [], [], []
    scale_medianfound_bias3, ell_medianfound_bias3, PA_medianfound_bias3, bg_medianfound_bias3 = [], [], [], []
    scale_medianfound_bias4, ell_medianfound_bias4, PA_medianfound_bias4, bg_medianfound_bias4 = [], [], [], []

    disp_scale, disp_ell, disp_PA, disp_bg     = [], [], [], []
    disp_scale1, disp_ell1, disp_PA1, disp_bg1 = [], [], [], []
    disp_scale2, disp_ell2, disp_PA2, disp_bg2 = [], [], [], []
    disp_scale3, disp_ell3, disp_PA3, disp_bg3 = [], [], [], []
    disp_scale4, disp_ell4, disp_PA4, disp_bg4 = [], [], [], []

    scale_medianguess_bias, ell_medianguess_bias, PA_medianguess_bias, bg_medianguess_bias     = [], [], [], []
    scale_medianguess_bias1, ell_medianguess_bias1, PA_medianguess_bias1, bg_medianguess_bias1 = [], [], [], []
    scale_medianguess_bias2, ell_medianguess_bias2, PA_medianguess_bias2, bg_medianguess_bias2 = [], [], [], []
    scale_medianguess_bias3, ell_medianguess_bias3, PA_medianguess_bias3, bg_medianguess_bias3 = [], [], [], []
    scale_medianguess_bias4, ell_medianguess_bias4, PA_medianguess_bias4, bg_medianguess_bias4 = [], [], [], []

    dispguess_scale, dispguess_ell, dispguess_PA, dispguess_bg     = [], [], [], []
    dispguess_scale1, dispguess_ell1, dispguess_PA1, dispguess_bg1 = [], [], [], []
    dispguess_scale2, dispguess_ell2, dispguess_PA2, dispguess_bg2 = [], [], [], []
    dispguess_scale3, dispguess_ell3, dispguess_PA3, dispguess_bg3 = [], [], [], []
    dispguess_scale4, dispguess_ell4, dispguess_PA4, dispguess_bg4 = [], [], [], []

    galaxies = []

print("Time taken : " + str(time.clock() - start) + " s")
















