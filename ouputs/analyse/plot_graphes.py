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
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', direction='in')
    for i in range(len(datax)):
        if i < 2:
            plot_one_graph(datax[i], datay[i], error_y[1], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)
        else:
            plot_one_graph(datax[i], datay[i], error_y[0], xlabels[i], ylabels[i], plot_labels[i], symbols[i], title)
    plt.legend()
    plt.xscale('log')
    plt.grid(linestyle='dashed')

    #dir_ext = "bias_ell_bg/"
    #dir_ext = "bias_ell_nobg/"
    #dir_ext = "bias_noell_bg/"
    dir_ext = "bias_noell_nobg/"
    #dir_ext = "bias_ell_bg_cen10000/"
    plt.savefig(dir_ext + export)
    plt.show()


###input data filenames###

#names = ["1280_bfgs_ellbg.output", "160_de_ellbg.output", "20_lbb_ellbg.output", "320_nm_ellbg.output", "40_tnc_ellbg.output", "80_bfgs_ellbg.output",
#         "1280_de_ellbg.output", "160_lbb_ellbg.output", "20_nm_ellbg.output", "320_tnc_ellbg.output", "640_bfgs_ellbg.output", "80_de_ellbg.output",
#         "1280_lbb_ellbg.output", "160_nm_ellbg.output", "20_tnc_ellbg.output", "40_bfgs_ellbg.output", "640_de_ellbg.output", "80_lbb_ellbg.output",
#         "1280_nm_ellbg.output", "160_tnc_ellbg.output", "320_bfgs_ellbg.output", "40_de_ellbg.output", "640_lbb_ellbg.output", "80_nm_ellbg.output",
#         "1280_tnc_ellbg.output", "20_bfgs_ellbg.output", "320_de_ellbg.output", "40_lbb_ellbg.output", "640_nm_ellbg.output", "80_tnc_ellbg.output",
#         "160_bfgs_ellbg.output", "20_de_ellbg.output", "320_lbb_ellbg.output", "40_nm_ellbg.output", "640_tnc_ellbg.output"]

#names = ["1280_bfgs_ellnobg.output", "160_de_ellnobg.output", "20_lbb_ellnobg.output", "320_nm_ellnobg.output", "40_tnc_ellnobg.output", "80_bfgs_ellnobg.output",
#         "1280_de_ellnobg.output", "160_lbb_ellnobg.output", "20_nm_ellnobg.output", "320_tnc_ellnobg.output", "640_bfgs_ellnobg.output", "80_de_ellnobg.output",
#         "1280_lbb_ellnobg.output", "160_nm_ellnobg.output", "20_tnc_ellnobg.output", "40_bfgs_ellnobg.output", "640_de_ellnobg.output", "80_lbb_ellnobg.output",
#         "1280_nm_ellnobg.output", "160_tnc_ellnobg.output", "320_bfgs_ellnobg.output", "40_de_ellnobg.output", "640_lbb_ellnobg.output", "80_nm_ellnobg.output",
#         "1280_tnc_ellnobg.output", "20_bfgs_ellnobg.output", "320_de_ellnobg.output", "40_lbb_ellnobg.output", "640_nm_ellnobg.output", "80_tnc_ellnobg.output",
#         "160_bfgs_ellnobg.output", "20_de_ellnobg.output", "320_lbb_ellnobg.output", "40_nm_ellnobg.output", "640_tnc_ellnobg.output"]

#names = ["1280_bfgs_noellbg.output", "160_de_noellbg.output", "20_lbb_noellbg.output", "320_nm_noellbg.output", "40_tnc_noellbg.output", "80_bfgs_noellbg.output",
#         "1280_de_noellbg.output", "160_lbb_noellbg.output", "20_nm_noellbg.output", "320_tnc_noellbg.output", "640_bfgs_noellbg.output", "80_de_noellbg.output",
#         "1280_lbb_noellbg.output", "160_nm_noellbg.output", "20_tnc_noellbg.output", "40_bfgs_noellbg.output", "640_de_noellbg.output", "80_lbb_noellbg.output",
#         "1280_nm_noellbg.output", "160_tnc_noellbg.output", "320_bfgs_noellbg.output", "40_de_noellbg.output", "640_lbb_noellbg.output", "80_nm_noellbg.output",
#         "1280_tnc_noellbg.output", "20_bfgs_noellbg.output", "320_de_noellbg.output", "40_lbb_noellbg.output", "640_nm_noellbg.output", "80_tnc_noellbg.output",
#         "160_bfgs_noellbg.output", "20_de_noellbg.output", "320_lbb_noellbg.output", "40_nm_noellbg.output", "640_tnc_noellbg.output"]

names = ["160_de_noellnobg.output", "20_lbb_noellnobg.output", "40_tnc_noellnobg.output",
         "1280_de_noellnobg.output", "160_lbb_noellnobg.output", "320_tnc_noellnobg.output", "80_de_noellnobg.output",
         "1280_lbb_noellnobg.output", "20_tnc_noellnobg.output", "640_de_noellnobg.output", "80_lbb_noellnobg.output",
         "160_tnc_noellnobg.output", "40_de_noellnobg.output", "640_lbb_noellnobg.output",
         "1280_tnc_noellnobg.output", "320_de_noellnobg.output", "40_lbb_noellnobg.output", "80_tnc_noellnobg.output",
         "20_de_noellnobg.output", "320_lbb_noellnobg.output", "640_tnc_noellnobg.output"]

#names = ["1280_bfgs_ellbgcen_10000.output", "160_de_ellbgcen_10000.output", "20_lbb_ellbgcen_10000.output", "320_nm_ellbgcen_10000.output", "40_tnc_ellbgcen_10000.output", "80_bfgs_ellbgcen_10000.output",
#         "1280_de_ellbgcen_10000.output", "160_lbb_ellbgcen_10000.output", "20_nm_ellbgcen_10000.output", "320_tnc_ellbgcen_10000.output", "640_bfgs_ellbgcen_10000.output", "80_de_ellbgcen_10000.output",
#         "1280_lbb_ellbgcen_10000.output", "160_nm_ellbgcen_10000.output", "20_tnc_ellbgcen_10000.output", "40_bfgs_ellbgcen_10000.output", "640_de_ellbgcen_10000.output", "80_lbb_ellbgcen_10000.output",
#         "1280_nm_ellbgcen_10000.output", "160_tnc_ellbgcen_10000.output", "320_bfgs_ellbgcen_10000.output", "40_de_ellbgcen_10000.output", "640_lbb_ellbgcen_10000.output", "80_nm_ellbgcen_10000.output",
#         "1280_tnc_ellbgcen_10000.output", "20_bfgs_ellbgcen_10000.output", "320_de_ellbgcen_10000.output", "40_lbb_ellbgcen_10000.output", "640_nm_ellbgcen_10000.output", "80_tnc_ellbgcen_10000.output",
#         "160_bfgs_ellbgcen_10000.output", "20_de_ellbgcen_10000.output", "320_lbb_ellbgcen_10000.output", "40_nm_ellbgcen_10000.output", "640_tnc_ellbgcen_10000.output"]


#nbgal = [1280, 160, 20, 320, 40, 80, 1280, 160, 20, 320, 640, 80, 1280, 160, 20, 40, 640, 80, 1280, 160, 320, 40, 640, 80, 1280, 20, 320, 40, 640, 80, 160, 20, 320, 40, 640]
nbgal = [160, 20, 40, 1280, 160, 320, 80, 1280, 20, 640, 80, 160, 40, 1280, 320, 40, 80, 20, 320, 640]
meth  = ["de", "lbb", "tnc"]*7

#Exact values
true_log_scale = -2.08
true_ell       = 0.5
true_PA        = 50
true_bg        = 1
N              = 100.

size_text      = 14

#bias plots
start = time.clock()

scale_medianguess_bias, ell_medianguess_bias, PA_medianguess_bias, bg_medianguess_bias = [], [], [], []
scale_meanguess_bias, ell_meanguess_bias, PA_meanguess_bias, bg_meanguess_bias = [], [], [], []
scale_medianfound_bias, ell_medianfound_bias, PA_medianfound_bias, bg_medianfound_bias = [], [], [], []
scale_meanfound_bias, ell_meanfound_bias, PA_meanfound_bias, bg_meanfound_bias = [], [], [], []
galaxies = []

disp_scaleguess, disp_ellguess, disp_PAguess, disp_bgguess = [], [], [], []
disp_scale, disp_ell, disp_PA, disp_bg = [], [], [], []

#directory = "../ell_bg/"
#directory = "../ell_nobg/"
#directory = "../noell_bg/"
directory = "../noell_nobg/"
#directory = "../ell_bg_cen10000/"
for j in range(1,len(meth)//7):
    for i in range(j, len(names), 5):
        log_scale_radius, ell, PA, log_bg, lsr_guess, ell_guess, PA_guess, log_bg_guess = np.genfromtxt(directory + names[i], usecols=(12, 13, 14, 15, 22, 23, 24, 25), unpack=True)

        scale_medianguess_bias.append(np.median(lsr_guess) - true_log_scale)
        ell_medianguess_bias.append(np.median(ell_guess) - true_ell)
        PA_medianguess_bias.append(np.median(PA_guess) - true_PA)
        bg_medianguess_bias.append(np.median(log_bg_guess) - np.log10(true_bg))

        scale_meanguess_bias.append(np.mean(lsr_guess) - true_log_scale)
        ell_meanguess_bias.append(np.mean(ell_guess) - true_ell)
        PA_meanguess_bias.append(np.mean(PA_guess) - true_PA)
        bg_meanguess_bias.append(np.mean(log_bg_guess) - np.log10(true_bg))

        scale_medianfound_bias.append(np.median(log_scale_radius) - true_log_scale)
        ell_medianfound_bias.append(np.median(ell) - true_ell)
        PA_medianfound_bias.append(np.median(PA) - true_PA)
        bg_medianfound_bias.append(np.median(log_bg) - np.log10(true_bg))

        scale_meanfound_bias.append(np.mean(log_scale_radius) - true_log_scale)
        ell_meanfound_bias.append(np.mean(ell) - true_ell)
        PA_meanfound_bias.append(np.mean(PA) - true_PA)
        bg_meanfound_bias.append(np.mean(log_bg) - np.log10(true_bg))

        disp_scaleguess.append( np.abs(np.std(lsr_guess))/np.sqrt(2.*N) )
        disp_ellguess.append( np.abs(np.std(ell_guess))/np.sqrt(2.*N) )
        disp_PAguess.append( np.abs(np.std(PA_guess))/np.sqrt(2.*N) )
        disp_bgguess.append( np.abs(np.std(log_bg_guess))/np.sqrt(2.*N) )

        disp_scale.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )
        disp_ell.append( np.abs(np.std(ell))/np.sqrt(2.*N) )
        disp_PA.append( np.abs(np.std(PA))/np.sqrt(2.*N) )
        disp_bg.append( np.abs(np.std(log_bg))/np.sqrt(2.*N) )

        galaxies.append(nbgal[i])

    step = 0.05
    x_data = np.array([[i*(1-2*step) for i in galaxies], [i*(1-step) for i in galaxies], [i*(1+step) for i in galaxies], [ i*(1+2*step) for i in galaxies]])


    labels_x = ["", "", "", "Number of galaxies"]
    labels_plots = ["Median guess", "Mean guess", "Median found", "Mean found"]
    symbols = [".", "*", "x", "d"]
    #title_labels = meth[j].upper() + " method (ellipticity + background)"
    #title_labels = meth[j].upper() + " method (ellipticity, no background)"
    #title_labels = meth[j].upper() + " method (no ellipticity + background)"
    title_labels = meth[j].upper() + " method (no ellipticity, no background)"
    #title_labels = meth[j].upper() + " method (ellipticity + background + centering + Monte-Carlo)"

    y_data = np.array([scale_medianguess_bias, scale_meanguess_bias, scale_medianfound_bias, scale_meanfound_bias])
    labels_y =  ["", "", "", r'$\mathrm{log_{10} (r_{-2} / r_{-2}^{true})}$']
    plot_many_graphs(x_data, y_data, [disp_scale, disp_scaleguess], labels_x, labels_y, labels_plots, symbols, title_labels, meth[j] + "_scale_radius.pdf")

    y_data = []
    y_data = np.array([ell_medianguess_bias, ell_meanguess_bias, ell_medianfound_bias, ell_meanfound_bias])
    labels_y =  ["", "", "", r'$\mathrm{e - e^{true}}$']
    plot_many_graphs(x_data, y_data, [disp_ell, disp_ellguess], labels_x, labels_y, labels_plots, symbols, title_labels, meth[j] + "_ell.pdf")

    y_data = []
    y_data = np.array([PA_medianguess_bias, PA_meanguess_bias, PA_medianfound_bias, PA_meanfound_bias])
    labels_y =  ["", "", "", r'$\mathrm{PA - PA^{true}}$' + " (deg)"]
    plot_many_graphs(x_data, y_data, [disp_PA, disp_PAguess], labels_x, labels_y, labels_plots, symbols, title_labels, meth[j] + "_PA.pdf")

    y_data = []
    y_data = np.array([bg_medianguess_bias, bg_meanguess_bias, bg_medianfound_bias, bg_meanfound_bias])
    labels_y =  ["", "", "", r'$\mathrm{log_{10} (N_{bg} / N_{bg}^{true})}$']
    plot_many_graphs(x_data, y_data, [disp_bg, disp_bgguess], labels_x, labels_y, labels_plots, symbols, title_labels, meth[j] + "_bg.pdf")

    scale_medianguess_bias, ell_medianguess_bias, PA_medianguess_bias, bg_medianguess_bias = [], [], [], []
    scale_meanguess_bias, ell_meanguess_bias, PA_meanguess_bias, bg_meanguess_bias = [], [], [], []
    scale_medianfound_bias, ell_medianfound_bias, PA_medianfound_bias, bg_medianfound_bias = [], [], [], []
    scale_meanfound_bias, ell_meanfound_bias, PA_meanfound_bias, bg_meanfound_bias = [], [], [], []
    disp_scaleguess, disp_ellguess, disp_PAguess, disp_bgguess = [], [], [], []
    disp_scale, disp_ell, disp_PA, disp_bg = [], [], [], []
    galaxies = []

print("Time taken : " + str(time.clock() - start) + " s")









