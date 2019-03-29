import matplotlib

#setting appropriate and available library to use with plots
matplotlib.use('GTK3Cairo')

import matplotlib.pyplot as plt
import numpy as np

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(which='both', direction='in')

def plot_one_graph(datax, datay, error_y, xlabel, ylabel, plot_label, symbol):
    plt.errorbar(datax, datay, yerr=error_y, label=plot_label, marker=symbol, linestyle='None')
    plt.xlabel(xlabel, fontsize=size_text)
    plt.ylabel(ylabel, fontsize=size_text)

size_text = 14

#names = ["1280_medsep_ellnobg.output", "160_medsep_ellnobg.output", "20_medsep_ellnobg.output", "320_medsep_ellnobg.output", "40_medsep_ellnobg.output", "640_medsep_ellnobg.output", "80_medsep_ellnobg.output"]
names = ["1280_mediansep_elliptical.output", "160_mediansep_elliptical.output", "20_mediansep_elliptical.output", "320_mediansep_elliptical.output", "40_mediansep_elliptical.output", "640_mediansep_elliptical.output", "80_mediansep_elliptical.output"]
#names = ["1280_medsep_noellbg.output", "160_medsep_noellbg.output", "20_medsep_noellbg.output", "320_medsep_noellbg.output", "40_medsep_noellbg.output", "640_medsep_noellbg.output", "80_medsep_noellbg.output"]
#names = ["1280_medsep_noellnobg.output", "160_medsep_noellnobg.output", "20_medsep_noellnobg.output", "320_medsep_noellnobg.output", "40_medsep_noellnobg.output", "640_medsep_noellnobg.output", "80_medsep_noellnobg.output"]
nbgal = [1280, 160, 20, 320, 40, 640, 80]

true_log_scale = -2.08

scale_median_bias, scale_mean_bias, disp_scale = [], [], []

directory = "../old/ell_bg/"
#directory = "../old/ell_nobg/"
#directory = "../old/noell_bg/"
#directory = "../old/noell_nobg/"
for i in range(len(names)):
    log_scale_radius = np.genfromtxt(directory + names[i], usecols=(12), unpack=True)

    N = len(log_scale_radius)

    scale_median_bias.append(np.median(log_scale_radius) - true_log_scale)
    scale_mean_bias.append(np.mean(log_scale_radius) - true_log_scale)

    disp_scale.append( np.abs(np.std(log_scale_radius))/np.sqrt(2.*N) )


bias    = [scale_median_bias, scale_mean_bias]
labels  = ["Median", "Mean"]
symbols = [".", "x"]

for i in range(len(bias)):
    step = i*0.1
    x_data = [j*(1+step) for j in nbgal]
    plot_one_graph(x_data, bias[i], disp_scale, "Number of galaxies", r'$\mathrm{log_{10} (r_{-2} / r_{-2}^{true})}$', labels[i], symbols[i])


plt.title("Median separation method", fontsize=size_text)
plt.legend()
plt.xscale('log')
plt.grid(linestyle="dashed")

ext_dir = "bias_ell_bg/"
#ext_dir = "bias_ell_nobg/"
#ext_dir = "bias_noell_bg/"
#ext_dir = "bias_noell_nobg/"
plt.savefig(ext_dir + "mediansep.pdf")
plt.show()
