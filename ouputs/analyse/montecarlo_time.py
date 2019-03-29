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

    plt.legend(prop={'size': size_text-6}, loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linestyle='dashed')

    dir_ext = "montecarlo_time/"
    plt.savefig(dir_ext + export)
    plt.show()


size_text = 14
directory = "../time_montecarlo/"
names     = ["10000", "50000", "100000", "250000", "500000", "750000", "1000000"]

medq, medsp, total     = [], [], []
dispq, dispsp, disptot = [], [], []
for i in range(len(names)):
    q, spline = np.genfromtxt(directory + names[i], usecols=(0,1), unpack=True)

    q      = np.asarray(q)
    spline = np.asarray(spline)

    medq.append(np.median(q))
    medsp.append(np.median(spline))
    total.append(np.median(q+spline))

    dispq.append(np.std(q))
    dispsp.append(np.std(spline))
    disptot.append(np.std(q+spline))

    q, spline = [], []

npt = [int(i) for i in names]

plot_many_graphs([npt, npt, npt], [medq, medsp, total], [dispq, dispsp, disptot], ["", "", "Number of Monte-Carlo points"], ["", "", "Median time per cluster (s)"], ["arcsinh generation", "spline generation", "total"], ["o", "s", "x"], "", "time.pdf")

