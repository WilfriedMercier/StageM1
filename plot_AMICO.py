import numpy as np
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
plt.style.use('classic')

rad, rad_g, rad_h, cut, ID, SNR, mass = np.genfromtxt("NFW_DE_0.001_nobg.dat", unpack=True)
rad2, rad_g2, rad_h2, cut2, ID2, SNR2, mass2 = np.genfromtxt("NFW_TNC_0.001_nobg.dat", unpack=True)

bias    = rad-rad_h
bias2   = rad2-rad_h2

bias = bias[SNR>3]
ID   = ID[SNR>3]
mass = mass[SNR>3]
SNR  = SNR[SNR >3]

bias_split  = np.array([])
SNR_split   = np.array([])
scat_rad    = np.array([])
mass_split  = np.array([])

#for i,j in [ [1, 100], [10000, 10100], [100000, 100100], [120000, 120100], [150000, 150100], [200000, 200100], [220000, 220100], [250000,250100] ]:
#    bias_split  = np.append(bias_split, np.median(bias[np.logical_and(ID>=i,ID<=j)]))
#    SNR_split   = np.append(SNR_split, np.median(SNR[np.logical_and(ID>=i, ID<=j)]))
#    scat_rad    = np.append(scat_rad, np.std(bias[np.logical_and(ID>=i,ID<=j)]))

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(which='both', direction='in')

plt.xscale('log')
plt.grid(linestyle="dashed")

size = 16

plt.plot(mass, bias, "s")
#plt.xlabel("S/N", fontsize=size)
#plt.ylabel(r'$10 \mathrm{\sigma[log_{10} (r_{-2} / r_{-2}^{true})]}$', fontsize=size)

#plt.plot(mass[SNR>3.5], bias[SNR>3.5], "s", label="DE")
#plt.plot(mass2[SNR2>3.5], bias2[SNR2>3.5], "x", label="TNC")
plt.title("AMICO - TNC - without background - tol = " + r'$10^{-3}$')
plt.xlabel("S/N", fontsize=size)
plt.ylabel(r'$\mathrm{log_{10} (r_{-2} / r_{-2}^{true})}$', fontsize=size)
plt.legend()
plt.show()
