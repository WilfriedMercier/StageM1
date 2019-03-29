import numpy as np

#name    = "NFW_TNC_0.001_bg_clust.dat"
#name    = "NFW_TNC_0.001_nobg_clust.dat"
#name    = "NFW_TNC_0.0001_bg_clust.dat"
#name    = "NFW_TNC_0.0001_nobg_clust.dat"
#name2   = "NFWtrunc_TNC_0.001_bg_clust.dat"
#name    = "NFW_DE_0.001_nobg_clust.dat"
name    = "medsep_amas.dat"
name2   = "medsep_halo.dat"

rad_h, ID, SNR, mass = np.genfromtxt(name2, usecols=(0, 1, 2, 3), unpack=True)
rad, ID_c = np.genfromtxt(name, usecols=(0, 1), unpack=True)

mask = np.isin(ID_c, ID)
rad = rad[mask]

file = open("medsep_AMICO.dat", "w")
for i in range(len(ID)):
    file.write(str(rad[i]) +" "+ " "+ str(rad_h[i]) +" "+ str(ID[i]) +" "+ str(SNR[i]) +" "+ str(mass[i]) + "\n")
file.close()
