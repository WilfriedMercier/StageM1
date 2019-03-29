import matplotlib
import numpy as np
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#loga, likfalse, liktrue = np.genfromtxt("var_loga.dat", usecols=(0,1,2), unpack=True)

#plt.plot(loga, likfalse, "ro", label="fixed params = found by PROFCL")
#plt.plot(loga, liktrue, "bx", label="fixed params = true values")
#plt.xlabel("log10(a/deg)")
#plt.ylabel("-log(likelihood)")
#plt.legend()
#plt.show()

size = 40
loga = np.linspace(-3, -0.7, size)
e = np.linspace(0, 0.99, size)
PA = np.linspace(0, 180, size)
log_bg = np.linspace(-5, 5, size)

#for i in range(len(loga)):
#    print(i, loga[i])
#exit()

#PA_from_file, bg_from_file, lnlik = np.genfromtxt("output_phase_space_RA10.0_Dec20.0.dat", usecols=(2, 3, 4), unpack=True)
PA_from_file, loga_from_file, lnlik = np.genfromtxt("output_phase_space_RA10.0_Dec20.0.dat", usecols=(2, 0, 4), unpack=True)

wanted_loga = loga[17]
wanted_PA   = PA[2]

mask = np.logical_and(PA_from_file==wanted_PA, loga_from_file==wanted_loga)
lnlik = lnlik[mask]

x, y = np.meshgrid(log_bg, e)
lnlik.shape = (size, size)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#p = ax.plot_surface(x[:-20,:], y[:-20,:], lnlik[:-20,:], cmap=cm.coolwarm, norm=Normalize(vmin=-4000, vmax=-4700), alpha=0.9)
#c = ax.contour(x[:-20,:], y[:-20,:], lnlik[:-20,:])
#fig.colorbar(p, label="- log10(likelihood)")

p = plt.pcolormesh(x[:,:], y[:,:], lnlik[:,:], norm=Normalize(vmin=0, vmax=-4500), cmap='viridis')
c = plt.contour(x[:,:], y[:,:], lnlik[:,:], colors='r')
plt.colorbar(p, label="- log10(likelihood)")

plt.xlabel("log(" + r'$\Sigma_{\rm{bg}} \times \rm{deg}^2$' + ")")
plt.ylabel("ellipticity")
plt.title("e = 0.4   log(bg) = 3.0   PA = " + str(round(wanted_PA,1)) + "   " + "log10(" + r'$r_{-2}/\rm{deg}$' + ") = " + str(round(wanted_loga, 3)))
plt.savefig("plots/fixedPA_and_loga.pdf")
plt.show()

exit(-1)


#with values values
wanted_PA = PA[2]
wanted_bg = log_bg[-7]

#with found values
wanted_PA = PA[3]
wanted_bg = log_bg[-15]

mask = np.logical_and(PA_from_file==wanted_PA, bg_from_file==wanted_bg)
lnlik = lnlik[mask]

x, y = np.meshgrid(loga, e)
lnlik.shape = (size, size)

fig = plt.figure()
ax = fig.gca(projection='3d')

p = ax.plot_surface(x[:,:-10], y[:,:-10], lnlik[:,:-10], cmap=cm.coolwarm, norm=Normalize(vmin=-4000, vmax=-4500), alpha=0.9)
c = ax.contour(x[:,:-10], y[:,:-10], lnlik[:,:-10])
fig.colorbar(p, label="- log10(likelihood)")

#p = plt.pcolormesh(x[:][:], y[:][:], lnlik[:][:], norm=Normalize(vmin=-3500, vmax=-5000))
#c = plt.contour(x, y, lnlik)
#plt.colorbar(p, label="- log10(likelihood)")

plt.xlabel("log(a/deg)")
plt.ylabel("ellipticity")
plt.title("PA = " + str(round(wanted_PA,1)) + "   " + "log10(bg) = " + str(round(wanted_bg, 2)))
plt.show()

