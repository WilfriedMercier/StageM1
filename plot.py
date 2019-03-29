import matplotlib.pyplot as plt
import numpy as np

x, y = np.genfromtxt("donnees", usecols=(0,1), unpack=True)

plt.plot(x[-40:], y[-40:], "o")
plt.show()
