import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

def barrier1(x):
    return ( (np.pi/2.)*(x-1) - np.tan(np.pi*(x-1)/2.) )

def barrier2(x):
    return np.sqrt(np.abs(x-1))


x = np.arange(0.01, 1, 0.001)
y = np.arange(-1, -0.01, 0.001)
plt.plot(x, barrier1(x), "b")
plt.plot(y, barrier1(y), "b")
plt.plot(x, barrier2(x), "r")
plt.plot(y, barrier2(y), "r")
plt.show()
