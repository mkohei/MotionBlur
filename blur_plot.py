# coding=utf-8

import numpy as np
from matplotlib import pylab as plt

target = np.array([0, 1, 2, 1, 0])
N, T = 9, 5
val = np.zeros(N)

for t in range(T):
    signal = np.zeros(N)
    signal[N//2-len(target)//2 + t-T//2:N//2+len(target)//2+1 + t-T//2] = target
    #signal[N//2-len(target)//2:N//2+len(target)//2+1] = target
    val += signal

    plt.figure()
    plt.plot(signal, label='observation signal')
    plt.plot(val, label='integrated value')
    plt.legend(loc="upper right")
    plt.title("t={}".format(t))
    plt.ylim(-1, 11)
    plt.savefig("{}".format(t))
