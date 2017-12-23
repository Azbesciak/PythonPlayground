from __future__ import division
from pylab import *
from numpy import *
from scipy import *
from ipywidgets import *
import math as mt


def do_the_job(A=1, LP=1, w=40, f=2.0):
    # generujemy momenty, w których pobieramy próbki
    n = len(t)
    signal = FUNC(t)
    # funkcja sprobkowana
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(121)
    ## --- POMOCNICZY SYGNAL
    base_t = np.arange(0, LP * T, 1.0 / 200.0)
    base_signal = FUNC(base_t)
    ax.plot(base_t, base_signal, linestyle='-', color='red')
    ax.set_ylim([min(base_signal), max(base_signal)])
    ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
    ## ---
    ax.plot(t, signal, 'o')

    signal1 = fft(signal)
    # sygnal w dziedzinie czestotliwosci
    signal1 = abs(signal1)
    # modul sygnalu

    freqs = range(int(n))

    ax = fig.add_subplot(122)
    ymax = max(signal1)
    if (ymax > 3.0):
        ax.set_ylim([0.0, ymax])
    else:
        ax.set_ylim([0.0, 3.0])
    stem(freqs, signal1, '-*')
    show()


do_the_job()