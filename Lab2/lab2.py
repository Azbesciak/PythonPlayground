from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from sympy.interactive import printing

v = np.linspace(-5, 5, 101)
np.sin(v)
plt.plot(v, np.sin(v))
plt.show()

printing.init_printing(use_latex=True)

import sympy as sym
from sympy import *

x, y, z = symbols("x y z")
# Definicje zmiennych
k = Symbol("k", integer=True)
f = Function('f')
eq = ((x+y)**2 * (x+1))
print(expand(eq))