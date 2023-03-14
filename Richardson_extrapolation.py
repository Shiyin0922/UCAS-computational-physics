from typing import Callable
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

def richardson_extrapolation(f: Callable[[float], float], k: int) -> Callable[[float], float]:
    """
    Apply Richardson extrapolation the function f(h) to remove the error term in order h**k.
    """
    def extrapolated_function(h: float) -> float:
        return (np.power(2,k)*f(h/2)-f(h))/(np.power(2,k)-1)

    return extrapolated_function

def first_forward_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    first forward differences of f at point x with step size h
    """
    return (f(x + h) - f(x)) / h

f = np.sin
df = np.cos
x0 = 0.6
h = np.logspace(-18,3,100)

difference_function = first_forward_difference
difference_function_name = difference_function.__name__.replace("_", " ")
##change first_forward_difference to first forward difference
applied_difference_function = partial(difference_function,f, x0)



fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
# make the plot presentable
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(h.min(), h.max())
ax.set_ylim(h.min(), h.max())
ax.set_xlabel("h")
ax.set_ylabel("absolute error")
ax.grid()
ax.set_title(f"Richardson extrapolation applied to {difference_function_name}")

# plot the error in the original approximation
error = abs(applied_difference_function(h) - df(x0))
ax.plot(h, error, label=difference_function_name)

# apply the extrapolation to remove error terms up to h**5,
# and plot the error in the new approximations
for k in range(1, 6, 1):
    applied_difference_function = richardson_extrapolation(applied_difference_function, k)
    error = abs(applied_difference_function(h) - df(x0))
    ax.plot(h, error, label=f"$R{{(k={k})}}$")

ax.legend()
plt.show()