from functools import partial
from typing import Callable

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def rectangle(f: Callable[[float], float], a: float | ndarray, b: float | ndarray) -> float | ndarray:

    h = b - a
    return h * f(a)

def trapezium(f: Callable[[float], float], a: float | ndarray, b: float | ndarray) -> float | ndarray:

    h = (b - a) / 2
    return h * (f(a) + f(b))

def simpson(f: Callable[[float], float], a: float | ndarray, b: float | ndarray) -> float | ndarray:

    h = (b - a) / 2
    return h / 3 * (f(a) + 4 * f((a + b) / 2) + f(b))


METHODS = {
    'rectangle': rectangle,
    'trapezium': trapezium,
    'simpson': simpson,
}


def gauss1(f: Callable[[float], float], a: float, b: float) -> float:
    """
    approximate the integral of f on the interval [a, b] using 1-node Gauss-Legendre quadrature

    >>> gauss1(f = lambda y: 4*y+4, a=1.0, b=3.0)
    24.0
    """
    h=b-a
    return  h*f((a+b)/2)


def gauss2(f: Callable[[float], float], a: float, b: float) -> float:
    """
    approximate the integral of f on the interval [a, b] using 2-node Gauss-Legendre quadrature

    >>> gauss2(f=lambda y: 4*y**3+3*y**2+4*y+4, a=1.0, b=3.0)
    130.0
    """
    h=(b-a)/2
    x0=(a+b)/2
    x1=1/np.sqrt(3)
    x2=-1/np.sqrt(3)
    return  h * ( f(x1*h+x0) + f(x2*h+x0) )


def gauss3(f: Callable[[float], float], a: float, b: float) -> float:
    """
    approximate the integral of f on the interval [a, b] using 3-node Gauss-Legendre quadrature

    >>> gauss3(f=lambda y: 6*y**5-5*y**4+4*y**3-3*y**2+2*y-1, a=1.0, b=3.0)
    546.0
    """
    h = (b - a) / 2
    x0 = (a + b) / 2
    x1 = np.sqrt(3) / np.sqrt(5)
    x2 = -np.sqrt(3) / np.sqrt(5)
    return h * (8*f(x0)/9 + 5*f(x1*h+x0)/9 + 5*f(x2*h+x0)/9)

METHODS.update({
    "gauss1": gauss1,
    "gauss2": gauss2,
    "gauss3": gauss3,
})

def integrate(f: Callable[[float], float], a: float, b: float, intervals: int, method: str = 'simpson') -> float:
    """
    compute the integral of f on the major interval [a,b] by numerically integrating over a number of minor intervals
    using the specified method.

    Methods available are:
        Newton--Cotes:
            rectangle: rectangle rule
            trapezium: trapezium rule
            simpson: Simpson's rule
        Gauss--Legendre:
            gauss1: Gauss--Legendre quadrature on one node
            gauss2: Gauss--Legendre quadrature on two nodes
            gauss3: Gauss--Legendre quadrature on three nodes

    >>> integrate(f = lambda y: 4, a=1.0, b=3.0, intervals=10, method='rectangle')
    8.0
    >>> integrate(f = lambda y: 4*y+4, a=1.0, b=3.0, intervals=10, method='trapezium')
    24.0
    >>> integrate(f=lambda y: 3*y**2+4*y+4, a=1.0, b=3.0, intervals=10, method='simpson')
    50.0
    """
    xs = np.linspace(a, b, intervals + 1)
    avals, bvals = xs[:-1], xs[1:]

    return METHODS[method](f, avals, bvals).sum()
##set initial variables
f = np.cos
big_f = np.sin
a, b = 0, np.pi / 3
exact = big_f(b) - big_f(a)

intervals = np.logspace(0, 6, 7).astype(int)
hs = (b - a) / intervals
##draw pic
fig, ax = plt.subplots(1, 1, constrained_layout=True)

methods = (
    "rectangle",
    "trapezium",
    "simpson",
)

for method in methods:
    integrals = np.array([
        integrate(f, a, b, intervals=n, method=method)
        for n in intervals
    ])
    errors = abs(exact - integrals)
    ax.plot(hs, errors, 'o--', label=f'{method}')

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("h")
ax.set_ylabel("absolute error")
ax.legend(ncols=2)
plt.savefig('E:\PyCharm 2022.3.2\程序\gausslegendre.png', dpi=160)
plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
