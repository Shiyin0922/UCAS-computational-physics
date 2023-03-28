from typing import Callable
import doctest
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import ndarray

QUINTIC_SOLUTION = 1.1673039782614185
QUARTIC_SOLUTION = 0.5
CUBIC_SOLUTION = 1

def quintic(x: float) -> float:
    return x ** 5 - x - 1

def quintic_derivative(x: float) -> float:
    return 5 * x ** 4 - 1

def quartic(x: float) -> float:
    return 16 * x ** 4 - 8 * x + 3

def quartic_derivative(x: float) -> float:
    return 64 * x ** 3 - 8

def cubic(x: float) -> float:
    return (x - 4) * (x - 1) * (x + 3)

def cubic_derivative(x: float) -> float:
    return 3 * x ** 2 - 4 * x - 11

def relative_error(a, b):
    return abs((b - a) / max(abs(a),abs(b)))

def bisection(f: Callable[[float], float], a: float, b: float, tol: float) -> [float]:
    """
    Use the bisection method to find a zero of a function f on the bracket [a, b] to relative precision tol.
    >>> bisection(quintic, -1.0, 2.0, 1e-10)  #doctest: +ELLIPSIS
    [..., 1.16730397...]
    >>> bisection(cubic, 3.0, 5.0, 1e-10)
    [3.0, 5.0, 4.0]
    >>> bisection(quintic, 2.0, 3.0, 1e-10)
    Traceback (most recent call last):
    ValueError: [a, b] does not bracket a zero; f(a)*f(b) = 6931.0
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        return [10]
        ##raise ValueError(f"[a, b] does not bracket a zero; f(a)*f(b) = {f(a) * f(b)}
    sequence = [a, b]
    while relative_error(a, b) > tol:
        c = (a + b) / 2
        fc = f(c)
        if fc == 0:
            # if an exact solution is found, we should exit immediately
            sequence.append(c)
            return sequence
        if fa * fc < 0.0:
            a,b,fa,fb=(a,c,fa,fc)
        else:
            a,b,fa,fb=(c,b,fc,fb)
        sequence.append(c)
    return sequence

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float) -> [float]:
    """
    Use the secant method to find a zero of a function f close [x0, x1] to relative precision tol.
    >>> secant(quintic, 2.0, 3.0, 1e-10)  #doctest: +ELLIPSIS
    [..., 1.16730397...]
    >>> secant(quintic, -2.5, -2.4, 1e-8)  #doctest: +ELLIPSIS
    [..., 0.66626933...]
    """
    if x0>=x1:
        t=x1
        x1=x0
        x0=t
    sequence = [x0, x1]

    while relative_error(x0, x1) > tol:
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if f(x1) == 0:
            sequence.append(x1)
            return sequence
        sequence.append(x1)
    return sequence

def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float | ndarray, tol: float) -> float:
    """
    Use the Newton--Raphson method to find a zero of a function f close x0 to relative precision tol.
    >>> newton_raphson(quintic, quintic_derivative, 3.0, 1e-10)  #doctest: +ELLIPSIS
    [..., 1.16730397...]
    >>> newton_raphson(quintic, quintic_derivative, 0.0, 1e-10)  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: Newton--Raphson got stuck in a cycle; cycle = [-1.00025756..., -0.75032182..., 0.08335709...]
    """
    x1 = x0 - f(x0) / df(x0)
    sequence = [x0, x1]
    while relative_error(x0, x1) > tol:
        x0 = x1
        x1 = x0 - f(x0) / df(x0)
        if f(x1)==0:
            sequence.append(x1)
            return sequence
        if x1 in sequence and sequence[-1] != x1:
            ##cycle = sequence[sequence.index(x1):]
            ## raise ValueError(f"Newton--Raphson got stuck in a cycle; cycle = {cycle}")
            break
        sequence.append(x1)

    return sequence


def inverse_quadratic_interpolation(f: Callable[[float], float], a: float, b: float, c: float, tol: float) -> [float]:
    """
    Use the IQR method to find a zero of a function f close to the points [a, b, c] to relative precision tol.

    >>> inverse_quadratic_interpolation(quintic, -1.0, 2.0, 3.0, 1e-10)  #doctest: +ELLIPSIS
    [..., 1.16730397...]
    >>> inverse_quadratic_interpolation(cubic, -3.0, 1.0, 4.0, 1e-10)
    Traceback (most recent call last):
    ValueError: two or more points have the same function value; f(a), f(b), f(c) = (0.0, -0.0, 0.0)
    """
    if a>=c:
        t=c
        c=a
        a=t
    sequence = [a,b,c]
    while relative_error(a,c) > tol:
        d1 = (f(a) - f(b)) * (f(a) - f(c))
        d2 = (f(b) - f(a)) * (f(b) - f(c))
        d3 = (f(c) - f(a)) * (f(c) - f(b))
        if d1==0 or d2 ==0 or d3==0:
            return [10]
            ##raise ValueError(f"two or more points have the same function value; f(a), f(b), f(c) = {f(a), f(b), f(c)}")
        d= (f(b) * f(c)*a / d1) + (f(a) * f(c)*b / d2) + (f(b) * f(a)*c / d3)
        if f(d) == 0:
            sequence.append(d)
            return sequence
        a, b, c = (b,c,d)
        sequence.append(d)
    return sequence

doctest.testmod(verbose=True)

def order_estimation(sequence:list)->float:
    n=len(sequence)-1
    delta1=abs(sequence[n-1]-sequence[n-2])
    delta2=abs(sequence[n-2]-sequence[n-3])
    delta3=abs(sequence[n-3]-sequence[n-4])
    return (math.log(delta1/delta2))/(math.log(delta2/delta3))

def rate_calculate(sequence:list,q:float)->[float]:
    n=len(sequence)-1
    solution=round(sequence[n],9)
    d1=abs(sequence[n-1]-solution)
    d2=abs(sequence[n-2]-solution)
    u1=abs(d1/(d2**q))
    print(d1,d2)
    return u1

##draw pic

'''
for i in range(1000):
    j=-2+0.004*i
    y[i]=len(newton_raphson(quintic,quintic_derivative,j,1e-10))
'''
'''
for i in range(1000):
    j=-4.5+0.008*i
    w=len(secant(quintic,j,j+1.0,1e-8))
    y[i]=secant(quintic,j,j+1.0,1e-8)[w-1]
'''


'''x=[0,1,2]
x1=[0.4,1.4,2.4]
width=0.4
y1= rate_calculate(newton_raphson(quintic, quintic_derivative, 10, 1e-10),1)
y2= rate_calculate(newton_raphson(quartic,quartic_derivative,10, 1e-10),1)
y3= rate_calculate(newton_raphson(cubic, cubic_derivative,10, 1e-10),1)
y=[y1,y2,y3]
z1= rate_calculate(newton_raphson(quintic, quintic_derivative,10, 1e-10),order_estimation(newton_raphson(quintic, quintic_derivative,10, 1e-10)))
z2= rate_calculate(newton_raphson(quartic,quartic_derivative,10, 1e-10),order_estimation(newton_raphson(quartic,quartic_derivative,10, 1e-10)))
z3= rate_calculate(newton_raphson(cubic, cubic_derivative,10, 1e-10),order_estimation(newton_raphson(cubic, cubic_derivative,10, 1e-10)))
z=[z1,z2,z3]
ax.set_xticks([x+width/2 for x in range(3)], ['quintic','quartic','cubic'])
ax.bar(x,y,width=width,label="rate of x0=10,q=1")
ax.bar(x1,z,width=width,label="rate of x0=10,q=order_estimation")
'''

'''for a,b,i in zip(x,y,range(len(x))):
    plt.text(a,b+0.01,"%.2f"%y[i],ha='center',fontsize=10)
for a,b,i in zip(x1,z,range(len(x1))):
    plt.text(a,b+0.01,"%.2f"%z[i],ha='center',fontsize=10)'''

def solution(sequence:list)->float:
    k=round(sequence[len(sequence)-1],0)
    if k==4 or k==1 or k==-3:
        return k
    else:
        return 10

x = np.linspace(-6,6,10)
y = np.linspace(-6,6,10)

z1 = np.zeros((10,10))
z2 = np.zeros((10,10))
z3 = np.zeros((10,10))
for i,a in enumerate(x):
    for j,b in enumerate(y):
        z1[i, j] = solution(bisection(cubic,a,b,1e-10))
        z2[i, j] = solution(secant(cubic,a,b, 1e-10))
        z3[i, j] = solution(inverse_quadratic_interpolation(cubic,a,(a+b)/2,b, 1e-10))
xx,yy = np.meshgrid(x,y)
plt.figure(figsize=(20,6),dpi=80)
plt.figure(1)
ax1=plt.subplot(131)
ax2=plt.subplot(132)
ax3=plt.subplot(133)
c1 = ax1.pcolor(xx,yy,z1.T,cmap='jet')
c2 = ax2.pcolor(xx,yy,z2.T,cmap='jet')
c3 = ax3.pcolor(xx,yy,z3.T,cmap='jet')
plt.colorbar(c1, ax=ax1)
plt.colorbar(c2, ax=ax2)
plt.colorbar(c3, ax=ax3)
plt.suptitle('Solution of three methods,x y from [-6,6] step=1.2,brown blocks mean wrong solution or no solution')

plt.savefig('E:\PyCharm 2022.3.2\程序\The solution of three methods', dpi=160)
plt.show()

