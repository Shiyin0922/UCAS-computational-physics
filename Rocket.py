import numpy as np
from typing import Callable
x=np.array([[0,68,136,204,272,340,408,476,544],[0,404,867,1407,2059,2877,3980,5677,9520]])
def w(n:float) -> float:
    for i in range(0,9,1):
        if int(n)==x[0,i]:
            return x[1,i]
            break
def first_forward_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    first forward differences of f at point x with step size h
    """
    return (f(x + h) - f(x)) / h

def first_central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    first central differences of f at point x with step size h
    """
    return (f(x + h) - f(x-h)) / (h*2)

def five_point_formula(f: Callable[[float], float], x: float, h: float) -> float:
    """
    five point formula of f at point x with step size 2h
    """
    return (f(x-3*h) -27* f(x - h)+27*f(x+h)-f(x+3*h)) / (h * 48)
f=w
dif_fuc_1=first_forward_difference
dif_fuc_2=first_central_difference
dif_fuc_3=five_point_formula
print(dif_fuc_1(f,476,68))
print(dif_fuc_2(f,238,34))
print(dif_fuc_3(f,238,34))