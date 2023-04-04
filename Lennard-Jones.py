import numpy as np
from numpy import ndarray
import doctest
from typing import Callable
import matplotlib.pyplot as plt
import math

def lj_potential_vectorised(rs: ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> ndarray:
    r6 = (sigma / rs) ** 2
    r6 *= r6 * r6
    return 4 * epsilon * r6 * (r6 - 1)
def force(rs: ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> ndarray:
    ri=1 / rs
    r6 = (sigma / rs) ** 6
    r12 = r6 ** 2
    return 24.0 * epsilon * ri * (-2 * r12 + r6)

def pair_potential_vectorised(xs: ndarray, potential: Callable[[ndarray, ...], ndarray],potential_args: tuple = ()) -> float:
    nparticles, ndim = xs.shape
    left_indices, right_indices = np.triu_indices(nparticles, k=1)
    rij = xs[left_indices] - xs[right_indices]
    dij = np.linalg.norm(rij, axis=1)
    return potential(dij, *potential_args).sum()

def gradient(xs: ndarray,dim:int) ->ndarray:
    xss=xs.reshape(-1,dim)
    nparticles, ndim = xss.shape
    grad=np.zeros((nparticles,nparticles,ndim))
    left, right = np.triu_indices(nparticles, k=1)
    xij = xss[left] - xss[right]
    dij = np.linalg.norm(xij, axis=1,keepdims=True)
    grad[left,right]+=force(dij)*xij/dij ##链式法则，把每个dij看成标量，dij对xij的导数是xij/dij，V（dij）对dij的导数由力函数force给出
    grad[right,left]=-grad[left,right]
    return grad.sum(axis=1).flatten()

def energy_calculation(matrix:ndarray) -> float:
    return pair_potential_vectorised(
        xs=matrix.reshape(13, 3),##输入坐标，记在xs这个数组里
        potential=lambda x: lj_potential_vectorised(x) ##lambda是关于x的匿名函数，此处就是省略了potential的def，直接用lambda定义
    )


def line_search(
    f: Callable[[ndarray], ndarray],
    df: Callable[[ndarray], ndarray],
    x: ndarray,
    c: float = 0.5,
    t: float = 0.5,
    max_alpha: float = 1):
    p=df(x)/ np.linalg.norm(df(x))
    m=(df(x).T@p)
    fx = f(x)
    alpha = max_alpha
    while f(x + alpha * p) > fx + c * alpha * m:
        alpha *= t
    return alpha

def gradient_descent(xk:ndarray,dim):
    count = 0
    while True:
        p=gradient(xk,dim)/np.linalg.norm(gradient(xk,dim))
        f=lambda x: energy_calculation(x)
        df=lambda x: gradient(x,dim)
        xk1 = xk - line_search(f,df,xk)*p
        count += 1
        if count>100:
            break;
    print('迭代次数为:',count)
    return xk


matrix1 = np.load("E:\PyCharm 2022.3.2\程序\伦纳德琼斯势\config.npy") ##加载npy文件
energy = energy_calculation(matrix1)
g=np.linalg.norm(gradient(matrix1,3),axis=0)
minimum=gradient_descent(matrix1,3)
print(minimum)
print(round(energy,8))
print(round(g,8))

