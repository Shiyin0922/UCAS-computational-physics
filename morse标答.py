from typing import Callable

import matplotlib.figure, matplotlib.axes
import numpy as np
from numpy import ndarray


def morse_potential(x: ndarray, rho: float, epsilon: float = 1.0, r0: float = 1.0) -> ndarray:
    """
    Compute the Morse potential at pair distance x with well depth epsilon,
    equilibrium distance r0, and shape parameter rho
    """
    x = x / r0
    v = np.exp(rho * (1 - x))  ##注意用np.exp，用math会报错
    return epsilon * v * (v - 2)  ##公式计算，没什么好说的


def pairwise_potential(potential: Callable[[ndarray], ndarray], xss: ndarray) -> float:
    """
    >>> import numpy as np
    >>> morse_13_6 = np.loadtxt("morse-n_13-rho_6.txt")
    >>> rho = 6.0
    >>> pairwise_potential(potential = lambda x: morse_potential(x, rho), xss = morse_13_6) # doctest: +ELLIPSIS
    -42.439862...
    """
    atoms, *_ = xss.shape  ##shape返回的是一组数，每维有几个值那一位就返回几，*_表示后面的参数随便赋值，并不需要，我们只关心atom，写成atoms=xss.shape[0]也可
    left, right = np.triu_indices(atoms, k=1)  ##返回函数的上三角矩阵，k表示跟对角线偏移，left表示横坐标，right表示纵坐标
    ##这种上三角方法实质是讲0-atom的所有有序整数对写出来了，因为上三角往右上偏一格的所有元素下标刚好是有序对
    return potential(
        np.linalg.norm(xss[left] - xss[right], axis=1)##linalg.norm表示求范数，axis=1表示按行向量来处理，axis=0表示按列向量来处理
        ## 此处我们输入的xss里都是行向量，因此取axis=1,potential里的变量实际是一个一维数组，因此x最好定义成ndarray而不是float
    ).sum()


def question2(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Subplot) -> None:
    """
    Plot the morse potential on r/r0 in (0, 2)
    for rho in {3, 6, 10, 14} .
    """
    x = np.linspace(0.1, 2.0, 100)
    rhos = [3, 6, 10, 14]
    linestyles = ['-', '--', '-.', ':']  ##给四种线不同的格式

    for rho, linestyle in zip(rhos, linestyles):
        ax.plot(
            x,  ##横坐标
            morse_potential(x, rho=rho),  ##纵坐标
            linestyle=linestyle,  ##设置线的格式
            label=fr"$\rho = {rho:.0f}$"  ##设置线的名字，.0f表示rho保留0位小数输出，f是十进制浮点数，
        )

    ax.set_xlim(0, 2) ##设置坐标轴范围
    ax.set_ylim(-1.1, 0.5)
    ax.legend(loc="upper left")  ##设置图例的位置
    ax.set_xlabel("$r/r_{0}$")   ##设置坐标轴名字
    ax.set_ylabel("$V/\epsilon$")


def question3() -> float:
    """
    Compute the total energy of the Morse cluster in "morse-n_13-rho_6.txt"
    """
    morse_13_6 = np.loadtxt("E:\PyCharm 2022.3.2\程序\数据\morse-n_13-rho_6.txt") ##加载txt文件
    return pairwise_potential(
        potential=lambda x: morse_potential(x, rho=6), ##lambda是关于x的匿名函数，此处就是省略了potential的def，直接用lambda定义
        xss=morse_13_6 ##输入坐标，记在xss这个数组里
    )


def main():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6 / 2 ** 0.5))
    ##给定几个子图，constrained_layout是启用约束布局，防止重叠，figsize给定了图的x，y方向大小
    question2(fig, ax)

    plt.savefig('E:\PyCharm 2022.3.2\程序\morse—new.png', dpi=160)##保存路径+文件名，dpi是保存时的像素
    plt.show()  ##show一定要打在savefig后面，否则就会输出空白图片

    energy = question3()
    print(f"the total potential energy of the Morse cluster is {energy} epsilon.")
    ##print（f‘ 字符串’），通过识别字符串中的花括号，将变量形式的值传入字符串并格式化

if __name__ == '__main__':  ##当只运行该程序时，自己就是main，当作为class下的一个module时，该module将不会执行，因为已经不是main了
    main()
