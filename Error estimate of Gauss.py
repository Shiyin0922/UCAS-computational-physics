import sympy as xp
from sympy import init_printing

init_printing()
##开启init printing后就可以打印表达式
f = xp.Function("f")#声明函数
x, y, h = xp.symbols("x y h")#声明变量
order = 9  #定义阶数

exact = xp.Integral(xp.series(f(x+y), y, 0, order), (y, -h, h)).doit()
##积分实际的泰勒展开，先将f（x+y）以y为小量在x=0展开，再将每一项从-h到h对y积分
gauss2 = xp.series(
    h*f(x-h*xp.sqrt(xp.Rational(1,3))) +
    h*f(x+h*xp.sqrt(xp.Rational(1,3))),
    h, xp.Integer(0), order).doit()
##高斯2项展开式，按h展开，以0为展开点
gauss2_undo = xp.series(
    h*f(x-h*xp.sqrt(xp.Rational(1,3))) +
    h*f(x+h*xp.sqrt(xp.Rational(1,3))),
    h, xp.Integer(0), order)
##doit的作用是将执行算法输出一个表达式（自动合并同类项），去掉doit则输出泰勒展开还未进行substitute的表达式（会先以xi为原点展开，然后再用给定的值替换xi）
gauss3 = xp.series(
    xp.Rational(5,9)*h*f(x-h*xp.sqrt(xp.Rational(3,5))) +
    xp.Rational(8,9)*h*f(x) +
    xp.Rational(5,9)*h*f(x+h*xp.sqrt(xp.Rational(3,5))),
    h, xp.Integer(0), order).doit()
##高斯3项展开式
print(exact)
print(gauss2)
print(gauss2_undo)
##可以对比一下两个输出的区别
print(gauss3)
print(exact-gauss2)
print(exact-gauss3)