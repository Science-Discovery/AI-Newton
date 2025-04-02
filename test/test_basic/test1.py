from aiphy.core import Exp
a = Exp("(x[1]' + x[2]')")
print(a)
t = Exp("t[0]")
b = Exp.DiffExp(a.__powi__(2), t, 1)
print(str(b))
