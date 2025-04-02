#%%
from aiphy import AtomExp
print(AtomExp.VariableIds("dis", [2, 3]))
print(AtomExp("dis[2,3]").allids)
print(AtomExp.get_t())
print(AtomExp.get_t().name)
dis23: AtomExp = AtomExp.VariableIds("dis", [2, 3])
print(dis23.name)
print(dis23.vec_ids)
print(dis23.substs({2: 3, 3: 33}))
#%%
from aiphy import Exp, AtomExp
print(Exp.Number(1))
exp_of_atom: Exp = Exp.Atom(AtomExp("dis[2,3]"))
print(exp_of_atom.unwrap_atom)
exp_of_atom = (exp_of_atom.__powi__(2) + exp_of_atom).__difft__(3)
print(exp_of_atom)
print(exp_of_atom.complexity)
print(exp_of_atom.subst_by_dict({2: 3, 3: 33}))
# %%
from aiphy import ExpData, NormalData
xx: ExpData = ExpData([[1, 1.1, 1.05], [1.05, 0.9, 1.13]])
assert xx.is_conserved
print(xx)

print(ExpData([[1.1, 1.2, 1.3], [-3.1, 1.2, 1.4]]).is_zero)
x: ExpData = ExpData([[1.1, 1.2, 1.3, 1.2, 2.2, 2.1], [-3.1, 1.2, 1.4, 1.1, 3.3, 0.0]])
x = (-x * x - x)*x / x
assert x.is_normal
xn : NormalData = x.normal_data
print(xn.badpts)
# %%
from aiphy import ExpData, ConstData
x: ExpData = ExpData([[0.99999911, 1.0001, 1.0000001], [1.000001, 0.99990001, 1.00001]])
xd = x.__powi__(2)/x.__powi__(2)
assert x.is_const
assert xd.is_const
print(x.const_data)
print(xd.const_data)
# %%
