from aiphy.core import Knowledge, struct_collision
from aiphy.core import Exp, Expression, Objstructure, search_relations_ver1
KK = Knowledge.default()
KK.list_experiments()

standard_sp = Objstructure.make_spring(1.8, 2.2, 6.0, 8.0)
standard_sp.random_settings()
KK.register_object("standard_sp", standard_sp)
exp = Expression("[#oscillation (1 -> Particle) [2->standard_sp] |- D[posx[1]]/D[posx[1]''] ]")
KK.register_expression("m", exp)
exp = Expression("(1->Particle) |- posx[1]' ")
KK.register_expression("v", exp)
exp = Expression("(1->Particle) |- m[1]*v[1]**2/2 ")
KK.register_expression("e", exp)
KK.list_concepts()

s = struct_collision()
s.random_settings()
data = KK.eval(Exp("m[1] * v[1] + m[2] * v[2]"), s)
print(data.is_conserved)

data = KK.eval(Exp("e[1] + e[2]"), s)
print(data.is_conserved)
print(s.obj_info)
ds = s.data_info
print(ds)
res = search_relations_ver1(ds)
for i in res:
    print(i[0])
