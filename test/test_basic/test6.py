from aiphy import Knowledge
from aiphy.core import MeasureType
from aiphy.core import search_trivial_relations
from aiphy.core import Concept

theorist = Knowledge.default()
theorist.fetch_exps
s = theorist.fetch_expstruct("motion0")
s.random_settings()
ds = s.collect_expdata(MeasureType.default())
assert len(str(s.data_info)) == len('[DataStruct] data:posx[1],posy[1],posz[1],t[0],.')

assert (s.data_info.fetch_data_by_str("posx[1]").__diff__(
    s.data_info.fetch_data_by_str("t[0]")).is_conserved)
res = search_trivial_relations(s.data_info)
for r, data in res:
    print(r, data)
result = set([str(r) for r, _ in res])
assert len(result) == 3 and 'D[posx[1]]/D[t[0]]' in result and 'posy[1]' in result and 'posz[1]' in result

theorist.register_expr("(1->Particle)|-(posx[1]')", "MP1")
print(theorist.fetch_concepts['MP1'])
s = theorist.fetch_expstruct("motion")
s.random_settings()
s.collect_expdata(MeasureType.default())
theorist.eval("MP1[1]", s)
print(str(s.data_info))
res = search_trivial_relations(s.data_info)
print(res[0][0])

concept = Concept("(1->Particle)(2->Particle)|-posx[1]-posx[2]")
exp = concept.subst_by_vec([2,1])
print(str(exp))
print(theorist.generalize("collision", str(exp)))
