from aiphy import Knowledge
from aiphy.core import MeasureType
from aiphy.core import Concept, AtomExp
theorist = Knowledge.default()
print(theorist.fetch_exps)
oscil = theorist.fetch_expstruct("oscillation")
oscil.random_settings()
oscil.collect_expdata(MeasureType.default())
print(oscil.data_info)
theorist.eval("posx[1] - length[2]", oscil).is_zero
concept: Concept = theorist.generalize("oscillation", "posx[1] - length[2]")
print(concept)
concept: Concept = theorist.generalize("oscillation", "D[posx[1]'']/D[posx[1]]")
print(concept)
theorist.register_expr(str(concept), "concept")
assert(concept == theorist.fetch_concept_by_name("concept"))

print(theorist.fetch_concepts)
res: list[AtomExp] = theorist.specialize_concept("concept", "oscillation")
print(res[0].vec_ids)
print(concept.subst_by_vec(res[0].vec_ids))

res = theorist.specialize(str(concept), "oscillation")
print(res[0])