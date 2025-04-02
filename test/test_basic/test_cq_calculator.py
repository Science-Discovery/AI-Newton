
#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import *

print(Exp("a+b").debug_str)
print(Exp("(P1-P2)").debug_str)
# %%
theorist = Theorist()
# %%
print(theorist.specific['motion']._sympy_of_raw_defi(Exp("posx[1]")))
print(theorist.specific['motion0']._sympy_of_raw_defi(Exp("D[t[0]]/D[posx[1]]")))
spm = theorist.specific['collision']
print(theorist.knowledge.specialize(
    '[Sum: Particle] (1 -> Particle) |- posx[1] * posx[1]',
    'collision'
)[0])
#%%
for data in theorist.specific['motion'].experiment.original_concept:
    print(data)
for name, data in theorist.knowledge.fetch_concepts.items():
    print(name, data)

# %%
from aiphy.dataplot import *
# %%
spm = theorist.specific['motion']
cqa = spm.make_conserved_info('a', Exp("posx[1]''"))
# %%
cqainv = spm.make_conserved_info('ainv', Exp("1/posx[1]''"))
# %%
from aiphy.specific_model import CQCalculator
# %%
cqcalc = CQCalculator(spm.exp_name, spm.knowledge)
# %%
print(cqcalc.insert_cq_info(cqa))
# %%
cqcalc.insert_cq_info(cqainv)
cqcalc.insert_cq_info(spm.make_conserved_info('azmulC', Exp("posz[1]'' * D[posx[1]]/D[posz[1]]")))
# %%
print('-'*50)
setconserved, setzero = cqcalc.calc_relations(debug=True)
for rel in setconserved:
    print(rel, 'conserved')
for rel in setzero:
    print(rel, 'zero')
assert len(setconserved) == 3
assert len(setzero) == 1
"""
calc_relations between  {'ainv', 'a', 'azmulC'}
Data for 1: [DataStruct] data:azmulC,ainv,a,.
Data for 2: [DataStruct] data:azmulC,a,ainv,.
Data for -1: [DataStruct] data:azmulC,a,ainv,.
(a / azmulC) conserved
(a * ainv) conserved
(ainv * azmulC) conserved
(azmulC + (-1 * a)) zero
"""
# %%
