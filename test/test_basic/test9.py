#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)
from aiphy import Knowledge, Exp

knowledge = Knowledge.default()
exp = Exp("posx[1] * posx[1]' + posx[1]' * posx[1]")
print(knowledge.generalize('collision', exp))
print(knowledge.generalize('collision', exp.doit()))

exp = Exp("posx[1] * posx[1]' + posx[2]' * posx[2]")
res = knowledge.generalize('collision', exp.doit())
assert str(res) == "[Sum:Particle] (1->Particle) |- (posx[1] * D[posx[1]]/D[t[0]])"

exp = Exp("posx[2]' * posx[2] + posx[1]' * posx[1]")
res = knowledge.generalize('collision', exp.doit())
assert str(res) == "[Sum:Particle] (1->Particle) |- (posx[1] * D[posx[1]]/D[t[0]])"

exp = Exp("((C_01[2] / C_09[2]) + (C_01[1] / C_09[1]))")
res = knowledge.generalize('collision', exp.doit())
print(res)
# %%
