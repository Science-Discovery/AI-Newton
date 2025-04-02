# %%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)
from aiphy import Knowledge, Exp
from aiphy.dataplot import plot_data
# %%
knowledge = Knowledge.default()
knowledge.fetch_exps
# %%
main_exp = knowledge.fetch_expstruct('collision')
main_exp.random_settings()
plot_data(knowledge.eval(Exp("posx[1]"), main_exp))
# %%
plot_data(knowledge.eval(Exp("D[posx[1]]/D[t[0]]"), main_exp))
# %%
plot_data(knowledge.eval(Exp("D^2[posx[1]]/D[t[0]]^2"), main_exp))
assert knowledge.eval(Exp("D^2[posx[1]]/D[t[0]]^2"), main_exp).is_zero
# %%
