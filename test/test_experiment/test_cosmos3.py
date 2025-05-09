# %%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType, Concept, Exp, Intrinsic, Expression
from aiphy.experiment.ground_laboratory import gravity_config
from aiphy.experiment.cosmos import celestial_2_xy_config, celestial_3_config, celestial_4_config
from aiphy.dataplot import plot_datastruct, plot_data

knowledge = Knowledge.default()
main_exp = celestial_2_xy_config()
main_exp.random_settings()
knowledge.register_expstruct('gravity', gravity_config())
knowledge.register_expstruct('celestial', main_exp)
freel = knowledge.register_expr(
    Expression("[#stringmotion0 (1->Spring) |- length[1]]"))
mass = knowledge.register_expr(
    Expression(f"[#gravity (1->Particle) |- length[2] - {freel}[2]]"))
# %%
Txy = knowledge.register_expr(
    Expression("(1 -> Particle) |- (D[posx[1]]/D[t[0]] ** 2 + D[posy[1]]/D[t[0]] ** 2)"))
sumT = knowledge.register_expr(
    Expression(f"[Sum:Particle] (1->Particle) |- {mass}[1] * {Txy}[1]"))

# %%
exp = Exp(f"D[{sumT}]/D[dist[1,2]**(-1)]")
assert knowledge.eval(exp, main_exp).is_const
# %%
G = knowledge.register_expr(Expression(f"[#celestial |- D[{sumT}]/D[dist[1,2]**(-1)] / {mass}[1] / {mass}[2]]"))
# %%
print(knowledge.eval(G, main_exp))
# %%
Energy = knowledge.register_expr(Expression(
    f"[Sum:Particle] (1->Particle) (2->Particle) |- {G} * {mass}[1] * {mass}[2] / dist[1,2]"
))
# %%
main_exp.random_settings()
data = knowledge.eval(f"{Energy} / 2 - {sumT}", main_exp)
assert data.is_const
# %%
main_exp = celestial_3_config()
main_exp.random_settings()
data = knowledge.eval(f"{Energy} / 2 - {sumT}", main_exp)
assert data.is_const
# %%
main_exp = celestial_4_config()
main_exp.random_settings()
data = knowledge.eval(f"{Energy} / 2 - {sumT}", main_exp)
assert data.is_const
# %%
plot_data(data)
# %%
