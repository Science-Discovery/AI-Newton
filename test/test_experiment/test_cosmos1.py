# %%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType, Concept, Exp, Intrinsic, Expression
from aiphy.experiment.ground_laboratory import gravity_config
from aiphy.experiment.cosmos import celestial_1x_config
from aiphy.dataplot import plot_datastruct, plot_data

knowledge = Knowledge.default()
main_exp_x = celestial_1x_config()
main_exp_x.random_settings()

plot_data(knowledge.eval("posx[1]", main_exp_x))

# %%
plot_data(knowledge.eval("posx[2]", main_exp_x))


# %%
knowledge.register_expstruct('gravity', gravity_config())
knowledge.register_expstruct('celestial', main_exp_x)
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
assert knowledge.eval(exp, main_exp_x).is_const
# %%
G = knowledge.register_expr(Expression(f"[#celestial |- D[{sumT}]/D[dist[1,2]**(-1)] / {mass}[1] / {mass}[2]]"))
# %%
print(knowledge.eval(G, main_exp_x))
# %%
def test_is_const(main_exp, expr):
    count = 0
    for ti in range(10):
        main_exp.random_settings()
        data = knowledge.eval(expr, main_exp)
        if data.is_const:
            count += 1
    print(f"expr: {expr}, success: {count}/10")
    return count >= 8
# %%
assert test_is_const(main_exp_x, f"{G}*{mass}[1]*{mass}[2]/dist[1,2] - {sumT}")
# %%

from aiphy.experiment.cosmos import celestial_1y_config
main_exp_y = celestial_1y_config()
assert test_is_const(main_exp_y, f"{G}*{mass}[1]*{mass}[2]/dist[1,2] - {sumT}")
# %%
