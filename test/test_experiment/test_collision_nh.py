# %%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType, Concept
from aiphy.experiment.basic import collision_nh_config
from aiphy.experiment.ground_laboratory import gravity_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt

# %%
knowledge = Knowledge.default()
knowledge.register_expstruct('gravity', gravity_config())
freel = knowledge.register_expr("[#stringmotion0 (1->Spring) |- length[1]]")
mass = knowledge.register_expr(f"[#gravity (1->Particle) |- length[2] - {freel}[2]]")

expstruct = collision_nh_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())
plot_datastruct(res.datastruct)
p = knowledge.register_expr(f"(1->Particle) |- {mass}[1] * D[posx[1]]/D[t[0]]")
py = knowledge.register_expr(f"(1->Particle) |- {mass}[1] * D[posy[1]]/D[t[0]]")
e = knowledge.register_expr(f"(1->Particle) |- {mass}[1] * (D[posx[1]]/D[t[0]] ** 2 + D[posy[1]]/D[t[0]] ** 2)")
# %%
assert knowledge.eval(f"{e}[1] + {e}[2]", expstruct).is_const
assert knowledge.eval(f"{p}[1] + {p}[2]", expstruct).is_const
assert knowledge.eval(f"{py}[1] + {py}[2]", expstruct).is_const
# %%
