#%%
import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType, ExpData
from aiphy.experiment.ground_laboratory import gravity_config
from aiphy.experiment.oscillation import oscillation_rot_1_config, oscillation_rot_2_config, oscillation_rot_3_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt

# %%
knowledge = Knowledge.default()
vx = knowledge.register_expr(f"(1->Particle) |- D[posx[1]]/D[t[0]]")
vy = knowledge.register_expr(f"(1->Particle) |- D[posy[1]]/D[t[0]]")
vz = knowledge.register_expr(f"(1->Particle) |- D[posz[1]]/D[t[0]]")
ax = knowledge.register_expr(f"(1->Particle) |- D[{vx}[1]]/D[t[0]]")
ay = knowledge.register_expr(f"(1->Particle) |- D[{vy}[1]]/D[t[0]]")
az = knowledge.register_expr(f"(1->Particle) |- D[{vz}[1]]/D[t[0]]")
# %%
# mass and elastic coefficient
knowledge.register_expstruct('gravity', gravity_config())
freel = knowledge.register_expr("[#stringmotion0 (1->Spring) |- length[1]]")
mass = knowledge.register_expr(f"[#gravity (1->Particle) |- (length[2] - {freel}[2])]")
k = knowledge.register_expr(f"[#oscillation (2->Spring) |- (-{mass}[1] * {ax}[1] / (length[2] - {freel}[2]))]")

# %%
expstruct = oscillation_rot_1_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())
print(res.datastruct)
plot_datastruct(res.datastruct)
# %%
L = knowledge.register_expr(f"(1->Particle) |- {mass}[1] * (posx[1] * D[posy[1]]/D[t[0]] - posy[1] * D[posx[1]]/D[t[0]])")
Lall = knowledge.register_expr(f"[Sum: Particle] (1->Particle) |- {L}[1]")
T = knowledge.register_expr(f"(1->Particle) |- {mass}[1] * (D[posx[1]]/D[t[0]] ** 2 + D[posy[1]]/D[t[0]] ** 2 + D[posz[1]]/D[t[0]] ** 2)")
Tall = knowledge.register_expr(f"[Sum: Particle] (1->Particle) |- {T}[1]")
V = knowledge.register_expr(f"(1->Spring) |- {k}[1] * (length[1] - {freel}[1])**2")
Vall = knowledge.register_expr(f"[Sum: Spring] (1->Spring) |- {V}[1]")

# %%
# check angular momentum and energy conserved
Lallv = knowledge.eval(Lall, expstruct)
plot_data(Lallv, 'L')
assert Lallv.is_conserved
Eallv = knowledge.eval(f"{Tall} + {Vall}", expstruct)
plot_data(Eallv, 'E')
assert Eallv.is_const
# %%
# check Newton the second
Fx1 = knowledge.eval(f"-{k}[2] * (1 - {freel}[2]/length[2]) * (posx[1])", expstruct)
ax1 = knowledge.eval(f"{ax}[1]", expstruct)
m1 = knowledge.eval(f"{mass}[1]", expstruct)
plot_data(Fx1-m1*ax1, 'eq1')
# %%

expstruct = oscillation_rot_2_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())
print(res.datastruct)
plot_datastruct(res.datastruct)
# %%
# check angular momentum and energy conserved
Lallv = knowledge.eval(Lall, expstruct)
plot_data(Lallv, 'L')
assert Lallv.is_conserved
Eallv = knowledge.eval(f"{Tall} + {Vall}", expstruct)
plot_data(Eallv, 'E')
assert Eallv.is_const
# %%
Fx1 = knowledge.eval(f"-{k}[3] * (1 - {freel}[3]/length[3]) * (posx[1]) + {k}[4] * (1 - {freel}[4]/length[4]) * (posx[2] - posx[1])", expstruct)
ax1 = knowledge.eval(f"{ax}[1]", expstruct)
m1 = knowledge.eval(f"{mass}[1]", expstruct)
eq1 = Fx1 - m1 * ax1
plot_data(eq1, 'eq1')
assert eq1.is_zero
# %%
expstruct = oscillation_rot_3_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())
print(res.datastruct)
plot_datastruct(res.datastruct)
# %%
# check angular momentum and energy conserved
Lallv = knowledge.eval(Lall, expstruct)
plot_data(Lallv, 'L')
assert Lallv.is_conserved
Eallv = knowledge.eval(f"{Tall} + {Vall}", expstruct)
plot_data(Eallv, 'E')
assert Eallv.is_const
# %%
Fx1 = knowledge.eval(f"-{k}[4] * (1 - {freel}[4]/length[4]) * (posx[1]) + {k}[5] * (1 - {freel}[5]/length[5]) * (posx[2] - posx[1])", expstruct)
ax1 = knowledge.eval(f"{ax}[1]", expstruct)
m1 = knowledge.eval(f"{mass}[1]", expstruct)
eq1 = Fx1 - m1 * ax1
plot_data(eq1, 'eq1')
assert eq1.is_zero

# %%
