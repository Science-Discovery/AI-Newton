#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType
from aiphy.experiment.basic import motion0_3d_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt

#%%
knowledge = Knowledge.default()
expstruct = motion0_3d_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())

plot_datastruct(res.datastruct)
# %%
vx = knowledge.eval("D[posx[1]]/D[t[0]]", expstruct)
ax = knowledge.eval("posx[1]''", expstruct)
assert vx.is_const
assert ax.is_zero
vy = knowledge.eval("D[posy[1]]/D[t[0]]", expstruct)
ay = knowledge.eval("posy[1]''", expstruct)
assert vy.is_const
assert ay.is_zero
vz = knowledge.eval("D[posz[1]]/D[t[0]]", expstruct)
assert vz.is_zero
# %%
