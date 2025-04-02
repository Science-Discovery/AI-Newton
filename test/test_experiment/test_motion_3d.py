#%%
import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType
from aiphy.experiment.basic import motion_3d_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt

#%%
knowledge = Knowledge.default()
expstruct = motion_3d_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())

plot_datastruct(res.datastruct)
expdata = knowledge.eval("posy[1]", expstruct)
plot_data(expdata, 'y')

# %%
posvy = knowledge.eval("posy[1]'", expstruct)
posay = knowledge.eval("posy[1]''", expstruct)
plot_data(posvy, 'posvy')
plot_data(posay, 'posay')
assert posvy.is_normal
assert posay.is_const
# %%
posvx = knowledge.eval("posx[1]'", expstruct)
posax = knowledge.eval("posx[1]''", expstruct)
posvz = knowledge.eval("posz[1]'", expstruct)
posaz = knowledge.eval("posz[1]''", expstruct)
assert posvx.is_normal
assert posax.is_const
assert posvz.is_normal
assert posaz.is_const
dxdy = knowledge.eval("posx[1]'/posy[1]'", expstruct)
dvxdvy = knowledge.eval("posx[1]''/posy[1]''", expstruct)
assert dxdy.is_const
assert dvxdvy.is_const
assert (dxdy - dvxdvy).is_zero
dxdy = knowledge.eval("posx[1]'/posz[1]'", expstruct)
dvxdvy = knowledge.eval("posx[1]''/posz[1]''", expstruct)
assert dxdy.is_const
assert dvxdvy.is_const
assert (dxdy - dvxdvy).is_zero

# %%
