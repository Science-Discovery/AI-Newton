#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType
from aiphy.experiment.oscillation.old_x.oscillation_1s2o import oscillation_1s2o_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt


#%%
knowledge = Knowledge.default()
expstruct = oscillation_1s2o_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())

plot_datastruct(res.datastruct)
# expdata = knowledge.eval("posx[2]", expstruct)
# plot_data('x2(t)', expdata)

# %%
