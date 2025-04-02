#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge, MeasureType
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
from aiphy.experiment.ground_laboratory import pendulum_1_config, pendulum_2_config

# %%
knowledge = Knowledge.default()
expstruct = pendulum_1_config()
knowledge.register_expstruct('pendulum_1', expstruct)
for _ in range(10):
    expstruct.random_settings()
    data = knowledge.eval("D[posx[1]' ** 2 + posz[1]' ** 2]/D[posz[1]]", expstruct)
    if not data.is_const:
        plot_data(knowledge.eval("posx[1]", expstruct), "posx")
        plot_data(data, "D[posx[1]' ** 2 + posz[1]' ** 2]/D[posz[1]]")
    assert data.is_const
# %%
expstruct = pendulum_2_config()
knowledge.register_expstruct('pendulum_2', expstruct)
total_count = 0
for i in range(10):
    expstruct.random_settings()
    # plot_data('posx[1]', knowledge.eval("posx[1]", expstruct))
    # plot_data('posz[1]', knowledge.eval("posz[1]", expstruct))
    vsq1 = knowledge.eval("posx[1]' ** 2 + posz[1]' ** 2", expstruct)
    gh1 = (2 * 9.801234567) * knowledge.eval("posz[1]", expstruct)
    e1 = vsq1 + gh1
    vsq2 = knowledge.eval("posx[2]' ** 2 + posz[2]' ** 2", expstruct)
    gh2 = (2 * 9.801234567) * knowledge.eval("posz[2]", expstruct)
    e2 = vsq2 + gh2
    de1vsde2 = e1.__diff__(e2)
    if not de1vsde2.is_const:
        plot_data(e1, "e1")
        plot_data(e2, "e2")
        plot_data(de1vsde2, "D[e1]/D[e2]")
        print(f'{i}th test failed')
    else:
        print(f'{i}th test passed')
        total_count += 1
assert total_count >= 8
# %%
plot_data(de1vsde2, "D[e1]/D[e2]")
# %%
