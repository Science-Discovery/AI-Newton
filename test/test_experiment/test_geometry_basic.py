#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata, ExpData, NormalData

from aiphy.main import Theorist
from aiphy.experiment.cosmos import celestial_2_xy_config 
from aiphy.experiment.basic import collision_elastic_3body_config, motion0_3d_config, motion_3d_config, collision_y_config, collision_2d_config, collision_nh_config
# %%
theorist = Theorist()
theorist.register_experiment(celestial_2_xy_config())
theorist.register_experiment(collision_elastic_3body_config())
theorist.register_experiment(motion0_3d_config())
theorist.register_experiment(motion_3d_config())
theorist.register_experiment(collision_y_config())
theorist.register_experiment(collision_2d_config())
theorist.register_experiment(collision_nh_config())

exps = theorist.knowledge.fetch_exps
for exp in exps:
    print(f'-------- {exp} --------')
    expstruct = theorist.knowledge.fetch_expstruct(exp)
    expstruct.random_settings()
    expstruct.print_geometry_info()
    assert theorist.knowledge.K.check_geometry_info(expstruct)

# %%
