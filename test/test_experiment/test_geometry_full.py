#%%
import os
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(work_dir)

from aiphy import Knowledge

from aiphy.experiment.basic import (
    motion0_3d_config,
    motion_3d_config,
    collision_y_config,
    collision_elastic_3body_config,
    collision_2d_config,
    collision_nh_config,
    circle_config,
    collision_circle_config,
    collision_circle_2_config
)
from aiphy.experiment.cosmos import (
    celestial_1x_config,
    celestial_1y_config,
    celestial_2_xy_config,
    celestial_3_config,
    celestial_4_config
)
from aiphy.experiment.ground_laboratory import (
    free_fall_config,
    projectile_config,
    oblique_projectile_x_config,
    oblique_projectile_y_config,
    projectile_oscillation_1_config,
    projectile_oscillation_2_config,
    oblique_projectile_config,
    vertical_projectile_config,
    gravity_config,
    hanging_spring_1_config,
    hanging_spring_2_config,
    hanging_spring_3_config,
    pendulum_1_config,
    pendulum_2_config,
    pendulum_3_config,
    pendulum_oscillation_1_config,
    pendulum_oscillation_2_config,
    slope_oscillation_1_config,
    slope_oscillation_2_config,
    pulley_1_config,
    pulley_2_config,
    pulley_3_config,
    pulley_oscillation_1_config,
    # pulley_oscillation_2_config,
    masspulley_oscillation_1_config,
    masspulley_oscillation_2_config
)
from aiphy.experiment.oscillation import (
    oscillation_2d_1_config,
    oscillation_2d_2_config,
    oscillation_2d_3_config,
    oscillation_r_3_config,
    oscillation_r_4_config,
    oscillation_rot_1_config,
    oscillation_rot_2_config,
    oscillation_rot_3_config,
    oscillation_x_1_config,
    oscillation_x_2_config,
    oscillation_x_3_config,
)

knowledge = Knowledge.default()
knowledge.register_expstruct('motion0_3d', motion0_3d_config())
knowledge.register_expstruct('motion_3d', motion_3d_config())
knowledge.register_expstruct('collision_y', collision_y_config())
knowledge.register_expstruct('collision_elastic_3body', collision_elastic_3body_config())
knowledge.register_expstruct('collision_2d', collision_2d_config())
knowledge.register_expstruct('collision_nh', collision_nh_config())
knowledge.register_expstruct('circle', circle_config())
knowledge.register_expstruct('collision_circle', collision_circle_config())
knowledge.register_expstruct('collision_circle_2', collision_circle_2_config())
knowledge.register_expstruct('celestial_1x', celestial_1x_config())
knowledge.register_expstruct('celestial_1y', celestial_1y_config())
knowledge.register_expstruct('celestial_2_xy', celestial_2_xy_config())
knowledge.register_expstruct('celestial_3', celestial_3_config())
knowledge.register_expstruct('celestial_4', celestial_4_config())
knowledge.register_expstruct('free_fall', free_fall_config())
knowledge.register_expstruct('projectile', projectile_config())
knowledge.register_expstruct('projectile_x', oblique_projectile_x_config())
knowledge.register_expstruct('projectile_y', oblique_projectile_y_config())
knowledge.register_expstruct('projectile_oscillation_1', projectile_oscillation_1_config())
knowledge.register_expstruct('projectile_oscillation_2', projectile_oscillation_2_config())
knowledge.register_expstruct('oblique_projectile', oblique_projectile_config())
knowledge.register_expstruct('vertical_projectile', vertical_projectile_config())
knowledge.register_expstruct('gravity', gravity_config())
knowledge.register_expstruct('hanging_spring_1', hanging_spring_1_config())
knowledge.register_expstruct('hanging_spring_2', hanging_spring_2_config())
knowledge.register_expstruct('hanging_spring_3', hanging_spring_3_config())
knowledge.register_expstruct('pendulum_1', pendulum_1_config())
knowledge.register_expstruct('pendulum_2', pendulum_2_config())
knowledge.register_expstruct('pendulum_3', pendulum_3_config())
knowledge.register_expstruct('pendulum_oscillation_1', pendulum_oscillation_1_config())
knowledge.register_expstruct('pendulum_oscillation_2', pendulum_oscillation_2_config())
knowledge.register_expstruct('slope_oscillation_1', slope_oscillation_1_config())
knowledge.register_expstruct('slope_oscillation_2', slope_oscillation_2_config())
knowledge.register_expstruct('pulley_1', pulley_1_config())
knowledge.register_expstruct('pulley_2', pulley_2_config())
knowledge.register_expstruct('pulley_3', pulley_3_config())
knowledge.register_expstruct('pulley_oscillation_1', pulley_oscillation_1_config())
# knowledge.register_expstruct('pulley_oscillation_2', pulley_oscillation_2_config())
knowledge.register_expstruct('masspulley_oscillation_1', masspulley_oscillation_1_config())
knowledge.register_expstruct('masspulley_oscillation_2', masspulley_oscillation_2_config())
knowledge.register_expstruct('oscillation_2d_1', oscillation_2d_1_config())
knowledge.register_expstruct('oscillation_2d_2', oscillation_2d_2_config())
knowledge.register_expstruct('oscillation_2d_3', oscillation_2d_3_config())
knowledge.register_expstruct('oscillation_r_3', oscillation_r_3_config())
knowledge.register_expstruct('oscillation_r_4', oscillation_r_4_config())
knowledge.register_expstruct('oscillation_rot_1', oscillation_rot_1_config())
knowledge.register_expstruct('oscillation_rot_2', oscillation_rot_2_config())
knowledge.register_expstruct('oscillation_rot_3', oscillation_rot_3_config())
knowledge.register_expstruct('oscillation_x_1', oscillation_x_1_config())
knowledge.register_expstruct('oscillation_x_2', oscillation_x_2_config())
knowledge.register_expstruct('oscillation_x_3', oscillation_x_3_config())


exp_list = list(knowledge.fetch_exps)
exp_list = sorted(exp_list)

for exp in exp_list:
    print(f'-------- {exp} --------')
    expstruct = knowledge.fetch_expstruct(exp)
    expstruct.random_settings()
    expstruct.print_geometry_info()
    assert knowledge.K.check_geometry_info(expstruct)

# %%
