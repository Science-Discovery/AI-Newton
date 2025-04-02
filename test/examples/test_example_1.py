import os
from aiphy import Theorist
from aiphy.experiment.basic import (
    motion0_3d_config, motion_3d_config,
    collision_elastic_3body_config, collision_y_config,
    collision_2d_config, collision_nh_config,
)
from aiphy.experiment.ground_laboratory import (
    gravity_config, free_fall_config,
    projectile_config, oblique_projectile_x_config, oblique_projectile_y_config,
    oblique_projectile_config, vertical_projectile_config,
    hanging_spring_1_config, hanging_spring_2_config, hanging_spring_3_config,
    pendulum_oscillation_1_config, pendulum_oscillation_2_config,
    slope_oscillation_1_config, slope_oscillation_2_config,
    projectile_oscillation_1_config, projectile_oscillation_2_config,
)
from aiphy.experiment.oscillation import (
    oscillation_x_1_config, oscillation_x_2_config, oscillation_x_3_config,
    oscillation_2d_1_config, oscillation_2d_2_config, oscillation_2d_3_config,
    oscillation_r_3_config, oscillation_r_4_config,
    oscillation_rot_1_config, oscillation_rot_2_config, oscillation_rot_3_config,
)
from aiphy.experiment.cosmos import (
    celestial_2_xy_config, celestial_2_xz_config, celestial_2_yz_config,
    celestial_3_config, celestial_4_config,
)

EXAMPLE_ID = 1

NUMBER_OF_ITERATIONS = 1200

NUM_THREADS = 32
INIT_TIME_LIMIT = 5
MUL = 3
MAX_ACTIONS = 7
BIAS = 5
ID = 1

EXAMPLE_PATH = f"data/test_cases/example_{EXAMPLE_ID}"

FILE_NAME = f"example_{EXAMPLE_ID}-{INIT_TIME_LIMIT}m{MUL}a{MAX_ACTIONS}p{BIAS}".replace(".", "-")

# Iterately create the path
FILE_PATH_PARENT = EXAMPLE_PATH + "/" + FILE_NAME
if ID is None:
    FILE_PATH = FILE_PATH_PARENT + '/' + FILE_NAME
else:
    FILE_PATH = FILE_PATH_PARENT + '/' + FILE_NAME + f"-{ID}"
os.makedirs(FILE_PATH_PARENT, exist_ok=True)


def work():
    theorist = Theorist(init_time_limit=INIT_TIME_LIMIT, num_threads=NUM_THREADS)
    # theorist = Theorist.read_from_file(FILE_PATH, num_threads=NUM_THREADS)
    theorist.register_experiment(motion0_3d_config())
    theorist.register_experiment(motion_3d_config())
    theorist.register_experiment(collision_elastic_3body_config())
    theorist.register_experiment(collision_y_config())
    theorist.register_experiment(collision_2d_config())
    theorist.register_experiment(collision_nh_config())

    theorist.register_experiment(gravity_config())
    theorist.register_experiment(free_fall_config())
    theorist.register_experiment(projectile_config())
    theorist.register_experiment(oblique_projectile_config())
    theorist.register_experiment(oblique_projectile_x_config())
    theorist.register_experiment(oblique_projectile_y_config())
    theorist.register_experiment(vertical_projectile_config())
    theorist.register_experiment(hanging_spring_1_config())
    theorist.register_experiment(hanging_spring_2_config())
    theorist.register_experiment(hanging_spring_3_config())
    theorist.register_experiment(pendulum_oscillation_1_config())
    theorist.register_experiment(pendulum_oscillation_2_config())
    theorist.register_experiment(slope_oscillation_1_config())
    theorist.register_experiment(slope_oscillation_2_config())
    theorist.register_experiment(projectile_oscillation_1_config())
    theorist.register_experiment(projectile_oscillation_2_config())

    theorist.register_experiment(oscillation_x_1_config())
    theorist.register_experiment(oscillation_x_2_config())
    theorist.register_experiment(oscillation_x_3_config())
    theorist.register_experiment(oscillation_2d_1_config())
    theorist.register_experiment(oscillation_2d_2_config())
    theorist.register_experiment(oscillation_2d_3_config())
    theorist.register_experiment(oscillation_r_3_config())
    theorist.register_experiment(oscillation_r_4_config())
    theorist.register_experiment(oscillation_rot_1_config())
    theorist.register_experiment(oscillation_rot_2_config())
    theorist.register_experiment(oscillation_rot_3_config())

    theorist.register_experiment(celestial_2_xy_config())
    theorist.register_experiment(celestial_2_xz_config())
    theorist.register_experiment(celestial_2_yz_config())
    theorist.register_experiment(celestial_3_config())
    theorist.register_experiment(celestial_4_config())

    theorist.save_to_file(FILE_PATH)

    theorist.perform_discovery(cpt_lim_mul=MUL,
                               max_actions=MAX_ACTIONS,
                               max_cpt_lim=3600.,
                               cpt_lim_bias=BIAS,
                               max_epochs=NUMBER_OF_ITERATIONS,
                               save_path=FILE_PATH)


work()

"""
EXAMPLE OUTPUT
"""
