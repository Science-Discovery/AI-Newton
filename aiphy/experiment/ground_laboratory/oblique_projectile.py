import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t


exp_para = {
    "x0": default_parastructure(-5.0, 5.0),
    "vx0": default_parastructure(1.0, 2.0),
    "y0": default_parastructure(-5.0, 5.0),
    "vy0": default_parastructure(1.0, 2.0),
    "z0": default_parastructure(-5.0, 5.0),
    "vz0": default_parastructure(1.0, 2.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "clock": Objstructure.clock()
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para('x0')
    vx0 = exp_config.para('vx0')
    y0 = exp_config.para('y0')
    vy0 = exp_config.para('vy0')
    z0 = exp_config.para('z0')
    vz0 = exp_config.para('vz0')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567
    x = x0 + vx0 * t
    y = y0 + vy0 * t
    z = z0 + vz0 * t - 1 / 2 * g * t ** 2
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def oblique_projectile_config() -> ExpStructure:
    expconfig = ExpConfig("oblique_projectile", 3, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
