import numpy as np
from math import fabs
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t

exp_para = {
    "vx0": default_parastructure(1.0, 2.0),
    "vy0": default_parastructure(1.0, 2.0),
    "vz0": default_parastructure(1.0, 2.0),
    "x0": default_parastructure(1.0, 5.0),
    "y0": default_parastructure(1.0, 5.0),
    "z0": default_parastructure(1.0, 5.0),
}
obj_info = {
    "o1": Objstructure.make_particle(1, 2),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    vx0 = exp_config.para('vx0')
    x0 = exp_config.para('x0')
    vy0 = exp_config.para('vy0')
    y0 = exp_config.para('y0')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    x = vx0 * t + x0
    y = vy0 * t + y0
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def motion0_3d_config() -> ExpStructure:
    expconfig = ExpConfig("motion0_3d", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def motion0_3d_test():
    expconfig = ExpConfig("motion0_3d", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
