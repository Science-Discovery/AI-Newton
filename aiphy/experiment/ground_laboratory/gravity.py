import numpy as np
from math import fabs
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length

exp_para = {}
obj_info = {
    "s1": Objstructure.make_spring(2, 2, 9.0, 11.0),
    "o1": Objstructure.make_particle(2.0, 3.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_length, ["s1"]),
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    m = exp_config.get_obj_para('o1', 'm')
    freel = exp_config.get_obj_para('s1', 'freel')
    k = exp_config.get_obj_para('s1', 'thickness') ** 3
    g = 9.801234567
    length = freel + m / k * g
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    x = np.zeros(t_num)
    y = np.zeros(t_num)
    z = np.zeros(t_num)
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_length, ['s1']), length + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def gravity_config() -> ExpStructure:
    expconfig = ExpConfig("gravity", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1] + length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def gravity_test():
    expconfig = ExpConfig("gravity", 1, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
