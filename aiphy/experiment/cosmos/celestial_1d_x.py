import numpy as np
import sympy as sp
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
from scipy.integrate import odeint
# from aiphy.experiment.cosmos.celestial_2d import create_eqs


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-4e-2, -3e-2),
    "x1": default_parastructure(3e-2, 4e-2),
    "v10": default_parastructure(-6e-5, -4e-5),
    "v20": default_parastructure(4e-5, 6e-5),
}

obj_info = {
    "o1": Objstructure.make_particle(5e5, 6e5),
    "o2": Objstructure.make_particle(5e5, 6e5),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_t, ["clock"]),
]

G = 6.6e-11


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x10 = exp_config.para('x0')
    x20 = exp_config.para('x1')
    v10 = exp_config.para('v10')
    v20 = exp_config.para('v20')

    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    x0 = [x10, x20, v10, v20]

    def f(x, t):
        x1, x2, vx1, vx2 = x
        k = G / (x1 - x2) ** 2
        return [vx1, vx2, k*m2 , -k*m1]
    sol = odeint(f, x0, t)
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    dist = abs(x1 - x2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), dist + np.random.normal(0, error, t_num))
    return data_struct


def celestial_1x_config() -> ExpStructure:
    expconfig = ExpConfig("celestial_1x", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] + (posx[o1] - posx[o2])")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
