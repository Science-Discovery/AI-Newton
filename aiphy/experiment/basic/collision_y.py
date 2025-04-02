import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
from aiphy.core import do_collision

exp_para = {
    "y1": default_parastructure(-4.0, -3.0),
    "v1": default_parastructure(3.0, 5.0),
    "y2": default_parastructure(3.0, 4.0),
    "v2": default_parastructure(-5.0, -3.0),
}
obj_info = {
    "MPa": Objstructure.make_particle(2.0, 10.0),
    "MPb": Objstructure.make_particle(2.0, 10.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["MPa"]),
    (concept_posx, ["MPb"]),
    (concept_posy, ["MPa"]),
    (concept_posy, ["MPb"]),
    (concept_posz, ["MPa"]),
    (concept_posz, ["MPb"]),
    (concept_dist, ["MPa", "MPb"]),
    (concept_dist, ["MPb", "MPa"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    y1 = exp_config.para("y1")
    y2 = exp_config.para("y2")
    v1 = exp_config.para("v1")
    v2 = exp_config.para("v2")
    m1 = exp_config.get_obj_para("MPa", "m")
    m2 = exp_config.get_obj_para("MPb", "m")
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    t_collision = (y2 - y1) / (v1 - v2)
    assert t_collision > 0.0
    assert t_collision < t_end
    vn1, vn2 = do_collision(m1, m2, v1, v2)
    data_y1 = np.array([
        (y1 + v1 * i) if i < t_collision else (y1 + v1 * t_collision + vn1 * (i - t_collision))
        for i in t
    ])
    data_y2 = np.array([
        (y2 + v2 * i) if i < t_collision else (y2 + v2 * t_collision + vn2 * (i - t_collision))
        for i in t
    ])
    data_dist = np.abs(data_y2 - data_y1)
    x = np.zeros(t_num)
    z = np.zeros(t_num)
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['MPa']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['MPb']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPa']), data_y1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPb']), data_y2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPa']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPb']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPa', 'MPb']), data_dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPb', 'MPa']), data_dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def collision_y_config() -> ExpStructure:
    expconfig = ExpConfig("collision_y", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPa, MPb] - dist[MPb, MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPa, MPb] ** 2 - (posy[MPa] - posy[MPb]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[MPb]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[MPb]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def collision_y_test():
    expconfig = ExpConfig("collsion_y", 1, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
