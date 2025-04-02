import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
from aiphy.core import do_collision

exp_para = {
    "x1": default_parastructure(-5.0, -4.8),
    "v1": default_parastructure(4.2, 4.5),
    "x2": default_parastructure(-0.1, 0.1),
    "v2": default_parastructure(2.2, 2.5),
    "x3": default_parastructure(4.5, 5.0),
    "v3": default_parastructure(-4.0, -3.5),
}
obj_info = {
    "MPa": Objstructure.make_particle(2.5, 2.75),
    "MPb": Objstructure.make_particle(2.75, 3.0),
    "MPc": Objstructure.make_particle(2.5, 3.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["MPa"]),
    (concept_posx, ["MPb"]),
    (concept_posx, ["MPc"]),
    (concept_posy, ["MPa"]),
    (concept_posy, ["MPb"]),
    (concept_posy, ["MPc"]),
    (concept_posz, ["MPa"]),
    (concept_posz, ["MPb"]),
    (concept_posz, ["MPc"]),
    (concept_dist, ["MPa", "MPb"]),
    (concept_dist, ["MPb", "MPa"]),
    (concept_dist, ["MPa", "MPc"]),
    (concept_dist, ["MPc", "MPa"]),
    (concept_dist, ["MPb", "MPc"]),
    (concept_dist, ["MPc", "MPb"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x1 = exp_config.para("x1")
    x2 = exp_config.para("x2")
    x3 = exp_config.para("x3")
    v1 = exp_config.para("v1")
    v2 = exp_config.para("v2")
    v3 = exp_config.para("v3")
    m1 = exp_config.get_obj_para("MPa", "m")
    m2 = exp_config.get_obj_para("MPb", "m")
    m3 = exp_config.get_obj_para("MPc", "m")
    step = t_end / t_num
    t = np.arange(0, t_end, step)

    # Collision between 2 and 3
    t_collision_23 = (x3 - x2) / (v2 - v3)
    t_collision_12ref = (x2 - x1) / (v2 - v1)
    if t_collision_23 < 0.0 or t_collision_23 > t_end:
        raise ValueError("No collision between 2 and 3")
    if t_collision_12ref > 0.0 and t_collision_12ref < t_collision_23:
        raise ValueError("Collision between 1 and 2 before 2 and 3")
    v2_1, v3_1 = do_collision(m2, m3, v2, v3)
    if v2_1 > 0.0 or v3_1 < 0.0:
        raise ValueError("Wrong collision between 2 and 3")

    # Collision between 1 and 2
    t_collision_12 = (x2 + v2 * t_collision_23 - (x1 + v1 * t_collision_23)) / (v1 - v2_1) + t_collision_23
    if t_collision_12 <= t_collision_23 or t_collision_12 > t_end:
        raise ValueError("No collision between 1 and 2")
    v1_2, v2_2 = do_collision(m1, m2, v1, v2_1)
    if v1_2 > 0.0:
        raise ValueError("Wrong collision between 1 and 2")

    # Collision between 2 and 3
    if v2_2 > v3_1:
        t_collision_23_again = (x3 + v3 * (t_collision_23) + v3_1 * (t_collision_12 - t_collision_23)
                                - (x2 + v2 * t_collision_23 + v2_1 * (t_collision_12 - t_collision_23))) / (v2_2 - v3_1) + t_collision_12
        if t_collision_23_again >= t_collision_12 and t_collision_23_again < t_end:
            raise ValueError("Wrong collision between 2 and 3")

    data_x2 = np.array([
        (x2 + v2 * i) if i < t_collision_23 else
        (x2 + v2 * t_collision_23 + v2_1 * (i - t_collision_23)) if (i >= t_collision_23 and i < t_collision_12) else
        (x2 + v2 * t_collision_23 + v2_1 * (t_collision_12 - t_collision_23) + v2_2 * (i - t_collision_12))
        for i in t
    ])
    data_x1 = np.array([
        (x1 + v1 * i) if i < t_collision_12 else
        (x1 + v1 * t_collision_12 + v1_2 * (i - t_collision_12))
        for i in t
    ])
    data_x3 = np.array([
        (x3 + v3 * i) if i < t_collision_23 else
        (x3 + v3 * t_collision_23 + v3_1 * (i - t_collision_23))
        for i in t
    ])

    y = np.zeros(t_num)
    z = np.zeros(t_num)
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['MPa']), data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['MPb']), data_x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['MPc']), data_x3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPa']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPb']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPc']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPa']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPb']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPc']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPa', 'MPb']), data_x2-data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPb', 'MPa']), data_x2-data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPa', 'MPc']), data_x3-data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPc', 'MPa']), data_x3-data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPb', 'MPc']), data_x3-data_x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPc', 'MPb']), data_x3-data_x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))

    return data_struct


def collision_elastic_3body_config() -> ExpStructure:
    expconfig = ExpConfig("collision_elastic_3body", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[MPb] - posx[MPa] - dist[MPb, MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPb, MPa] - dist[MPa, MPb]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[MPc] - posx[MPa] - dist[MPa, MPc]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPa, MPc] - dist[MPc, MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[MPc] - posx[MPb] - dist[MPb, MPc]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPb, MPc] - dist[MPc, MPb]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[MPb]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[MPc]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def collision_elastic_3body_test():
    expconfig = ExpConfig("collision_elastic_3body", 1, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
