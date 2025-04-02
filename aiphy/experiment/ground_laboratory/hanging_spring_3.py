import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_length, concept_t, concept_posy, concept_posx, concept_posz, concept_dist

exp_para = {
    "z0": default_parastructure(10, 12),  # 弹簧固定的上端点的位置
    "z10": default_parastructure(-5, 5),  # 弹簧初始的下端点的位置
    "z20": default_parastructure(-15, -10),  # 弹簧初始的下端点的位置
    "z30": default_parastructure(-25, -20),  # 弹簧初始的下端点的位置
}

obj_info = {
    "o1": Objstructure.make_particle(5, 6),
    "s1": Objstructure.make_spring(1.7, 1.9, 5, 10),
    "o2": Objstructure.make_particle(5, 6),
    "s2": Objstructure.make_spring(1.7, 1.9, 5, 10),
    "o3": Objstructure.make_particle(5, 6),
    "s3": Objstructure.make_spring(1.7, 1.9, 5, 10),
    "clock": Objstructure.clock()
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_length, ["s2"]),
    (concept_posx, ["o3"]),
    (concept_posy, ["o3"]),
    (concept_posz, ["o3"]),
    (concept_length, ["s3"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_dist, ["o1", "o3"]),
    (concept_dist, ["o3", "o1"]),
    (concept_dist, ["o2", "o3"]),
    (concept_dist, ["o3", "o2"]),
    (concept_t, ["clock"]),
]


acs0 = [sp.sympify("-g+(k1*(z0-z1-l1)-k2*(z1-z2-l2))/m1"), sp.sympify("-g+(k2*(z1-z2-l2)-k3*(z2-z3-l3))/m2"), sp.sympify("-g+k3*(z2-z3-l3)/m3")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z0 = exp_config.para('z0')
    z10 = exp_config.para('z10')
    z20 = exp_config.para('z20')
    z30 = exp_config.para('z30')
    l1 = exp_config.get_obj_para('s1', 'freel')
    k1 = exp_config.get_obj_para('s1', 'thickness') ** 3
    m1 = exp_config.get_obj_para('o1', 'm')
    l2 = exp_config.get_obj_para('s2', 'freel')
    k2 = exp_config.get_obj_para('s2', 'thickness') ** 3
    m2 = exp_config.get_obj_para('o2', 'm')
    l3 = exp_config.get_obj_para('s3', 'freel')
    k3 = exp_config.get_obj_para('s3', 'thickness') ** 3
    m3 = exp_config.get_obj_para('o3', 'm')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    numeric = {'k1': k1, 'l1': l1, 'k2': k2, 'l2': l2, 'k3': k3, 'l3': l3,
               'm1': m1, 'm2': m2, 'm3': m3, 'g': g, "z0": z0}
    acs = [ac.subs(numeric) for ac in acs0]

    y0 = [z10, z20, z30, 0.0, 0.0, 0.0]

    def f(y, t):
        z1, z2, z3, v1, v2, v3 = y
        numeric = {'z1': z1, 'v1': v1, 'z2': z2, 'v2': v2, 'z3': z3, 'v3': v3}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, v2, v3] + acs_n

    sol = odeint(f, y0, t)
    posz1 = sol[:, 0]
    posz2 = sol[:, 1]
    posz3 = sol[:, 2]

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o2"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o2"]), posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ["o3"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o3"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o3"]), posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), z0-posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s2"]), posz1-posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s3"]), posz2-posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o1", "o2"]), posz1-posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o2", "o1"]), posz1-posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o1", "o3"]), posz1-posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o3", "o1"]), posz1-posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o2", "o3"]), posz2-posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ["o3", "o2"]), posz2-posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct


def hanging_spring_3_config() -> ExpStructure:
    expconfig = ExpConfig("hanging_spring_3", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1] + length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2] - posz[o1] + length[s2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o3] - posz[o2] + length[s3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - posz[o1] + posz[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - dist[o3, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - posz[o1] + posz[o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - dist[o3, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - posz[o2] + posz[o3]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
