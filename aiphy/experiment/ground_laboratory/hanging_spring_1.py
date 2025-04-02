import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_length, concept_t, concept_posy, concept_posx, concept_posz

exp_para = {
    "zl": default_parastructure(10, 12),  # 弹簧固定的上端点的位置
    "zr": default_parastructure(-5, 3), # 弹簧初始的下端点的位置
}

obj_info = {
    "o1": Objstructure.make_particle(5, 6),
    "s1": Objstructure.make_spring(1.7, 1.9, 13, 17),
    "clock": Objstructure.clock()
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    zl = exp_config.para('zl')
    zr = exp_config.para('zr')
    l = exp_config.get_obj_para('s1', 'freel')
    k = exp_config.get_obj_para('s1', 'thickness') ** 3
    m = exp_config.get_obj_para('o1', 'm')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    z = (g * m) / k - (g * m + k * (l + zr - zl)) / k * np.cos(np.sqrt(k / m) * t)
    posz = zl - l - z
    length = l + z

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ["o1"]), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ["s1"]), length + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return data_struct

def hanging_spring_1_config() -> ExpStructure:
    expconfig = ExpConfig("hanging_spring_1", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1] + length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
