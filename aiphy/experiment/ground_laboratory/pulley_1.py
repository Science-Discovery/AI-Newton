import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t

exp_para = {
    "z1": default_parastructure(-5.0, 5.0),
    "z2": default_parastructure(-5.0, 5.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z1 = exp_config.para('z1')
    z2 = exp_config.para('z2')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    a1 = (m2-m1)*g/(m1+m2)
    a2 = (m1-m2)*g/(m1+m2)
    z1t = z1 + a1*t**2 / 2
    z2t = z2 + a2*t**2 / 2
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct

def pulley_1_config() -> ExpStructure:
    expconfig = ExpConfig("pulley_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1] + posz[o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
