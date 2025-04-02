import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t


pi = 3.1415926
exp_para = {
    "l": default_parastructure(5.0, 7.0),
    "theta0": default_parastructure(0.0, 2*pi),
    "omega0": default_parastructure(1.0, 2.0),
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
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    l = exp_config.para('l')
    theta0 = exp_config.para('theta0')
    omega0 = exp_config.para('omega0')
    x = l*np.sin(omega0*t+theta0)
    y = l*np.cos(omega0*t+theta0)
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    return data_struct


def circle_config() -> ExpStructure:
    expconfig = ExpConfig("circle", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posx[o1] ** 2 + posy[o1] ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
