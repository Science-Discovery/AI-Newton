import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_length, concept_t

exp_para = {
    "z1": default_parastructure(-5.0, -3.0),
    "z2": default_parastructure(-5.0, -3.0),
    "posr1": default_parastructure(3.5, 5.5),
    "pz1": default_parastructure(-0.5, 0.5),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "p1": Objstructure.make_particle(3, 6),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_posx, ["p1"]),
    (concept_posy, ["p1"]),
    (concept_posz, ["p1"]),
    (concept_length, ["s1"]),
    (concept_t, ["clock"]),
]

acs0 = [sp.sympify("(2*k1*m2*(posr1 - pz1 - l1))/((m2+m1)*pm1+4*m1*m2) - g"),
        sp.sympify("(2*k1*m1*(posr1 - pz1 - l1))/((m2+m1)*pm1+4*m1*m2) - g"),
        sp.sympify("(k1*(m1+m2)*(posr1 - pz1 - l1))/((m2+m1)*pm1+4*m1*m2) - g")]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z10 = exp_config.para('z1')
    z20 = exp_config.para('z2')
    posr1 = exp_config.para('posr1')
    pz10 = exp_config.para('pz1')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    pm1 = exp_config.get_obj_para('p1', 'm')

    l0 = 2*pz10-z10-z20
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    numeric = {'k1': k1, 'l1': l1, 'posr1': posr1, 'pm1': pm1,
               'm1': m1, 'm2': m2,
               'l0': l0, 'g': g}
    acs = [ac.subs(numeric) for ac in acs0]
    input0 = [z10, z20, pz10, 0, 0, 0]

    def f(y, t):
        z1, z2, pz1, v1, v2, pv1 = y
        numeric = {'z1': z1, 'z2': z2, 'pz1': pz1}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, v2, pv1] + acs_n

    sol = odeint(f, input0, t)
    z1t = sol[:, 0]
    z2t = sol[:, 1]
    pz1t = sol[:, 2]
    len1t = posr1 - pz1t

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['p1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['p1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['p1']), pz1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), len1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct

def masspulley_oscillation_1_config() -> ExpStructure:
    expconfig = ExpConfig("masspulley_oscillation_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[p1] + length[s1]")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "2*posz[p1] - posz[o1] - posz[o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
