import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_length, concept_t

exp_para = {
    "z1": default_parastructure(-5.0, -3.0),
    "z2": default_parastructure(-5.0, -3.0),
    "z3": default_parastructure(-5.0, -3.0),
    "posr1": default_parastructure(3.5, 5.5),
    "pz1": default_parastructure(-0.5, 0.5),
    "posr2": default_parastructure(3.5, 5.5),
    "pz2": default_parastructure(-0.5, 0.5),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "o3": Objstructure.make_particle(3, 6),
    "p1": Objstructure.make_particle(3, 6),
    "p2": Objstructure.make_particle(3, 6),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "s2": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_posx, ["o3"]),
    (concept_posy, ["o3"]),
    (concept_posz, ["o3"]),
    (concept_posx, ["p1"]),
    (concept_posy, ["p1"]),
    (concept_posz, ["p1"]),
    (concept_posx, ["p2"]),
    (concept_posy, ["p2"]),
    (concept_posz, ["p2"]),
    (concept_length, ["s1"]),
    (concept_length, ["s2"]),
    (concept_t, ["clock"]),
]

acs0 = [sp.sympify("2*m2*m3*(k1*pm2*(posr1-pz1-l1)+k2*pm1*(posr2-pz2-l2))/((4*m1*m3+m1*m2+m2*m3)*pm1*pm2 + 4*m1*m2*m3*(pm1 + pm2)) - g"),
        sp.sympify("4*m1*m3*(k1*pm2*(posr1-pz1-l1)+k2*pm1*(posr2-pz2-l2))/((4*m1*m3+m1*m2+m2*m3)*pm1*pm2 + 4*m1*m2*m3*(pm1 + pm2)) - g"),
        sp.sympify("2*m1*m2*(k1*pm2*(posr1-pz1-l1)+k2*pm1*(posr2-pz2-l2))/((4*m1*m3+m1*m2+m2*m3)*pm1*pm2 + 4*m1*m2*m3*(pm1 + pm2)) - g"),
        sp.sympify("(k1*(4*m1*m3*(pm2+m2)+(m1+m3)*m2*pm2)*(posr1-pz1-l1)-4*k2*m1*m2*m3*(posr2-pz2-l2))/((4*m1*m3+m1*m2+m2*m3)*pm1*pm2 + 4*m1*m2*m3*(pm1 + pm2)) - g"),
        sp.sympify("(k2*(4*m1*m3*(pm1+m2)+(m1+m3)*m2*pm1)*(posr2-pz2-l2)-4*k1*m1*m2*m3*(posr1-pz1-l1))/((4*m1*m3+m1*m2+m2*m3)*pm1*pm2 + 4*m1*m2*m3*(pm1 + pm2)) - g"),
        ]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    z10 = exp_config.para('z1')
    z20 = exp_config.para('z2')
    z30 = exp_config.para('z3')
    posr1 = exp_config.para('posr1')
    pz10 = exp_config.para('pz1')
    posr2 = exp_config.para('posr2')
    pz20 = exp_config.para('pz2')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    k2 = exp_config.get_obj_para('s2', 'thickness')**3
    l2 = exp_config.get_obj_para('s2', 'freel')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    m3 = exp_config.get_obj_para('o3', 'm')
    pm1 = exp_config.get_obj_para('p1', 'm')
    pm2 = exp_config.get_obj_para('p2', 'm')

    l0 = 2*pz10+2*pz20-z10-2*z20-z30
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    numeric = {'k1': k1, 'l1': l1, 'posr1': posr1, 'pm1': pm1,
               'k2': k2, 'l2': l2, 'posr2': posr2, 'pm2': pm2,
               'm1': m1, 'm2': m2, 'm3': m3,
               'l0': l0, 'g': g}
    acs = [ac.subs(numeric) for ac in acs0]
    input0 = [z10, z20, z30, pz10, pz20, 0, 0, 0, 0, 0]

    def f(y, t):
        z1, z2, z3, pz1, pz2, v1, v2, v3, pv1, pv2 = y
        numeric = {'z1': z1, 'z2': z2, 'z3': z3, 'pz1': pz1, 'pz2': pz2}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, v2, v3, pv1, pv2] + acs_n

    sol = odeint(f, input0, t)
    z1t = sol[:, 0]
    z2t = sol[:, 1]
    z3t = sol[:, 2]
    pz1t = sol[:, 3]
    pz2t = sol[:, 4]
    len1t = posr1 - pz1t
    len2t = posr2 - pz2t

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o3']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o3']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o3']), z3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['p1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['p1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['p1']), pz1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['p2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['p2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['p2']), pz2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), len1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s2']), len2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct

def masspulley_oscillation_2_config() -> ExpStructure:
    expconfig = ExpConfig("masspulley_oscillation_2", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[p1] + length[s1]")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[p2] + length[s2]")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "2*posz[p1] + 2*posz[p2] - posz[o1] - 2*posz[o2] - posz[o3]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
