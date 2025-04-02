import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length, concept_dist
# from aiphy.experiment.oscillation.free_oscillation import create_eqs


pi = 3.1415926

exp_para = {
    "x1": default_parastructure(-35, -25),
    "x2": default_parastructure(25, 35),
    "v1": default_parastructure(-20, 20),
    "v2": default_parastructure(-20, 20),
    "theta": default_parastructure(0, pi)
}
obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "o2": Objstructure.make_particle(3.0, 6.0),
    "s1": Objstructure.make_spring(1.5, 2.5, 40, 60),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o1"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o1"]),
    (concept_posz, ["o2"]),
    (concept_length, ["s1"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_t, ["clock"]),
]


# eqs = create_eqs(n=1+1)
# acs0 = [sp.symbols('a'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, acs0)
# acs0 = [sp.simplify(ac.subs(sol)) for ac in acs0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('acs0 = [' + ', '.join(['sp.sympify("' + str(ac) + '")' for ac in acs0]) + ']')

acs0 = [sp.sympify("k1*(-l1 - x1 + x2)/m1"), sp.sympify("k1*(l1 + x1 - x2)/m2")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    theta = exp_config.para('theta')
    v10 = exp_config.para('v1')
    x10 = exp_config.para('x1')
    m1 = exp_config.get_obj_para('o1', 'm')
    v20 = exp_config.para('v2')
    x20 = exp_config.para('x2')
    m2 = exp_config.get_obj_para('o2', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    numeric = {'k1': k1, 'l1': l1, 'm1': m1, 'm2': m2}
    acs = [ac.subs(numeric) for ac in acs0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, x20, v10, v20]

    def f(y, t):
        x1, x2, v1, v2 = y
        numeric = {'x1': x1, 'v1': v1, 'x2': x2, 'v2': v2}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, v2] + acs_n

    sol = odeint(f, y0, t)

    x1t = sol[:, 0]
    x2t = sol[:, 1]

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x1t*np.cos(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2t*np.cos(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), x1t*np.sin(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), x2t*np.sin(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), x2t-x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), x2t-x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), x2t-x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def oscillation_2d_1_config() -> ExpStructure:
    expconfig = ExpConfig("oscillation_2d_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def oscillation_2d_1_test():
    expconfig = ExpConfig("oscillation_2d_1", 1, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
