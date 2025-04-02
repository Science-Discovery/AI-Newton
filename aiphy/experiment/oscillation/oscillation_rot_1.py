import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length
# from aiphy.experiment.oscillation.free_oscillation import create_eqs


pi = 3.1415926

exp_para = {
    "r10": default_parastructure(5.0, 7.0),
    "v10": default_parastructure(-2.0, 2.0),
    "theta10": default_parastructure(0, pi/2),
    "omega10": default_parastructure(pi/4.0, pi/2.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_t, ["clock"]),
]


# eqs = create_eqs(n=1+1)
# acs0 = [sp.symbols('a'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, acs0)
# acs0 = [sp.simplify(ac.subs(sol)) for ac in acs0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('acs0 = [' + ', '.join(['sp.sympify("' + str(ac) + '")' for ac in acs0]) + ']')

acs0 = [sp.sympify("-k1*(1-l1/r1)*x1/m1"),
        sp.sympify("-k1*(1-l1/r1)*y1/m1"),]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    r10 = exp_config.para('r10')
    v10 = exp_config.para('v10')
    theta10 = exp_config.para('theta10')
    omega10 = exp_config.para('omega10')
    m1 = exp_config.get_obj_para('o1', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    numeric = {'k1': k1, 'l1': l1, 'm1': m1}
    acs = [ac.subs(numeric) for ac in acs0]

    x10 = r10*np.cos(theta10)
    y10 = r10*np.sin(theta10)
    vx10 = v10*np.cos(theta10)-omega10*r10*np.sin(theta10)
    vy10 = v10*np.sin(theta10)+omega10*r10*np.cos(theta10)
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, y10, vx10, vy10]

    def f(y, t):
        x1, y1, vx1, vy1 = y
        r1 = np.sqrt(x1**2+y1**2)
        numeric = {'x1': x1, 'y1': y1, 'vx1': vx1, 'vy1': vy1, 'r1': r1}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vy1] + acs_n

    sol = odeint(f, y0, t)

    x1t = sol[:, 0]
    y1t = sol[:, 1]
    r1t = np.sqrt(x1t**2 + y1t**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def oscillation_rot_1_config() -> ExpStructure:
    expconfig = ExpConfig("oscillation_rot_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "length[s1] ** 2 - posx[o1] ** 2 - posy[o1] ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def oscillation_rot_1_test():
    expconfig = ExpConfig("oscillation_rot_1", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
