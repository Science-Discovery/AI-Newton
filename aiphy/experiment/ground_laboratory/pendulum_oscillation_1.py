import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length
# from aiphy.experiment.ground_laboratory.pendulum_oscillation import create_eqs


pi = 3.1415926

exp_para = {
    'r1': default_parastructure(4.0, 6.0),
    'v1': default_parastructure(-2.0, 2.0),
    'theta1': default_parastructure(-pi/3, pi/3),
    'omega1': default_parastructure(-pi/3, pi/3),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ['s1']),
    (concept_t, ["clock"]),
]

g = 9.801234567

acs0 = [sp.sympify("g*cos(theta1) + omega1**2*r1 + k1*(l1-r1)/m1"),
        sp.sympify("-(g*sin(theta1) + 2*omega1*v1)/r1")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    r10 = exp_config.para('r1')
    v10 = exp_config.para('v1')
    omega10 = exp_config.para('omega1')
    theta10 = exp_config.para('theta1')
    l1 = exp_config.get_obj_para('s1', 'freel')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    m1 = exp_config.get_obj_para('o1', 'm')
    numeric = {'m1': m1, 'l1': l1, 'k1': k1, 'g': g}
    acs = [ac.subs(numeric) for ac in acs0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    y0 = [r10, theta10, v10, omega10]

    def f(y, t):
        r1, theta1, v1, omega1 = y
        numeric = {'r1': r1, 'theta1': theta1, 'v1': v1, 'omega1': omega1}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, omega1] + acs_n

    sol = odeint(f, y0, t)
    r1 = sol[:, 0]
    theta1 = sol[:, 1]
    posx1 = r1*np.sin(theta1)
    posz1 = -r1*np.cos(theta1)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), posx1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def pendulum_oscillation_1_config() -> ExpStructure:
    expconfig = ExpConfig("pendulum_oscillation_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]**2 + posx[o1]**2 - length[s1]**2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def pendulum_oscillation_1_test():
    expconfig = ExpConfig("pendulum_oscillation_1", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
