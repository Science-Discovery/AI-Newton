import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t
# from aiphy.experiment.ground_laboratory.pendulum import create_eqs


pi = 3.1415926

exp_para = {
    'l1': default_parastructure(4, 5),
    'theta1': default_parastructure(-pi/8, pi/8),
    'omega1': default_parastructure(-pi/8, pi/8),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]

g = 9.801234567
# eqs = create_eqs(n=1)
# betas0 = [sp.symbols('beta'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, betas0)
# betas0 = [sp.simplify(beta.subs(sol)) for beta in betas0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('betas0 = [' + ', '.join(['sp.sympify("' + str(beta) + '")' for beta in betas0]) + ']')
# eqs = [sp.sympify("l1*m1*(beta1*l1 + g*sin(theta1))")]
betas0 = [sp.sympify("-g*sin(theta1)/l1")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    omega10 = exp_config.para('omega1')
    theta10 = exp_config.para('theta1')
    l1 = exp_config.para('l1')
    m1 = exp_config.get_obj_para('o1', 'm')
    numeric = {'m1': m1, 'l1': l1, 'g': g}
    betas = [beta.subs(numeric) for beta in betas0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    y0 = [theta10, omega10]

    def f(y, t):
        theta1, omega1 = y
        numeric = {'theta1': theta1, 'omega1': omega1}
        betas_n = [beta.subs(numeric) for beta in betas]
        return [omega1] + betas_n

    sol = odeint(f, y0, t)
    theta1 = sol[:, 0]
    posx1 = l1*np.sin(theta1)
    posz1 = -l1*np.cos(theta1)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), posx1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def pendulum_1_config() -> ExpStructure:
    expconfig = ExpConfig("pendulum_1", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1]**2 + posx[o1]**2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def pendulum_1_test():
    expconfig = ExpConfig("pendulum_1", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
