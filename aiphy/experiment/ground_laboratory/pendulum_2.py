import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
# from aiphy.experiment.ground_laboratory.pendulum import create_eqs


pi = 3.1415926

exp_para = {
    'l1': default_parastructure(5.2, 5.8),
    'theta1': default_parastructure(-pi/6, pi/6),
    'omega1': default_parastructure(-5*pi/20, 5*pi/20),
    'l2': default_parastructure(5.2, 5.8),
    'theta2': default_parastructure(-pi/6, pi/6),
    'omega2': default_parastructure(-5*pi/20, 5*pi/20)
}
obj_info = {
    "o1": Objstructure.make_particle(4, 5),
    "o2": Objstructure.make_particle(4, 5),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_t, ["clock"]),
]

g = 9.801234567
# eqs = create_eqs(n=2)
# betas0 = [sp.symbols('beta'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, betas0)
# betas0 = [sp.simplify(beta.subs(sol)) for beta in betas0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('betas0 = [' + ', '.join(['sp.sympify("' + str(beta) + '")' for beta in betas0]) + ']')

# eqs = [sp.sympify("l1*(beta1*l1*m1 + beta1*l1*m2 + beta2*l2*m2*cos(theta1 - theta2) + g*m1*sin(theta1) + g*m2*sin(theta1) + l2*m2*omega2**2*sin(theta1 - theta2))"),
#        sp.sympify("l2*m2*(beta1*l1*cos(theta1 - theta2) + beta2*l2 + g*sin(theta2) - l1*omega1**2*sin(theta1 - theta2))")]
betas0 = [sp.sympify("(-g*m1*sin(theta1) - g*m2*sin(theta1)/2 - g*m2*sin(theta1 - 2*theta2)/2 - l1*m2*omega1**2*sin(2*theta1 - 2*theta2)/2 - l2*m2*omega2**2*sin(theta1 - theta2))/(l1*(m1 - m2*cos(theta1 - theta2)**2 + m2))"),
          sp.sympify("(-g*m1*sin(theta2)/2 + g*m1*sin(2*theta1 - theta2)/2 - g*m2*sin(theta2)/2 + g*m2*sin(2*theta1 - theta2)/2 + l1*m1*omega1**2*sin(theta1 - theta2) + l1*m2*omega1**2*sin(theta1 - theta2) + l2*m2*omega2**2*sin(2*theta1 - 2*theta2)/2)/(l2*(m1 - m2*cos(theta1 - theta2)**2 + m2))")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    omega10 = exp_config.para('omega1')
    theta10 = exp_config.para('theta1')
    omega20 = exp_config.para('omega2')
    theta20 = exp_config.para('theta2')
    l1 = exp_config.para('l1')
    l2 = exp_config.para('l2')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    numeric = {'m1': m1, 'm2': m2, 'l1': l1, 'l2': l2, 'g': g}
    betas = [beta.subs(numeric) for beta in betas0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    y0 = [theta10, theta20, omega10, omega20]

    def f(y, t):
        theta1, theta2, omega1, omega2 = y
        numeric = {'theta1': theta1, 'theta2': theta2, 'omega1': omega1, 'omega2': omega2}
        betas_n = [beta.subs(numeric) for beta in betas]
        return [omega1, omega2] + betas_n

    sol = odeint(f, y0, t)
    theta1 = sol[:, 0]
    theta2 = sol[:, 1]
    posx1 = l1*np.sin(theta1)
    posz1 = -l1*np.cos(theta1)
    posx2 = posx1 + l2*np.sin(theta2)
    posz2 = posz1 - l2*np.cos(theta2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), posx1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), posx2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), l2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), l2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def pendulum_2_config() -> ExpStructure:
    expconfig = ExpConfig("pendulum_2", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1]**2 + posx[o1]**2")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "(posz[o2]-posz[o1])**2 + (posx[o2]-posx[o1])**2 - dist[o1, o2]**2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "dist[o1, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def pendulum_2_test():
    expconfig = ExpConfig("pendulum_2", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
