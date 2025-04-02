import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
# from aiphy.experiment.ground_laboratory.pendulum import create_eqs


pi = 3.1415926

exp_para = {
    'l1': default_parastructure(0.2, 0.8),
    'theta1': default_parastructure(-pi/3, pi/3),
    'omega1': default_parastructure(-5*pi/3, 5*pi/3),
    'l2': default_parastructure(0.2, 0.8),
    'theta2': default_parastructure(-pi/3, pi/3),
    'omega2': default_parastructure(-5*pi/3, 5*pi/3),
    'l3': default_parastructure(0.2, 0.8),
    'theta3': default_parastructure(-pi/3, pi/3),
    'omega3': default_parastructure(-5*pi/3, 5*pi/3)
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "o3": Objstructure.make_particle(3, 6),
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
    (concept_t, ["clock"]),
]

g = 9.801234567
# eqs = create_eqs(n=3)
# betas0 = [sp.symbols('beta'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, betas0)
# betas0 = [sp.simplify(beta.subs(sol)) for beta in betas0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('betas0 = [' + ', '.join(['sp.sympify("' + str(beta) + '")' for beta in betas0]) + ']')

betas0 = [
    sp.sympify("(g*m1*m2*sin(theta1) - g*m1*m3*sin(theta1)*cos(theta2 - theta3)**2 + g*m1*m3*sin(theta1) + g*m2**2*sin(theta1) - g*m2**2*sin(theta2)*cos(theta1 - theta2) - g*m2*m3*sin(theta1)*cos(theta2 - theta3)**2 + 2*g*m2*m3*sin(theta1) - 2*g*m2*m3*sin(theta2)*cos(theta1 - theta2) + g*m2*m3*sin(theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + g*m2*m3*sin(theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m2*m3*sin(theta3)*cos(theta1 - theta3) - g*m3**2*sin(theta1)*cos(theta2 - theta3)**2 + g*m3**2*sin(theta1) - g*m3**2*sin(theta2)*cos(theta1 - theta2) + g*m3**2*sin(theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + g*m3**2*sin(theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m3**2*sin(theta3)*cos(theta1 - theta3) + l1*m2**2*omega1**2*sin(2*theta1 - 2*theta2)/2 - l1*m2*m3*omega1**2*sin(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) - l1*m2*m3*omega1**2*sin(theta1 - theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) + l1*m2*m3*omega1**2*sin(2*theta1 - 2*theta2) + l1*m2*m3*omega1**2*sin(2*theta1 - 2*theta3)/2 - l1*m3**2*omega1**2*sin(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) - l1*m3**2*omega1**2*sin(theta1 - theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) + l1*m3**2*omega1**2*sin(2*theta1 - 2*theta2)/2 + l1*m3**2*omega1**2*sin(2*theta1 - 2*theta3)/2 + l2*m2**2*omega2**2*sin(theta1 - theta2) - l2*m2*m3*omega2**2*(-sin(theta1 - 3*theta2 + 2*theta3) + sin(theta1 + theta2 - 2*theta3))/4 - l2*m2*m3*omega2**2*sin(theta1 - theta2)*cos(theta2 - theta3)**2 + 2*l2*m2*m3*omega2**2*sin(theta1 - theta2) + l2*m2*m3*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta3) - l2*m3**2*omega2**2*(-sin(theta1 - 3*theta2 + 2*theta3) + sin(theta1 + theta2 - 2*theta3))/4 - l2*m3**2*omega2**2*sin(theta1 - theta2)*cos(theta2 - theta3)**2 + l2*m3**2*omega2**2*sin(theta1 - theta2) + l2*m3**2*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta3) + l3*m2*m3*omega3**2*sin(theta1 - theta3) - l3*m2*m3*omega3**2*sin(theta2 - theta3)*cos(theta1 - theta2) + l3*m3**2*omega3**2*(-sin(theta1 - 2*theta2 + theta3) + sin(theta1 + 2*theta2 - 3*theta3))/4 - l3*m3**2*omega3**2*sin(theta1 - theta3)*cos(theta2 - theta3)**2 + l3*m3**2*omega3**2*sin(theta1 - theta3) - l3*m3**2*omega3**2*sin(theta2 - theta3)*cos(theta1 - theta2))/(l1*(-m1*m2 + m1*m3*cos(theta2 - theta3)**2 - m1*m3 + m2**2*cos(theta1 - theta2)**2 - m2**2 + 2*m2*m3*cos(theta1 - theta2)**2 - 2*m2*m3*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m2*m3*cos(theta1 - theta3)**2 + m2*m3*cos(theta2 - theta3)**2 - 2*m2*m3 + m3**2*cos(theta1 - theta2)**2 - 2*m3**2*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m3**2*cos(theta1 - theta3)**2 + m3**2*cos(theta2 - theta3)**2 - m3**2))"),
    sp.sympify("(-g*m1*m2*sin(theta1)*cos(theta1 - theta2) + g*m1*m2*sin(theta2) - g*m1*m3*sin(theta1)*cos(theta1 - theta2) + g*m1*m3*sin(theta1)*cos(theta1 - theta3)*cos(theta2 - theta3) + g*m1*m3*sin(theta2) - g*m1*m3*sin(theta3)*cos(theta2 - theta3) - g*m2**2*sin(theta1)*cos(theta1 - theta2) + g*m2**2*sin(theta2) - 2*g*m2*m3*sin(theta1)*cos(theta1 - theta2) + g*m2*m3*sin(theta1)*cos(theta1 - theta3)*cos(theta2 - theta3) - g*m2*m3*sin(theta2)*cos(theta1 - theta3)**2 + 2*g*m2*m3*sin(theta2) + g*m2*m3*sin(theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) - g*m2*m3*sin(theta3)*cos(theta2 - theta3) - g*m3**2*sin(theta1)*cos(theta1 - theta2) + g*m3**2*sin(theta1)*cos(theta1 - theta3)*cos(theta2 - theta3) - g*m3**2*sin(theta2)*cos(theta1 - theta3)**2 + g*m3**2*sin(theta2) + g*m3**2*sin(theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) - g*m3**2*sin(theta3)*cos(theta2 - theta3) - l1*m1*m2*omega1**2*sin(theta1 - theta2) - l1*m1*m3*omega1**2*sin(theta1 - theta2) + l1*m1*m3*omega1**2*sin(theta1 - theta3)*cos(theta2 - theta3) - l1*m2**2*omega1**2*sin(theta1 - theta2) - l1*m2*m3*omega1**2*(-sin(-3*theta1 + theta2 + 2*theta3) + sin(theta1 + theta2 - 2*theta3))/4 + l1*m2*m3*omega1**2*sin(theta1 - theta2)*cos(theta1 - theta3)**2 - 2*l1*m2*m3*omega1**2*sin(theta1 - theta2) + l1*m2*m3*omega1**2*sin(theta1 - theta3)*cos(theta2 - theta3) - l1*m3**2*omega1**2*(-sin(-3*theta1 + theta2 + 2*theta3) + sin(theta1 + theta2 - 2*theta3))/4 + l1*m3**2*omega1**2*sin(theta1 - theta2)*cos(theta1 - theta3)**2 - l1*m3**2*omega1**2*sin(theta1 - theta2) + l1*m3**2*omega1**2*sin(theta1 - theta3)*cos(theta2 - theta3) + l2*m1*m3*omega2**2*sin(2*theta2 - 2*theta3)/2 - l2*m2**2*omega2**2*sin(2*theta1 - 2*theta2)/2 + l2*m2*m3*omega2**2*sin(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) - l2*m2*m3*omega2**2*sin(2*theta1 - 2*theta2) - l2*m2*m3*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) + l2*m2*m3*omega2**2*sin(2*theta2 - 2*theta3)/2 + l2*m3**2*omega2**2*sin(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) - l2*m3**2*omega2**2*sin(2*theta1 - 2*theta2)/2 - l2*m3**2*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) + l2*m3**2*omega2**2*sin(2*theta2 - 2*theta3)/2 + l3*m1*m3*omega3**2*sin(theta2 - theta3) - l3*m2*m3*omega3**2*sin(theta1 - theta3)*cos(theta1 - theta2) + l3*m2*m3*omega3**2*sin(theta2 - theta3) + l3*m3**2*omega3**2*(-sin(-2*theta1 + theta2 + theta3) + sin(2*theta1 + theta2 - 3*theta3))/4 - l3*m3**2*omega3**2*sin(theta1 - theta3)*cos(theta1 - theta2) - l3*m3**2*omega3**2*sin(theta2 - theta3)*cos(theta1 - theta3)**2 + l3*m3**2*omega3**2*sin(theta2 - theta3))/(l2*(-m1*m2 + m1*m3*cos(theta2 - theta3)**2 - m1*m3 + m2**2*cos(theta1 - theta2)**2 - m2**2 + 2*m2*m3*cos(theta1 - theta2)**2 - 2*m2*m3*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m2*m3*cos(theta1 - theta3)**2 + m2*m3*cos(theta2 - theta3)**2 - 2*m2*m3 + m3**2*cos(theta1 - theta2)**2 - 2*m3**2*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m3**2*cos(theta1 - theta3)**2 + m3**2*cos(theta2 - theta3)**2 - m3**2))"),
    sp.sympify("(g*m1*m2*sin(theta1)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m1*m2*sin(theta1)*cos(theta1 - theta3) - g*m1*m2*sin(theta2)*cos(theta2 - theta3) + g*m1*m2*sin(theta3) + g*m1*m3*sin(theta1)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m1*m3*sin(theta1)*cos(theta1 - theta3) - g*m1*m3*sin(theta2)*cos(theta2 - theta3) + g*m1*m3*sin(theta3) + g*m2**2*sin(theta1)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m2**2*sin(theta1)*cos(theta1 - theta3) + g*m2**2*sin(theta2)*cos(theta1 - theta2)*cos(theta1 - theta3) - g*m2**2*sin(theta2)*cos(theta2 - theta3) - g*m2**2*sin(theta3)*cos(theta1 - theta2)**2 + g*m2**2*sin(theta3) + 2*g*m2*m3*sin(theta1)*cos(theta1 - theta2)*cos(theta2 - theta3) - 2*g*m2*m3*sin(theta1)*cos(theta1 - theta3) + 2*g*m2*m3*sin(theta2)*cos(theta1 - theta2)*cos(theta1 - theta3) - 2*g*m2*m3*sin(theta2)*cos(theta2 - theta3) - 2*g*m2*m3*sin(theta3)*cos(theta1 - theta2)**2 + 2*g*m2*m3*sin(theta3) + g*m3**2*sin(theta1)*cos(theta1 - theta2)*cos(theta2 - theta3) - g*m3**2*sin(theta1)*cos(theta1 - theta3) + g*m3**2*sin(theta2)*cos(theta1 - theta2)*cos(theta1 - theta3) - g*m3**2*sin(theta2)*cos(theta2 - theta3) - g*m3**2*sin(theta3)*cos(theta1 - theta2)**2 + g*m3**2*sin(theta3) + l1*m1*m2*omega1**2*sin(theta1 - theta2)*cos(theta2 - theta3) - l1*m1*m2*omega1**2*sin(theta1 - theta3) + l1*m1*m3*omega1**2*sin(theta1 - theta2)*cos(theta2 - theta3) - l1*m1*m3*omega1**2*sin(theta1 - theta3) - l1*m2**2*omega1**2*(-sin(-3*theta1 + 2*theta2 + theta3) + sin(theta1 - 2*theta2 + theta3))/4 + l1*m2**2*omega1**2*sin(theta1 - theta2)*cos(theta2 - theta3) + l1*m2**2*omega1**2*sin(theta1 - theta3)*cos(theta1 - theta2)**2 - l1*m2**2*omega1**2*sin(theta1 - theta3) - l1*m2*m3*omega1**2*(-sin(-3*theta1 + 2*theta2 + theta3) + sin(theta1 - 2*theta2 + theta3))/2 + 2*l1*m2*m3*omega1**2*sin(theta1 - theta2)*cos(theta2 - theta3) + 2*l1*m2*m3*omega1**2*sin(theta1 - theta3)*cos(theta1 - theta2)**2 - 2*l1*m2*m3*omega1**2*sin(theta1 - theta3) - l1*m3**2*omega1**2*(-sin(-3*theta1 + 2*theta2 + theta3) + sin(theta1 - 2*theta2 + theta3))/4 + l1*m3**2*omega1**2*sin(theta1 - theta2)*cos(theta2 - theta3) + l1*m3**2*omega1**2*sin(theta1 - theta3)*cos(theta1 - theta2)**2 - l1*m3**2*omega1**2*sin(theta1 - theta3) - l2*m1*m2*omega2**2*sin(theta2 - theta3) - l2*m1*m3*omega2**2*sin(theta2 - theta3) + l2*m2**2*omega2**2*(-sin(-2*theta1 + theta2 + theta3) + sin(2*theta1 - 3*theta2 + theta3))/4 - l2*m2**2*omega2**2*sin(theta1 - theta2)*cos(theta1 - theta3) + l2*m2**2*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta2)**2 - l2*m2**2*omega2**2*sin(theta2 - theta3) + l2*m2*m3*omega2**2*(-sin(-2*theta1 + theta2 + theta3) + sin(2*theta1 - 3*theta2 + theta3))/2 - 2*l2*m2*m3*omega2**2*sin(theta1 - theta2)*cos(theta1 - theta3) + 2*l2*m2*m3*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta2)**2 - 2*l2*m2*m3*omega2**2*sin(theta2 - theta3) + l2*m3**2*omega2**2*(-sin(-2*theta1 + theta2 + theta3) + sin(2*theta1 - 3*theta2 + theta3))/4 - l2*m3**2*omega2**2*sin(theta1 - theta2)*cos(theta1 - theta3) + l2*m3**2*omega2**2*sin(theta2 - theta3)*cos(theta1 - theta2)**2 - l2*m3**2*omega2**2*sin(theta2 - theta3) - l3*m1*m3*omega3**2*sin(2*theta2 - 2*theta3)/2 + l3*m2*m3*omega3**2*sin(theta1 - theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) - l3*m2*m3*omega3**2*sin(2*theta1 - 2*theta3)/2 + l3*m2*m3*omega3**2*sin(theta2 - theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) - l3*m2*m3*omega3**2*sin(2*theta2 - 2*theta3)/2 + l3*m3**2*omega3**2*sin(theta1 - theta3)*cos(theta1 - theta2)*cos(theta2 - theta3) - l3*m3**2*omega3**2*sin(2*theta1 - 2*theta3)/2 + l3*m3**2*omega3**2*sin(theta2 - theta3)*cos(theta1 - theta2)*cos(theta1 - theta3) - l3*m3**2*omega3**2*sin(2*theta2 - 2*theta3)/2)/(l3*(-m1*m2 + m1*m3*cos(theta2 - theta3)**2 - m1*m3 + m2**2*cos(theta1 - theta2)**2 - m2**2 + 2*m2*m3*cos(theta1 - theta2)**2 - 2*m2*m3*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m2*m3*cos(theta1 - theta3)**2 + m2*m3*cos(theta2 - theta3)**2 - 2*m2*m3 + m3**2*cos(theta1 - theta2)**2 - 2*m3**2*cos(theta1 - theta2)*cos(theta1 - theta3)*cos(theta2 - theta3) + m3**2*cos(theta1 - theta3)**2 + m3**2*cos(theta2 - theta3)**2 - m3**2))")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    omega10 = exp_config.para('omega1')
    theta10 = exp_config.para('theta1')
    l1 = exp_config.para('l1')
    m1 = exp_config.get_obj_para('o1', 'm')
    omega20 = exp_config.para('omega2')
    theta20 = exp_config.para('theta2')
    l2 = exp_config.para('l2')
    m2 = exp_config.get_obj_para('o2', 'm')
    omega30 = exp_config.para('omega3')
    theta30 = exp_config.para('theta3')
    l3 = exp_config.para('l3')
    m3 = exp_config.get_obj_para('o3', 'm')
    numeric = {'g': g, 'm1': m1, 'l1': l1, 'm2': m2, 'l2': l2, 'm3': m3, 'l3': l3}
    betas = [beta.subs(numeric) for beta in betas0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    y0 = [theta10, theta20, theta30, omega10, omega20, omega30]

    def f(y, t):
        theta1, theta2, theta3, omega1, omega2, omega3 = y
        numeric = {'theta1': theta1, 'omega1': omega1,
                   'theta2': theta2, 'omega2': omega2,
                   'theta3': theta3, 'omega3': omega3
                   }
        betas_n = [beta.subs(numeric) for beta in betas]
        return [omega1, omega2, omega3] + betas_n

    sol = odeint(f, y0, t)
    theta1 = sol[:, 0]
    theta2 = sol[:, 1]
    theta3 = sol[:, 2]
    posx1 = l1*np.sin(theta1)
    posz1 = -l1*np.cos(theta1)
    posx2 = posx1 + l2*np.sin(theta2)
    posz2 = posz1 - l2*np.cos(theta2)
    posx3 = posx2 + l3*np.sin(theta3)
    posz3 = posz2 - l3*np.cos(theta3)

    dist13 = np.sqrt((posx3-posx1)**2 + (posz3-posz1)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), posx1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), posx2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o3']), posx3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o3']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o3']), posz3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), l2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), l2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o3']), l3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o2']), l3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o3']), dist13 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o1']), dist13 + np.random.normal(0, error, t_num))

    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def pendulum_3_config() -> ExpStructure:
    expconfig = ExpConfig("pendulum_3", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posz[o1]**2 + posx[o1]**2")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "(posz[o2]-posz[o1])**2 + (posx[o2]-posx[o1])**2 - dist[o1, o2]**2")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "(posz[o3]-posz[o2])**2 + (posx[o3]-posx[o2])**2 - dist[o2, o3]**2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - dist[o3, o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def pendulum_3_test():
    expconfig = ExpConfig("pendulum_3", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
