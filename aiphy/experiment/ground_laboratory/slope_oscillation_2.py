import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length, concept_dist, concept_cx, concept_cy, concept_cz


pi = 3.1415926

exp_para = {
    "theta": default_parastructure(pi/6, pi/3),
    "phi": default_parastructure(pi/6, pi/3),
    'r1': default_parastructure(4.0, 6.0),
    'v1': default_parastructure(-2.0, 2.0),
    'theta1': default_parastructure(-pi/3, pi/3),
    'omega1': default_parastructure(-pi/6, pi/6),
    'r2': default_parastructure(4.0, 6.0),
    'v2': default_parastructure(-2.0, 2.0),
    'theta2': default_parastructure(-pi/3, pi/3),
    'omega2': default_parastructure(-pi/6, pi/6),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "o2": Objstructure.make_particle(3, 6),
    "s2": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "slope": Objstructure.make_slope(),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ['s1']),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_length, ['s2']),
    (concept_cx, ["slope"]),
    (concept_cy, ["slope"]),
    (concept_cz, ["slope"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_t, ["clock"]),
]

g = 9.801234567
acs0 = [
    sp.sympify("a*cos(theta1) + omega1**2*r1 + (k1*(l1-r1)+k2*cos(theta1 - theta2)*(r2-l2))/m1"),
    sp.sympify("omega2**2*r2 + k1*cos(theta1 - theta2)*(r1-l1)/m1 + k2*(m1+m2)*(l2-r2)/(m1*m2)"),
    sp.sympify("-(a*sin(theta1) + 2*omega1*v1)/r1 + k2*(l2-r2)*sin(theta1 - theta2)/(r1*m1)"),
    sp.sympify("-2*omega2*v2/r2 + k1*(r1-l1)*sin(theta1 - theta2)/(m1*r2)")
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    theta = exp_config.para('theta')
    phi = exp_config.para('phi')
    # unit vector on slope
    n1 = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)]
    n2 = [-np.sin(phi), np.cos(phi), 0]

    r10 = exp_config.para('r1')
    v10 = exp_config.para('v1')
    omega10 = exp_config.para('omega1')
    theta10 = exp_config.para('theta1')
    l1 = exp_config.get_obj_para('s1', 'freel')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    m1 = exp_config.get_obj_para('o1', 'm')
    r20 = exp_config.para('r2')
    v20 = exp_config.para('v2')
    omega20 = exp_config.para('omega2')
    theta20 = exp_config.para('theta2')
    l2 = exp_config.get_obj_para('s2', 'freel')
    k2 = exp_config.get_obj_para('s2', 'thickness')**3
    m2 = exp_config.get_obj_para('o2', 'm')
    a = g*np.cos(theta)
    numeric = {'a': a, 'm1': m1, 'l1': l1, 'k1': k1, 'm2': m2, 'l2': l2, 'k2': k2}
    acs = [ac.subs(numeric) for ac in acs0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)

    y0 = [r10, r20, theta10, theta20, v10, v20, omega10, omega20]

    def f(y, t):
        r1, r2, theta1, theta2, v1, v2, omega1, omega2 = y
        numeric = {'r1': r1, 'theta1': theta1, 'v1': v1, 'omega1': omega1,
                   'r2': r2, 'theta2': theta2, 'v2': v2, 'omega2': omega2}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [v1, v2, omega1, omega2] + acs_n

    sol = odeint(f, y0, t)
    r1 = sol[:, 0]
    r2 = sol[:, 1]
    theta1 = sol[:, 2]
    theta2 = sol[:, 3]
    posx1 = r1*np.cos(theta1)*n1[0] + r1*np.sin(theta1)*n2[0]
    posy1 = r1*np.cos(theta1)*n1[1] + r1*np.sin(theta1)*n2[1]
    posz1 = r1*np.cos(theta1)*n1[2] + r1*np.sin(theta1)*n2[2]
    posx2 = (r1*np.cos(theta1) + r2*np.cos(theta2))*n1[0] + (r1*np.sin(theta1) + r2*np.sin(theta2))*n2[0]
    posy2 = (r1*np.cos(theta1) + r2*np.cos(theta2))*n1[1] + (r1*np.sin(theta1) + r2*np.sin(theta2))*n2[1]
    posz2 = (r1*np.cos(theta1) + r2*np.cos(theta2))*n1[2] + (r1*np.sin(theta1) + r2*np.sin(theta2))*n2[2]

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), posx1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), posy1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), posz1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), posx2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), posy2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), posz2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s2']), r2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), r2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), r2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cx, ['slope']), np.cos(theta)*np.cos(phi) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cy, ['slope']), np.cos(theta)*np.sin(phi) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cz, ['slope']), np.sin(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def slope_oscillation_2_config() -> ExpStructure:
    expconfig = ExpConfig("slope_oscillation_2", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(expconfig.gen_prop(
        "cx[slope] is conserved"
    ))
    expconfig.register_geometry_info(expconfig.gen_prop(
        "cy[slope] is conserved"
    ))
    expconfig.register_geometry_info(expconfig.gen_prop(
        "cz[slope] is conserved"
    ))
    expconfig.register_geometry_info(expconfig.gen_prop(
        "cx[slope] * posx[o1] + cy[slope] * posy[o1] + cz[slope] * posz[o1] is conserved"
    ))
    expconfig.register_geometry_info(expconfig.gen_prop(
        "cx[slope] * posx[o2] + cy[slope] * posy[o2] + cz[slope] * posz[o2] is conserved"
    ))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posx[o1]**2 + posy[o1]**2 + posz[o1]**2 - length[s1]**2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2 - (posz[o1] - posz[o2]) ** 2")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def slope_oscillation_2_test():
    expconfig = ExpConfig("slope_oscillation_2", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
