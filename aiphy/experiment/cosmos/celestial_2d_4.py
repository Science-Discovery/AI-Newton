import numpy as np
import sympy as sp
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
from scipy.integrate import odeint
# from aiphy.experiment.cosmos.celestial_2d import create_eqs


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-1e-1, 1e-1),
    "y0": default_parastructure(-1e-1, 1e-1),
    "r0": default_parastructure(2.2e-2, 2.4e-2),
    "theta0": default_parastructure(0.0, pi/2),
    "omega10": default_parastructure(3*pi/5, 4*pi/5),
    "v10": default_parastructure(-5e-5, 5e-5),
    "v20": default_parastructure(-5e-5, 5e-5),
    "v30": default_parastructure(-5e-5, 5e-5),
    "v40": default_parastructure(-5e-5, 5e-5),
}

obj_info = {
    "o1": Objstructure.make_particle(5e5, 6e5),
    "o2": Objstructure.make_particle(5e5, 6e5),
    "o3": Objstructure.make_particle(5e5, 6e5),
    "o4": Objstructure.make_particle(5e5, 6e5),
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
    (concept_posx, ["o4"]),
    (concept_posy, ["o4"]),
    (concept_posz, ["o4"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_dist, ["o1", "o3"]),
    (concept_dist, ["o3", "o1"]),
    (concept_dist, ["o3", "o2"]),
    (concept_dist, ["o2", "o3"]),
    (concept_dist, ["o1", "o4"]),
    (concept_dist, ["o4", "o1"]),
    (concept_dist, ["o2", "o4"]),
    (concept_dist, ["o4", "o2"]),
    (concept_dist, ["o3", "o4"]),
    (concept_dist, ["o4", "o3"]),
    (concept_t, ["clock"]),
]

G = 6.6e-11
acs0 = [sp.sympify("G*(m2*(x2-x1)/r12**3 + m3*(x3-x1)/r13**3 + m4*(x4-x1)/r14**3)"),
        sp.sympify("G*(m1*(x1-x2)/r12**3 + m3*(x3-x2)/r23**3 + m4*(x4-x2)/r24**3)"),
        sp.sympify("G*(m1*(x1-x3)/r13**3 + m2*(x2-x3)/r23**3 + m4*(x4-x3)/r34**3)"),
        sp.sympify("G*(m1*(x1-x4)/r14**3 + m2*(x2-x4)/r24**3 + m3*(x3-x4)/r34**3)"),
        sp.sympify("G*(m2*(y2-y1)/r12**3 + m3*(y3-y1)/r13**3 + m4*(y4-y1)/r14**3)"),
        sp.sympify("G*(m1*(y1-y2)/r12**3 + m3*(y3-y2)/r23**3 + m4*(y4-y2)/r24**3)"),
        sp.sympify("G*(m1*(y1-y3)/r13**3 + m2*(y2-y3)/r23**3 + m4*(y4-y3)/r34**3)"),
        sp.sympify("G*(m1*(y1-y4)/r14**3 + m2*(y2-y4)/r24**3 + m3*(y3-y4)/r34**3)"),]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para('x0')
    y0 = exp_config.para('y0')
    r0 = exp_config.para('r0')
    theta0 = exp_config.para('theta0')
    omega10 = exp_config.para('omega10')
    v10 = exp_config.para('v10')
    v20 = exp_config.para('v20')
    v30 = exp_config.para('v30')
    v40 = exp_config.para('v40')

    x10 = x0 + r0*np.cos(theta0)
    x20 = x0 + r0*np.cos(theta0 + pi/2)
    x30 = x0 + r0*np.cos(theta0 + pi)
    x40 = x0 + r0*np.cos(theta0 + 3*pi/2)
    y10 = y0 + r0*np.sin(theta0)
    y20 = y0 + r0*np.sin(theta0 + pi/2)
    y30 = y0 + r0*np.sin(theta0 + pi)
    y40 = y0 + r0*np.sin(theta0 + 3*pi/2)
    vx10 = v10*np.cos(theta0) - omega10*r0*np.sin(theta0)
    vy10 = v10*np.sin(theta0) + omega10*r0*np.cos(theta0)
    vx20 = v20*np.cos(theta0 + pi/2) - omega10*r0*np.sin(theta0 + pi/2)
    vy20 = v20*np.sin(theta0 + pi/2) + omega10*r0*np.cos(theta0 + pi/2)
    vx30 = v30*np.cos(theta0 + pi) - omega10*r0*np.sin(theta0 + pi)
    vy30 = v30*np.sin(theta0 + pi) + omega10*r0*np.cos(theta0 + pi)
    vx40 = v40*np.cos(theta0 + 3*pi/2) - omega10*r0*np.sin(theta0 + 3*pi/2)
    vy40 = v40*np.sin(theta0 + 3*pi/2) + omega10*r0*np.cos(theta0 + 3*pi/2)

    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    m3 = exp_config.get_obj_para('o3', 'm')
    m4 = exp_config.get_obj_para('o4', 'm')
    numeric = {'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'G': G}
    acs = [ac.subs(numeric) for ac in acs0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, x20, x30, x40, y10, y20, y30, y40, vx10, vx20, vx30, vx40, vy10, vy20, vy30, vy40]

    def f(y, t):
        x1, x2, x3, x4, y1, y2, y3, y4, vx1, vx2, vx3, vx4, vy1, vy2, vy3, vy4 = y
        r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        r13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
        r14 = np.sqrt((x1-x4)**2 + (y1-y4)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        r24 = np.sqrt((x4-x2)**2 + (y4-y2)**2)
        r34 = np.sqrt((x3-x4)**2 + (y3-y4)**2)
        numeric = {'r12': r12, 'r23': r23, 'r13': r13, 'r14': r14, 'r24': r24, 'r34': r34,
                   'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4,
                   'vx1': vx1, 'vx2': vx2, 'vx3': vx3, 'vx4': vx4, 'vy1': vy1, 'vy2': vy2, 'vy3': vy3, 'vy4': vy4}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vx2, vx3, vx4, vy1, vy2, vy3, vy4] + acs_n
    sol = odeint(f, y0, t)
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    x3 = sol[:, 2]
    x4 = sol[:, 3]
    y1 = sol[:, 4]
    y2 = sol[:, 5]
    y3 = sol[:, 6]
    y4 = sol[:, 7]
    dist12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    dist23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    dist34 = np.sqrt((x3-x4)**2 + (y3-y4)**2)
    dist13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
    dist24 = np.sqrt((x4-x2)**2 + (y4-y2)**2)
    dist41 = np.sqrt((x1-x4)**2 + (y1-y4)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), y2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o3']), x3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o3']), y3 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o3']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o4']), x4 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o4']), y4 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o4']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), dist12 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), dist12 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o3']), dist13 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o1']), dist13 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o4']), dist41 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o4', 'o1']), dist41 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o2']), dist23 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o3']), dist23 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o4', 'o2']), dist24 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o4']), dist24 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o4']), dist34 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o4', 'o3']), dist34 + np.random.normal(0, error, t_num))
    return data_struct


def celestial_4_config() -> ExpStructure:
    expconfig = ExpConfig("celestial_4", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o3, o2] - dist[o2, o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o3, o2] ** 2 - (posx[o3] - posx[o2]) ** 2 - (posy[o3] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - dist[o3, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] ** 2 - (posx[o1] - posx[o3]) ** 2 - (posy[o1] - posy[o3]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o4] - dist[o4, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o4] ** 2 - (posx[o1] - posx[o4]) ** 2 - (posy[o1] - posy[o4]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o4] - dist[o4, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o4] ** 2 - (posx[o2] - posx[o4]) ** 2 - (posy[o2] - posy[o4]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o3, o4] - dist[o4, o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o3, o4] ** 2 - (posx[o3] - posx[o4]) ** 2 - (posy[o3] - posy[o4]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o4]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
