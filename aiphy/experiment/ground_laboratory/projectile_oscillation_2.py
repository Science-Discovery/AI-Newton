import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_dist, concept_length, concept_t


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-5.0, 5.0),
    "y0": default_parastructure(-5.0, 5.0),
    "z0": default_parastructure(-5.0, 5.0),
    "r1": default_parastructure(3.5, 5.5),
    "theta1": default_parastructure(0, pi),
    "phi1": default_parastructure(0, 2*pi),
    "r2": default_parastructure(3.5, 5.5),
    "theta2": default_parastructure(0, pi),
    "phi2": default_parastructure(0, 2*pi),
    "vx1": default_parastructure(-1.0, 1.0),
    "vy1": default_parastructure(-1.0, 1.0),
    "vz1": default_parastructure(-1.0, 1.0),
    "vx2": default_parastructure(-1.0, 1.0),
    "vy2": default_parastructure(-1.0, 1.0),
    "vz2": default_parastructure(-1.0, 1.0),
    "vx3": default_parastructure(-1.0, 1.0),
    "vy3": default_parastructure(-1.0, 1.0),
    "vz3": default_parastructure(-1.0, 1.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "o3": Objstructure.make_particle(3, 6),
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
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_dist, ["o1", "o3"]),
    (concept_dist, ["o3", "o1"]),
    (concept_dist, ["o3", "o2"]),
    (concept_dist, ["o2", "o3"]),
    (concept_length, ['s1']),
    (concept_length, ['s2']),
    (concept_t, ["clock"]),
]

# (ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3)
acs0 = [sp.sympify("k1*(1-l1/r12)*(x2-x1)/m1"), sp.sympify("k1*(1-l1/r12)*(y2-y1)/m1"), sp.sympify("k1*(1-l1/r12)*(z2-z1)/m1 - g"),
        sp.sympify("(k1*(1-l1/r12)*(x1-x2) + k2*(1-l2/r23)*(x3-x2))/m2"), sp.sympify("(k1*(1-l1/r12)*(y1-y2) + k2*(1-l2/r23)*(y3-y2))/m2"), sp.sympify("(k1*(1-l1/r12)*(z1-z2) + k2*(1-l2/r23)*(z3-z2))/m2 - g"),
        sp.sympify("k2*(1-l2/r23)*(x2-x3)/m3"), sp.sympify("k2*(1-l2/r23)*(y2-y3)/m3"), sp.sympify("k2*(1-l2/r23)*(z2-z3)/m3 - g")]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x10 = exp_config.para('x0')
    y10 = exp_config.para('y0')
    z10 = exp_config.para('z0')
    r1 = exp_config.para('r1')
    theta1 = exp_config.para('theta1')
    phi1 = exp_config.para('phi1')
    r2 = exp_config.para('r2')
    theta2 = exp_config.para('theta2')
    phi2 = exp_config.para('phi2')
    vx10 = exp_config.para('vx1')
    vy10 = exp_config.para('vy1')
    vz10 = exp_config.para('vz1')
    vx20 = exp_config.para('vx2')
    vy20 = exp_config.para('vy2')
    vz20 = exp_config.para('vz2')
    vx30 = exp_config.para('vx3')
    vy30 = exp_config.para('vy3')
    vz30 = exp_config.para('vz3')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    m3 = exp_config.get_obj_para('o3', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    k2 = exp_config.get_obj_para('s2', 'thickness')**3
    l2 = exp_config.get_obj_para('s2', 'freel')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    x20 = x10 + r1*np.sin(theta1)*np.cos(phi1)
    y20 = y10 + r1*np.sin(theta1)*np.sin(phi1)
    z20 = z10 + r1*np.cos(theta1)
    x30 = x20 + r2*np.sin(theta2)*np.cos(phi2)
    y30 = y20 + r2*np.sin(theta2)*np.sin(phi2)
    z30 = z20 + r2*np.cos(theta2)

    numeric = {'k1': k1, 'l1': l1, 'k2': k2, 'l2': l2,
               'm1': m1, 'm2': m2, 'm3': m3, 'g': g}
    acs = [ac.subs(numeric) for ac in acs0]
    input0 = [x10, y10, z10, x20, y20, z20, x30, y30, z30,
              vx10, vy10, vz10, vx20, vy20, vz20, vx30, vy30, vz30]

    def f(y, t):
        x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3 = y
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2)
        numeric = {'x1': x1, 'y1': y1, 'z1': z1, 'vx1': vx1, 'vy1': vy1, 'vz1': vz1,
                   'x2': x2, 'y2': y2, 'z2': z2, 'vx2': vx2, 'vy2': vy2, 'vz2': vz2,
                   'x3': x3, 'y3': y3, 'z3': z3, 'vx3': vx3, 'vy3': vy3, 'vz3': vz3,
                   'r12': r12, 'r23': r23}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3] + acs_n

    sol = odeint(f, input0, t)

    x1t = sol[:, 0]
    y1t = sol[:, 1]
    z1t = sol[:, 2]
    x2t = sol[:, 3]
    y2t = sol[:, 4]
    z2t = sol[:, 5]
    x3t = sol[:, 6]
    y3t = sol[:, 7]
    z3t = sol[:, 8]
    r12t = np.sqrt((x2t-x1t)**2 + (y2t-y1t)**2 + (z2t-z1t)**2)
    r23t = np.sqrt((x2t-x3t)**2 + (y2t-y3t)**2 + (z2t-z3t)**2)
    r13t = np.sqrt((x3t-x1t)**2 + (y3t-y1t)**2 + (z3t-z1t)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), y2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o3']), x3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o3']), y3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o3']), z3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o3']), r13t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o1']), r13t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o2']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o3']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s2']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct

def projectile_oscillation_2_config() -> ExpStructure:
    expconfig = ExpConfig("projectile_oscillation_2", 3, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2  - (posz[o1] - posz[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - dist[o3, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] ** 2 - (posx[o2] - posx[o3]) ** 2 - (posy[o2] - posy[o3]) ** 2  - (posz[o2] - posz[o3]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - length[s2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - dist[o3, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] ** 2 - (posx[o1] - posx[o3]) ** 2 - (posy[o1] - posy[o3]) ** 2  - (posz[o1] - posz[o3]) ** 2")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
