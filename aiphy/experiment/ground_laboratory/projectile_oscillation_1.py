import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_dist, concept_t, concept_length


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-5.0, 5.0),
    "y0": default_parastructure(-5.0, 5.0),
    "z0": default_parastructure(-5.0, 5.0),
    "r1": default_parastructure(3.5, 5.5),
    "theta1": default_parastructure(0, pi),
    "phi1": default_parastructure(0, 2*pi),
    "vx1": default_parastructure(-1.0, 1.0),
    "vy1": default_parastructure(-1.0, 1.0),
    "vz1": default_parastructure(-1.0, 1.0),
    "vx2": default_parastructure(-1.0, 1.0),
    "vy2": default_parastructure(-1.0, 1.0),
    "vz2": default_parastructure(-1.0, 1.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3, 6),
    "o2": Objstructure.make_particle(3, 6),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
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
    (concept_length, ['s1']),
    (concept_t, ["clock"]),
]

# (ax1, ay1, az1, ax2, ay2, az2)
acs0 = [sp.sympify("k1*(1-l1/r12)*(x2-x1)/m1"), sp.sympify("k1*(1-l1/r12)*(y2-y1)/m1"), sp.sympify("k1*(1-l1/r12)*(z2-z1)/m1 - g"),
        sp.sympify("k1*(1-l1/r12)*(x1-x2)/m2"), sp.sympify("k1*(1-l1/r12)*(y1-y2)/m2"), sp.sympify("k1*(1-l1/r12)*(z1-z2)/m2 - g")]

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x10 = exp_config.para('x0')
    y10 = exp_config.para('y0')
    z10 = exp_config.para('z0')
    r1 = exp_config.para('r1')
    theta1 = exp_config.para('theta1')
    phi1 = exp_config.para('phi1')
    vx10 = exp_config.para('vx1')
    vy10 = exp_config.para('vy1')
    vz10 = exp_config.para('vz1')
    vx20 = exp_config.para('vx2')
    vy20 = exp_config.para('vy2')
    vz20 = exp_config.para('vz2')
    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    g = 9.801234567

    x20 = x10 + r1*np.sin(theta1)*np.cos(phi1)
    y20 = y10 + r1*np.sin(theta1)*np.sin(phi1)
    z20 = z10 + r1*np.cos(theta1)

    numeric = {'k1': k1, 'l1': l1, 'm1': m1, 'm2': m2, 'g': g}
    acs = [ac.subs(numeric) for ac in acs0]
    input0 = [x10, y10, z10, x20, y20, z20, vx10, vy10, vz10, vx20, vy20, vz20]

    def f(y, t):
        x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = y
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        numeric = {'x1': x1, 'y1': y1, 'z1': z1, 'vx1': vx1, 'vy1': vy1, 'vz1': vz1,
                   'x2': x2, 'y2': y2, 'z2': z2, 'vx2': vx2, 'vy2': vy2, 'vz2': vz2,
                   'r12': r12}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vy1, vz1, vx2, vy2, vz2] + acs_n

    sol = odeint(f, input0, t)

    x1t = sol[:, 0]
    y1t = sol[:, 1]
    z1t = sol[:, 2]
    x2t = sol[:, 3]
    y2t = sol[:, 4]
    z2t = sol[:, 5]
    r12t = np.sqrt((x2t-x1t)**2 + (y2t-y1t)**2 + (z2t-z1t)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), y2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct

def projectile_oscillation_1_config() -> ExpStructure:
    expconfig = ExpConfig("projectile_oscillation_1", 3, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2  - (posz[o1] - posz[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s1]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
