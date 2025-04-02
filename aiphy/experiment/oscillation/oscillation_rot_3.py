import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length, concept_dist
# from aiphy.experiment.oscillation.free_oscillation import create_eqs


pi = 3.1415926

exp_para = {
    "r10": default_parastructure(4.0, 7.0),
    "v10": default_parastructure(-2.0, 2.0),
    "theta10": default_parastructure(0, pi/2),
    "omega10": default_parastructure(pi/4.0, pi/2.0),
    "r20": default_parastructure(4.0, 7.0),
    "v20": default_parastructure(-2.0, 2.0),
    "theta20": default_parastructure(0, pi/2),
    "omega20": default_parastructure(-pi/4.0, pi/4.0),
    "r30": default_parastructure(4.0, 7.0),
    "v30": default_parastructure(-2.0, 2.0),
    "theta30": default_parastructure(0, pi/2),
    "omega30": default_parastructure(-pi/4.0, pi/4.0),
}
obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "o2": Objstructure.make_particle(3.0, 6.0),
    "s2": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "o3": Objstructure.make_particle(3.0, 6.0),
    "s3": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_length, ["s1"]),
    (concept_posx, ["o2"]),
    (concept_posy, ["o2"]),
    (concept_posz, ["o2"]),
    (concept_length, ["s2"]),
    (concept_posx, ["o3"]),
    (concept_posy, ["o3"]),
    (concept_posz, ["o3"]),
    (concept_length, ["s3"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_dist, ["o1", "o3"]),
    (concept_dist, ["o3", "o1"]),
    (concept_dist, ["o3", "o2"]),
    (concept_dist, ["o2", "o3"]),
    (concept_t, ["clock"]),
]


# eqs = create_eqs(n=1+1)
# acs0 = [sp.symbols('a'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, acs0)
# acs0 = [sp.simplify(ac.subs(sol)) for ac in acs0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('acs0 = [' + ', '.join(['sp.sympify("' + str(ac) + '")' for ac in acs0]) + ']')

acs0 = [sp.sympify("(-k1*(1-l1/r1)*x1+k2*(1-l2/r12)*(x2-x1))/m1"),
        sp.sympify("(k2*(1-l2/r12)*(x1-x2)+k3*(1-l3/r23)*(x3-x2))/m2"),
        sp.sympify("k3*(1-l3/r23)*(x2-x3)/m3"),
        sp.sympify("(-k1*(1-l1/r1)*y1+k2*(1-l2/r12)*(y2-y1))/m1"),
        sp.sympify("(k2*(1-l2/r12)*(y1-y2)+k3*(1-l3/r23)*(y3-y2))/m2"),
        sp.sympify("k3*(1-l3/r23)*(y2-y3)/m3"),]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    r10 = exp_config.para('r10')
    v10 = exp_config.para('v10')
    theta10 = exp_config.para('theta10')
    omega10 = exp_config.para('omega10')
    m1 = exp_config.get_obj_para('o1', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    r20 = exp_config.para('r20')
    v20 = exp_config.para('v20')
    theta20 = exp_config.para('theta20')
    omega20 = exp_config.para('omega20')
    m2 = exp_config.get_obj_para('o2', 'm')
    k2 = exp_config.get_obj_para('s2', 'thickness')**3
    l2 = exp_config.get_obj_para('s2', 'freel')
    r30 = exp_config.para('r30')
    v30 = exp_config.para('v30')
    theta30 = exp_config.para('theta30')
    omega30 = exp_config.para('omega30')
    m3 = exp_config.get_obj_para('o3', 'm')
    k3 = exp_config.get_obj_para('s3', 'thickness')**3
    l3 = exp_config.get_obj_para('s3', 'freel')
    numeric = {'k1': k1, 'l1': l1, 'm1': m1,
               'k2': k2, 'l2': l2, 'm2': m2,
               'k3': k3, 'l3': l3, 'm3': m3}
    acs = [ac.subs(numeric) for ac in acs0]

    x10 = r10*np.cos(theta10)
    y10 = r10*np.sin(theta10)
    vx10 = v10*np.cos(theta10)-omega10*r10*np.sin(theta10)
    vy10 = v10*np.sin(theta10)+omega10*r10*np.cos(theta10)
    x20 = x10 + r20*np.cos(theta20)
    y20 = y10 + r20*np.sin(theta20)
    vx20 = vx10 + v20*np.cos(theta20)-omega20*r20*np.sin(theta20)
    vy20 = vy10 + v20*np.sin(theta20)+omega20*r20*np.cos(theta20)
    x30 = x20 + r30*np.cos(theta30)
    y30 = y20 + r30*np.sin(theta30)
    vx30 = vx20 + v30*np.cos(theta30)-omega30*r30*np.sin(theta30)
    vy30 = vy20 + v30*np.sin(theta30)+omega30*r30*np.cos(theta30)
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, x20, x30, y10, y20, y30, vx10, vx20, vx30, vy10, vy20, vy30]

    def f(y, t):
        x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3 = y
        r1 = np.sqrt(x1**2+y1**2)
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        numeric = {'x1': x1, 'y1': y1, 'vx1': vx1, 'vy1': vy1, 'r1': r1,
                   'x2': x2, 'y2': y2, 'vx2': vx2, 'vy2': vy2, 'r12': r12,
                   'x3': x3, 'y3': y3, 'vx3': vx3, 'vy3': vy3, 'r23': r23}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vx2, vx3, vy1, vy2, vy3] + acs_n

    sol = odeint(f, y0, t)

    x1t = sol[:, 0]
    x2t = sol[:, 1]
    x3t = sol[:, 2]
    y1t = sol[:, 3]
    y2t = sol[:, 4]
    y3t = sol[:, 5]
    r1t = np.sqrt(x1t**2 + y1t**2)
    r12t = np.sqrt((x2t - x1t)**2 + (y2t - y1t)**2)
    r23t = np.sqrt((x3t - x2t)**2 + (y3t - y2t)**2)
    r13t = np.sqrt((x3t - x1t)**2 + (y3t - y1t)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), y2t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o3']), x3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o3']), y3t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o3']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o2']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o3']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o3']), r13t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o1']), r13t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), r1t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s2']), r12t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s3']), r23t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def oscillation_rot_3_config() -> ExpStructure:
    expconfig = ExpConfig("oscillation_rot_3", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "length[s1] ** 2 - posx[o1] ** 2 - posy[o1] ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - dist[o3, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] ** 2 - (posx[o3] - posx[o2]) ** 2 - (posy[o3] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - length[s3]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] ** 2 - (posx[o1] - posx[o3]) ** 2 - (posy[o1] - posy[o3]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - dist[o3, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o3]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def oscillation_rot_3_test():
    expconfig = ExpConfig("oscillation_rot_3", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
