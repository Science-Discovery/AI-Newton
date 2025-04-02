import numpy as np
import sympy as sp
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_length, concept_dist
# from aiphy.experiment.oscillation.free_oscillation import create_eqs


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-1.0, 1.0),
    "y0": default_parastructure(-1.0, 1.0),
    "r0": default_parastructure(5.0, 7.0),
    "theta0": default_parastructure(0.0, 2*pi/3),
    "omega10": default_parastructure(1.0, 2.0),
    "omega20": default_parastructure(1.0, 2.0),
    "omega30": default_parastructure(1.0, 2.0),
    "v10": default_parastructure(-0.05, 0.05),
    "v20": default_parastructure(-0.05, 0.05),
    "v30": default_parastructure(-0.05, 0.05),
}
obj_info = {
    "o1": Objstructure.make_particle(3.0, 6.0),
    "o2": Objstructure.make_particle(3.0, 6.0),
    "o3": Objstructure.make_particle(3.0, 6.0),
    "s1": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "s2": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "s3": Objstructure.make_spring(1.5, 2.5, 4.0, 6.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posx, ["o2"]),
    (concept_posx, ["o3"]),
    (concept_posy, ["o1"]),
    (concept_posy, ["o2"]),
    (concept_posy, ["o3"]),
    (concept_posz, ["o1"]),
    (concept_posz, ["o2"]),
    (concept_posz, ["o3"]),
    (concept_dist, ["o1", "o2"]),
    (concept_dist, ["o2", "o1"]),
    (concept_dist, ["o1", "o3"]),
    (concept_dist, ["o3", "o1"]),
    (concept_dist, ["o3", "o2"]),
    (concept_dist, ["o2", "o3"]),
    (concept_length, ["s1"]),
    (concept_length, ["s2"]),
    (concept_length, ["s3"]),
    (concept_t, ["clock"]),
]


# eqs = create_eqs(n=1+2)
# acs0 = [sp.symbols('a'+str(i+1)) for i in range(len(eqs))]
# sol = sp.solve(eqs, acs0)
# acs0 = [sp.simplify(ac.subs(sol)) for ac in acs0]
# print('eqs = [' + ', '.join(['sp.sympify("' + str(eq) + '")' for eq in eqs]) + ']')
# print('acs0 = [' + ', '.join(['sp.sympify("' + str(ac) + '")' for ac in acs0]) + ']')

# eqs = [sp.sympify("a1*m1 + k1*(2*l1 + 2*x1 - 2*x2)/2"), sp.sympify("a2*m2 + k1*(-2*l1 - 2*x1 + 2*x2)/2 + k2*(2*l2 + 2*x2 - 2*x3)/2"), sp.sympify("a3*m3 + k2*(-2*l2 - 2*x2 + 2*x3)/2")]
acs0 = [sp.sympify("(k1*(1-l1/r12)*(x2-x1)+k3*(1-l3/r13)*(x3-x1))/m1"),
        sp.sympify("(k2*(1-l2/r23)*(x3-x2)+k1*(1-l1/r12)*(x1-x2))/m2"),
        sp.sympify("(k3*(1-l3/r13)*(x1-x3)+k2*(1-l2/r23)*(x2-x3))/m3"),
        sp.sympify("(k1*(1-l1/r12)*(y2-y1)+k3*(1-l3/r13)*(y3-y1))/m1"),
        sp.sympify("(k2*(1-l2/r23)*(y3-y2)+k1*(1-l1/r12)*(y1-y2))/m2"),
        sp.sympify("(k3*(1-l3/r13)*(y1-y3)+k2*(1-l2/r23)*(y2-y3))/m3")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para('x0')
    y0 = exp_config.para('y0')
    r0 = exp_config.para('r0')
    theta0 = exp_config.para('theta0')
    omega10 = exp_config.para('omega10')
    omega20 = exp_config.para('omega20')
    omega30 = exp_config.para('omega30')
    v10 = exp_config.para('v10')
    v20 = exp_config.para('v20')
    v30 = exp_config.para('v30')

    x10 = x0 + r0*np.cos(theta0)
    x20 = x0 + r0*np.cos(theta0 + 2*pi/3)
    x30 = x0 + r0*np.cos(theta0 + 4*pi/3)
    y10 = y0 + r0*np.sin(theta0)
    y20 = y0 + r0*np.sin(theta0 + 2*pi/3)
    y30 = y0 + r0*np.sin(theta0 + 4*pi/3)
    vx10 = v10*np.cos(theta0) - omega10*r0*np.sin(theta0)
    vy10 = v10*np.sin(theta0) + omega10*r0*np.cos(theta0)
    vx20 = v20*np.cos(theta0 + 2*pi/3) - omega20*r0*np.sin(theta0 + 2*pi/3)
    vy20 = v20*np.sin(theta0 + 2*pi/3) + omega20*r0*np.cos(theta0 + 2*pi/3)
    vx30 = v30*np.cos(theta0 + 4*pi/3) - omega30*r0*np.sin(theta0 + 4*pi/3)
    vy30 = v30*np.sin(theta0 + 4*pi/3) + omega30*r0*np.cos(theta0 + 4*pi/3)

    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    m3 = exp_config.get_obj_para('o3', 'm')
    k1 = exp_config.get_obj_para('s1', 'thickness')**3
    l1 = exp_config.get_obj_para('s1', 'freel')
    k2 = exp_config.get_obj_para('s2', 'thickness')**3
    l2 = exp_config.get_obj_para('s2', 'freel')
    k3 = exp_config.get_obj_para('s3', 'thickness')**3
    l3 = exp_config.get_obj_para('s3', 'freel')
    numeric = {'k1': k1, 'l1': l1, 'k2': k2, 'l2': l2, 'k3': k3, 'l3': l3,
               'm1': m1, 'm2': m2, 'm3': m3}
    acs = [ac.subs(numeric) for ac in acs0]

    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, x20, x30, y10, y20, y30, vx10, vx20, vx30, vy10, vy20, vy30]

    def f(y, t):
        x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3 = y
        r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        r13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
        numeric = {'r12': r12, 'r23': r23, 'r13': r13,
                   'x1': x1, 'x2': x2, 'x3': x3, 'y1': y1, 'y2': y2, 'y3': y3,
                   'vx1': vx1, 'vx2': vx2, 'vx3': vx3, 'vy1': vy1, 'vy2': vy2, 'vy3': vy3}
        acs_n = [ac.subs(numeric) for ac in acs]
        return [vx1, vx2, vx3, vy1, vy2, vy3] + acs_n

    sol = odeint(f, y0, t)

    x1 = sol[:, 0]
    x2 = sol[:, 1]
    x3 = sol[:, 2]
    y1 = sol[:, 3]
    y2 = sol[:, 4]
    y3 = sol[:, 5]
    dist12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    dist23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    dist31 = np.sqrt((x1-x3)**2 + (y1-y3)**2)

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
    data_struct.insert_data((concept_dist, ['o1', 'o2']), dist12 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), dist12 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o3']), dist31 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o1']), dist31 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o3', 'o2']), dist23 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o3']), dist23 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s1']), dist12 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s2']), dist23 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_length, ['s3']), dist31 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def oscillation_r_3_config() -> ExpStructure:
    expconfig = ExpConfig("oscillation_r_3", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - length[s1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - length[s2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] ** 2 - (posx[o2] - posx[o3]) ** 2 - (posy[o2] - posy[o3]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o2, o3] - dist[o3, o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o3] - length[s3]")))
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


def oscillation_r_3_test():
    expconfig = ExpConfig("oscillation_r_3", 1, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
