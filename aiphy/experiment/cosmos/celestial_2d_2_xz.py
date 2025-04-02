import numpy as np
import sympy as sp
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist
from scipy.integrate import odeint
# from aiphy.experiment.cosmos.celestial_2d import create_eqs


pi = 3.1415926

exp_para = {
    "x0": default_parastructure(-1e-1, 1e-1),
    "z0": default_parastructure(-1e-1, 1e-1),
    "r0": default_parastructure(1.2e-2, 1.5e-2),
    "theta0": default_parastructure(0.0, pi/2),
    "omega10": default_parastructure(2*pi/3, pi),
    "v10": default_parastructure(-5e-5, 5e-5),
    "v20": default_parastructure(-5e-5, 5e-5),
}

obj_info = {
    "o1": Objstructure.make_particle(5e5, 6e5),
    "o2": Objstructure.make_particle(5e5, 6e5),
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

G = 6.6e-11
acs0 = [sp.sympify("G*m2*(x2-x1)/r12**3"),
        sp.sympify("G*m1*(x1-x2)/r12**3"),
        sp.sympify("G*m2*(z2-z1)/r12**3"),
        sp.sympify("G*m1*(z1-z2)/r12**3")]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para('x0')
    z0 = exp_config.para('z0')
    r0 = exp_config.para('r0')
    theta0 = exp_config.para('theta0')
    omega10 = exp_config.para('omega10')
    omega20 = exp_config.para('omega10')
    v10 = exp_config.para('v10')
    v20 = exp_config.para('v20')

    x10 = x0 + r0*np.cos(theta0)
    x20 = x0 - r0*np.cos(theta0)
    z10 = z0 + r0*np.sin(theta0)
    z20 = z0 - r0*np.sin(theta0)
    vx10 = v10*np.cos(theta0) - omega10*r0*np.sin(theta0)
    vz10 = v10*np.sin(theta0) + omega10*r0*np.cos(theta0)
    vx20 = v20*np.cos(theta0) + omega20*r0*np.sin(theta0)
    vz20 = v20*np.sin(theta0) - omega20*r0*np.cos(theta0)

    m1 = exp_config.get_obj_para('o1', 'm')
    m2 = exp_config.get_obj_para('o2', 'm')
    numeric = {'m1': m1, 'm2': m2, 'G': G}
    acs = [ac.subs(numeric) for ac in acs0]
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    z0 = [x10, x20, z10, z20, vx10, vx20, vz10, vz20]

    def f(y, t):
        x1, x2, z1, z2, vx1, vx2, vz1, vz2 = y
        r12 = np.sqrt((x1-x2)**2 + (z1-z2)**2)
        numeric = {'x1': x1, 'x2': x2, 'z1': z1, 'z2': z2, 'r12': r12,
                   'vx1': vx1, 'vx2': vx2, 'vz1': vz1, 'vz2': vz2}
        acs_n = [ac.subs(numeric).evalf() for ac in acs]
        return [vx1, vx2, vz1, vz2] + acs_n
    sol = odeint(f, z0, t)
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    z1 = sol[:, 2]
    z2 = sol[:, 3]
    dist = np.sqrt((x1-x2)**2 + (z1-z2)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), z2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), dist + np.random.normal(0, error, t_num))
    return data_struct


def celestial_2_xz_config() -> ExpStructure:
    expconfig = ExpConfig("celestial_2_xz", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posz[o1] - posz[o2]) ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posy[o2]")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
