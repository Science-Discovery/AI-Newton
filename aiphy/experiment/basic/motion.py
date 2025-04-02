import numpy as np
from math import fabs
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_cx, concept_cy, concept_cz


pi = 3.1415926

exp_para = {
    "theta": default_parastructure(pi/6, pi/3),
    "phi": default_parastructure(pi/6, pi/3),
    "x0": default_parastructure(1.0, 5.0),
    "y0": default_parastructure(1.0, 5.0),
    "z0": default_parastructure(-2.0, -1.0),
    "v0": default_parastructure(1.0, 2.0)
}
obj_info = {
    "o1": Objstructure.make_particle(1, 2),
    "clock": Objstructure.clock(),
    "slope": Objstructure.make_slope()
}
data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_cx, ["slope"]),
    (concept_cy, ["slope"]),
    (concept_cz, ["slope"]),
    (concept_t, ["clock"]),
]


g = 9.801234567


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x0 = exp_config.para('x0')
    y0 = exp_config.para('y0')
    z0 = exp_config.para('z0')
    theta = exp_config.para('theta')
    phi = exp_config.para('phi')
    v0 = exp_config.para('v0')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    dis = 1/2 * g*np.cos(theta) * t ** 2 + v0 * t
    x = x0 + dis*np.sin(theta)*np.cos(phi)
    y = y0 + dis*np.sin(theta)*np.sin(phi)
    z = z0 - dis*np.cos(theta)
    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['o1']), x + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), y + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), z + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cx, ['slope']), np.cos(theta)*np.cos(phi) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cy, ['slope']), np.cos(theta)*np.sin(phi) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_cz, ['slope']), np.sin(theta) + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def motion_3d_config() -> ExpStructure:
    expconfig = ExpConfig("motion_3d", 3, exp_para, obj_info, data_info)
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
        "cy[slope] * posx[o1] - cx[slope] * posy[o1] is conserved"
    ))
    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def motion_3d_test():
    expconfig = ExpConfig("motion_3d", 3, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
