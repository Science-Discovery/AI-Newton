import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist


pi = 3.1415926
exp_para = {
    "x1": default_parastructure(-4.0, -3.0),
    "v1": default_parastructure(3.0, 5.0),
    "x2": default_parastructure(3.0, 4.0),
    "v2": default_parastructure(-5.0, -3.0),
    "y0": default_parastructure(-2.0, 2.0),
    "theta": default_parastructure(0, pi),
    "phi": default_parastructure(pi/6, pi),
}
obj_info = {
    "MPa": Objstructure.make_particle(2.0, 10.0),
    "MPb": Objstructure.make_particle(2.0, 10.0),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["MPa"]),
    (concept_posx, ["MPb"]),
    (concept_posy, ["MPa"]),
    (concept_posy, ["MPb"]),
    (concept_posz, ["MPa"]),
    (concept_posz, ["MPb"]),
    (concept_dist, ["MPa", "MPb"]),
    (concept_dist, ["MPb", "MPa"]),
    (concept_t, ["clock"]),
]


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    theta = exp_config.para("theta")
    phi = exp_config.para("phi")
    x1 = exp_config.para("x1")
    x2 = exp_config.para("x2")
    v1 = exp_config.para("v1")
    v2 = exp_config.para("v2")
    y0 = exp_config.para("y0")
    m1 = exp_config.get_obj_para("MPa", "m")
    m2 = exp_config.get_obj_para("MPb", "m")
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    t_collision = (x2 - x1) / (v1 - v2)
    assert t_collision > 0.0
    assert t_collision < t_end

    vx1 = np.cos(theta)**2*(m1*v1+m2*v2-np.sqrt(m2**2*(v1-v2)**2-(m1+m2)*v1*(m1*v1-m2*v1+2*m2*v2)*np.tan(theta)**2))/(m1+m2)
    vx2 = np.cos(theta)**2*(m1*m2*v1+m2**2*v2+(m1+m2)*(m1*v1+m2*v2)*np.tan(theta)**2+m1*np.sqrt(m2**2*(v1-v2)**2-(m1+m2)*v1*(m1*v1-m2*v1+2*m2*v2)*np.tan(theta)**2))/(m2*(m1 + m2))
    vy1 = np.cos(theta)*np.sin(theta)*(m1*v1+m2*v2-np.sqrt(m2**2*(v1-v2)**2-(m1+m2)*v1*(m1*v1-m2*v1+2*m2*v2)*np.tan(theta)**2))/(m1 + m2)
    vy2 = np.cos(theta)*np.sin(theta)*m1*(-m1*v1-m2*v2+np.sqrt(m2**2*(v1-v2)**2-(m1+m2)*v1*(m1*v1-m2*v1+2*m2*v2)*np.tan(theta)**2))/(m2*(m1 + m2))

    data_x1 = np.array([
        (x1 + v1 * i) if i < t_collision else (x1 + v1 * t_collision + vx1 * (i - t_collision))
        for i in t
    ])
    data_x2 = np.array([
        (x2 + v2 * i) if i < t_collision else (x2 + v2 * t_collision + vx2 * (i - t_collision))
        for i in t
    ])
    data_y1 = np.array([
        y0 if i < t_collision else (y0 + vy1 * (i - t_collision))
        for i in t
    ])
    data_y2 = np.array([
        y0 if i < t_collision else (y0 + vy2 * (i - t_collision))
        for i in t
    ])

    datat_x1 = np.cos(phi) * data_x1 - np.sin(phi) * data_y1
    datat_y1 = np.sin(phi) * data_x1 + np.cos(phi) * data_y1
    datat_x2 = np.cos(phi) * data_x2 - np.sin(phi) * data_y2
    datat_y2 = np.sin(phi) * data_x2 + np.cos(phi) * data_y2
    data_dist = np.sqrt((data_x2 - data_x1)**2 + (data_y2 - data_y1)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_posx, ['MPa']), datat_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['MPb']), datat_x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPa']), datat_y1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['MPb']), datat_y2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPa']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['MPb']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPa', 'MPb']), data_dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['MPb', 'MPa']), data_dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return data_struct


def collision_nh_config() -> ExpStructure:
    expconfig = ExpConfig("collision_nh", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(expconfig.gen_prop("posz[MPa] is zero"))
    expconfig.register_geometry_info(expconfig.gen_prop("posz[MPb] is zero"))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPa, MPb] - dist[MPb, MPa]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[MPa, MPb] ** 2 - (posx[MPa] - posx[MPb]) ** 2 - (posy[MPa] - posy[MPb]) ** 2")))
    expstructure = ExpStructure(expconfig, doexp)
    return expstructure


def collision_nh_test():
    expconfig = ExpConfig("collision_nh", 2, exp_para, obj_info, data_info)
    expconfig.random_settings()
    do_experiment(10, 100, 0.1, expconfig)
