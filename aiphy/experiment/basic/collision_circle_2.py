import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t, concept_dist


pi = 3.1415926
exp_para = {
    "l1": default_parastructure(5.0, 7.0),
    "theta10": default_parastructure(0.0, pi/2.0),
    "omega10": default_parastructure(pi/4.0, pi/2.0),
    "l2": default_parastructure(3.0, 4.0),
    "theta20": default_parastructure(pi/2.0, pi),
    "omega20": default_parastructure(-pi/2.0, -pi/4.0),
}

obj_info = {
    "o1": Objstructure.make_particle(1, 2),
    "o2": Objstructure.make_particle(1, 2),
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


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    l1 = exp_config.para('l1')
    theta10 = exp_config.para('theta10')
    omega10 = exp_config.para('omega10')
    l2 = exp_config.para('l2')
    theta20 = exp_config.para('theta20')
    omega20 = exp_config.para('omega20')
    m1 = exp_config.get_obj_para("o1", "m")
    m2 = exp_config.get_obj_para("o2", "m")
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    t_collision = (theta20 - theta10) / (omega10 - omega20)

    # after collison:
    omega1 = (l1**2*m1*omega10 + l2**2*m2*(2*omega20-omega10))/(l1**2*m1 + l2**2*m2)
    omega2 = (l1**2*m1*(2*omega10-omega20) + l2**2*m2*omega20)/(l1**2*m1 + l2**2*m2)

    # motion trajectory
    data_theta1 = np.array([theta10+omega10*i if i < t_collision else theta10+omega10*t_collision+omega1*(i-t_collision) for i in t])
    data_theta2 = np.array([theta20+omega20*i if i < t_collision else theta20+omega20*t_collision+omega2*(i-t_collision) for i in t])

    data_x1 = np.cos(data_theta1) * l1
    data_y1 = np.sin(data_theta1) * l1
    data_x2 = np.cos(data_theta2) * l2
    data_y2 = np.sin(data_theta2) * l2
    data_dist = np.sqrt((data_x2 - data_x1)**2 + (data_y2 - data_y1)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o1']), data_x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o1']), data_y1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o1']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['o2']), data_x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['o2']), data_y2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posz, ['o2']), np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o1', 'o2']), data_dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['o2', 'o1']), data_dist + np.random.normal(0, error, t_num))
    return data_struct


def collision_circle_2_config() -> ExpStructure:
    expconfig = ExpConfig("collision_circle_2", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posx[o1] ** 2 + posy[o1] ** 2")))
    expconfig.register_geometry_info(Proposition.IsConserved(expconfig.gen_exp(
        "posx[o2] ** 2 + posy[o2] ** 2")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "posz[o2]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] - dist[o2, o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[o1, o2] ** 2 - (posx[o1] - posx[o2]) ** 2 - (posy[o1] - posy[o2]) ** 2")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
