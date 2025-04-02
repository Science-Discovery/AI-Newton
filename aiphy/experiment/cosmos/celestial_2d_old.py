import numpy as np
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_t, concept_dist
from scipy.integrate import odeint

exp_para = {
    "x10": default_parastructure(500.0, 520.0),
    "y10": default_parastructure(500.0, 520.0),
    "vx10": default_parastructure(-10.0, -8.0),
    "vy10": default_parastructure(-2.0, 0.0),
    "x20": default_parastructure(450.0, 460.0),
    "y20": default_parastructure(450.0, 460.0),
    "vx20": default_parastructure(10.0, 12.0),
    "vy20": default_parastructure(0.0, 2.0),
}
obj_info = {
    "earth": Objstructure.make_particle(800, 1000),
    "moon": Objstructure.make_particle(300, 400),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["earth"]),
    (concept_posy, ["earth"]),
    (concept_posx, ["moon"]),
    (concept_posy, ["moon"]),
    (concept_dist, ["earth", "moon"]),
    (concept_dist, ["moon", "earth"]),
    (concept_t, ["clock"]),
]

G = 0.0023456789

def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    x10 = exp_config.para('x10')
    y10 = exp_config.para('y10')
    vx10 = exp_config.para('vx10')
    vy10 = exp_config.para('vy10')
    x20 = exp_config.para('x20')
    y20 = exp_config.para('y20')
    vx20 = exp_config.para('vx20')
    vy20 = exp_config.para('vy20')
    earth_mass = exp_config.get_obj_para('earth', 'm')
    moon_mass = exp_config.get_obj_para('moon', 'm')
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    y0 = [x10, y10, vx10, vy10, x20, y20, vx20, vy20]
    def f(y, t):
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
        r = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        ax1 = -G * moon_mass * (x1-x2) / r**3
        ay1 = -G * moon_mass * (y1-y2) / r**3
        ax2 = -G * earth_mass * (x2-x1) / r**3
        ay2 = -G * earth_mass * (y2-y1) / r**3
        return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]
    sol = odeint(f, y0, t)
    x1 = sol[:, 0]
    y1 = sol[:, 1]
    x2 = sol[:, 4]
    y2 = sol[:, 5]
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    data_struct = exp_config.new_datastruct_of_doexperiment(t_num)
    data_struct.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['earth']), x1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['earth']), y1 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posx, ['moon']), x2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_posy, ['moon']), y2 + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['earth', 'moon']), dist + np.random.normal(0, error, t_num))
    data_struct.insert_data((concept_dist, ['moon', 'earth']), dist + np.random.normal(0, error, t_num))
    return data_struct

def celestial_2d_config() -> ExpStructure:
    expconfig = ExpConfig("celestial_2d", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)

    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[earth, moon] - dist[moon, earth]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(
        "dist[earth, moon] ** 2 - (posx[earth] - posx[moon]) ** 2 - (posy[earth] - posy[moon]) ** 2")))

    expstructure = ExpStructure(expconfig, doexp)
    return expstructure
