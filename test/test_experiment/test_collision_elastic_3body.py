# %%
import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

from aiphy import Knowledge, MeasureType, Concept
from aiphy.experiment.basic import collision_elastic_3body_config
from aiphy.dataplot import plot_datastruct, plot_data, plot_normaldata
import matplotlib.pyplot as plt

# %%
knowledge = Knowledge.default()
expstruct = collision_elastic_3body_config()
expstruct.random_settings()
res = expstruct.collect_expdata(MeasureType.default())

plot_datastruct(res.datastruct)
assert knowledge.eval("posz[1]", expstruct).is_zero
assert knowledge.eval("posz[2]", expstruct).is_zero
assert knowledge.eval("posz[3]", expstruct).is_zero
assert knowledge.eval("posy[3]", expstruct).is_zero
assert knowledge.eval("posy[2]", expstruct).is_zero
assert knowledge.eval("posy[1]", expstruct).is_zero

assert knowledge.eval("posx[1]'", expstruct).is_normal
assert knowledge.eval("posx[2]'", expstruct).is_normal
assert knowledge.eval("posx[3]'", expstruct).is_normal
concept = Concept.Mksum("Particle", Concept("(1->Particle) |- posx[1]''"))
knowledge.register_expr(str(concept), "sum_of_a")
assert knowledge.eval("sum_of_a", expstruct).is_zero

# %%
