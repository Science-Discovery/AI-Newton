from aiphy import (
    DoExpType, ExpConfig, Parastructure,
    Objstructure, Concept, ExpStructure, DATA,
    Proposition, Exp, Concept
)

# Concept for Particle
concept_posx = DATA.particle_posx()
concept_posy = DATA.particle_posy()
concept_posz = DATA.particle_posz()
concept_dist = DATA.particle_dist()

# Concept for Slope
concept_cx = DATA.slope_cx()
concept_cy = DATA.slope_cy()
concept_cz = DATA.slope_cz()

# Concept for Spring
concept_length = DATA.spring_length()

# Concept for Particle and Pulley
concept_distfix = Concept("(1->Particle) (2->Fixpoint) |- distfix[1,2]")

# Concept for Clock
concept_t = DATA.clock_time()

def default_parastructure(l: float, r: float) -> Parastructure:
    return Parastructure(f"[Parastructure] value: None, range: ({l}, {r})")
