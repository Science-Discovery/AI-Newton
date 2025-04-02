use crate::r;
use std::collections::HashMap;
use crate::language::Concept;
use crate::experiments::{
    Parastructure, Objstructure, ObjType, ATTR, DATA
};

impl ObjType {
    pub fn particle() -> Self { ObjType::new("Particle") }
}

/// In an experiment, each particle object corresponds to two adjustable knobs,
/// one for adjusting its mass (range is `mass_range`), and the other for adjusting its charge (range is `elec_range`)
impl Objstructure {
    pub fn particle(mass_range: (f64, f64)) -> Self {
        assert!(mass_range.0 >= 0.0 && mass_range.0 <= mass_range.1);
        Objstructure::new(
            ObjType::particle(),
            HashMap::from([
                (ATTR::mass(),
                    Parastructure::new(Some(mass_range))),
                (ATTR::elec(),
                    Parastructure::new(Some((0.0, 0.0)))),
            ]),
        )
    }
    pub fn elec_particle(mass_range: (f64, f64), elec_range: (f64, f64)) -> Self {
        assert!(mass_range.0 >= 0.0 && mass_range.1 <= 1000.0 && mass_range.0 <= mass_range.1);
        assert!(elec_range.0 >= -100.0 && elec_range.1 <= 100.0 && elec_range.0 <= elec_range.1);
        Objstructure::new(
            ObjType::particle(),
            HashMap::from([
                (ATTR::mass(),
                    Parastructure::new(Some(mass_range))),
                (ATTR::elec(),
                    Parastructure::new(Some(elec_range))),
            ]),
        )
    }
}
impl ATTR {
    /// attribute for a particle object
    /// (these attributes can be accessed by experimenters, but is not exposed to theorists).
    pub fn mass() -> Self { ATTR::new(ObjType::particle(),"m") }
    pub fn elec() -> Self { ATTR::new(ObjType::particle(),"e") }
}
impl DATA {
    /// `posx`, `posy`, `posz` are concepts related to `Particle` object.
    /// `dist` is a concept related to two `Particle` objects.
    pub fn posx() -> Concept { DATA::data(vec![r!("Particle")], r!("posx")) }
    pub fn posy() -> Concept { DATA::data(vec![r!("Particle")], r!("posy")) }
    pub fn posz() -> Concept { DATA::data(vec![r!("Particle")], r!("posz")) }
    pub fn dist() -> Concept { DATA::data(vec![r!("Particle"), r!("Particle")], r!("dist")) }
}
