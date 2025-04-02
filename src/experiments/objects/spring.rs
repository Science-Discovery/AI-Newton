use crate::r;
use crate::language::Concept;
use std::collections::HashMap;
use crate::experiments::{
    Parastructure, Objstructure, ObjType, DATA, ATTR
};

impl ObjType {
    pub fn spring() -> Self { ObjType::new("Spring") }
}

/// In an experiment, each spring object corresponds to two adjustable knobs,
/// one adjusting its thickness (range is `thickness_range`), and the other adjusting its free length (range is `freel_range`).
impl Objstructure {
    pub fn spring(thickness_range: (f64, f64), freel_range: (f64, f64)) -> Self {
        assert!(thickness_range.0 <= thickness_range.1 && freel_range.0 <= freel_range.1);
        assert!(thickness_range.0 >= 1.5 && thickness_range.1 <= 2.5);
        assert!(freel_range.0 >= 1.0 && freel_range.1 <= 10000.0);
        Objstructure::new(
            ObjType::spring(),
            HashMap::from([
                (ATTR::thickness(),
                    Parastructure::new(Some(thickness_range))),
                (ATTR::freel(),
                    Parastructure::new(Some(freel_range))),
            ]),
        )
    }
}
impl ATTR {
    /// Attribute of a spring object (these attributes can be accessed by experimenters, but is not exposed to theorists).
    pub fn thickness() -> Self { ATTR::new(ObjType::spring(), "thickness") }
    pub fn freel() -> Self { ATTR::new(ObjType::spring(), "freel") }
}
impl DATA {
    /// Basic concept of a spring object.
    pub fn length() -> Concept { DATA::data(vec![r!("Spring")], r!("length")) }
}
