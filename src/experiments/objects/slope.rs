use crate::r;
use std::collections::HashMap;
use crate::language::Concept;
use crate::experiments::{
    Objstructure, ObjType, DATA
};

impl ObjType {
    pub fn slope() -> Self { ObjType::new("Slope") }
}
impl Objstructure {
    /// A slope object
    pub fn slope() -> Self {
        Objstructure::new(
            ObjType::slope(),
            HashMap::from([]),
        )
    }
}
impl DATA {
    /// Basic concept for Slope
    pub fn cx() -> Concept { DATA::data(vec![r!("Slope")], r!("cx")) }
    pub fn cy() -> Concept { DATA::data(vec![r!("Slope")], r!("cy")) }
    pub fn cz() -> Concept { DATA::data(vec![r!("Slope")], r!("cz")) }
}

