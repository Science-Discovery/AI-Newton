mod topy;
use std::collections::HashMap;

pub use topy::register_experiment;
mod expstructure;
pub use expstructure::{
    add_errors, DataStruct, DataStructOfDoExperiment, DoExpType, ExpConfig, ExpStructure,
    Objstructure, Parastructure,
};
pub mod objects {
    pub mod clock;
    pub mod obj;
    pub mod particle;
    pub mod slope;
    pub mod spring;
}
pub use objects::obj::{ObjType, ATTR, DATA};
pub mod simulation {
    pub mod collision;
    pub mod motion;
    pub mod motion0;
    pub mod motion0y;
    pub mod motiony;
    pub mod oscillation;
    pub mod oscillationy;
    pub mod stringmotion0;
}

use crate::r;
use simulation::{
    collision::struct_collision, motion::struct_motion, motion0::struct_motion0,
    motion0y::struct_motion0y, motiony::struct_motiony, oscillation::struct_oscillation,
    oscillationy::struct_oscillationy, stringmotion0::struct_stringmotion0,
};
pub fn builtin_experiment() -> HashMap<String, ExpStructure> {
    HashMap::from([
        (r!("oscillation"), struct_oscillation()),
        (r!("collision"), struct_collision()),
        (r!("motion"), struct_motion()),
        (r!("motion0"), struct_motion0()),
        (r!("stringmotion0"), struct_stringmotion0()),
        (r!("oscillationy"), struct_oscillationy()),
        (r!("motion0y"), struct_motion0y()),
        (r!("motiony"), struct_motiony()),
    ])
}
