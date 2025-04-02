mod topy;
use std::collections::HashMap;

pub use topy::register_experiment;
mod expstructure;
pub use expstructure::{
    DataStructOfDoExperiment,
    Parastructure,
    ExpStructure,
    Objstructure,
    add_errors,
    DoExpType,
    DataStruct,
    ExpConfig
};
pub mod objects{
    pub mod obj;
    pub mod clock;
    pub mod particle;
    pub mod spring;
    pub mod slope;
}
pub use objects::obj::{ObjType, ATTR, DATA};
pub mod simulation{
    pub mod motion0;
    pub mod motion;
    pub mod collision;
    pub mod oscillation;
    pub mod stringmotion0;
    pub mod oscillationy;
    pub mod motion0y;
    pub mod motiony;
}

use simulation::{
    motion0::struct_motion0,
    motion::struct_motion,
    collision::struct_collision,
    oscillation::struct_oscillation,
    stringmotion0::struct_stringmotion0,
    oscillationy::struct_oscillationy,
    motion0y::struct_motion0y,
    motiony::struct_motiony
};
use crate::r;
pub fn builtin_experiment() -> HashMap<String, ExpStructure> {
    HashMap::from([
        (r!("oscillation"), struct_oscillation()),
        (r!("collision"), struct_collision()),
        (r!("motion"), struct_motion()),
        (r!("motion0"), struct_motion0()),
        (r!("stringmotion0"), struct_stringmotion0()),
        (r!("oscillationy"), struct_oscillationy()),
        (r!("motion0y"), struct_motion0y()),
        (r!("motiony"), struct_motiony())
    ])
}