pub mod knowledge;
pub mod macros;
pub mod parser;
pub mod regression;
pub mod serialize;
pub mod semantic {
    pub mod evaluation;
    pub mod exprcharacter;
    pub mod generalize;
    pub mod specialize;
}
pub mod expdata;
pub mod experiments;
pub mod language;

mod impl_for_pyo3;
use crate::experiments::simulation::{
    collision::{do_collision, struct_collision},
    motion::struct_motion,
    motion0::struct_motion0,
    motion0y::struct_motion0y,
    motiony::struct_motiony,
    oscillation::struct_oscillation,
    oscillationy::struct_oscillationy,
    stringmotion0::struct_stringmotion0,
};
use expdata::register_data;
use experiments::register_experiment;
use knowledge::Knowledge;
use parser::parsing::register_sentence;
use pyo3::prelude::*;
use regression::{
    search_binary_relations, search_relations_ver1, search_relations_ver2, search_relations_ver3,
    search_trivial_relations,
};
use semantic::exprcharacter::KeyValueHashed;
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_sentence(m)?;
    m.add_class::<language::BinaryOp>()?;
    m.add_class::<language::UnaryOp>()?;
    m.add_class::<language::Proposition>()?;
    m.add_class::<language::Exp>()?;
    m.add_class::<language::SExp>()?;
    m.add_class::<language::Concept>()?;
    m.add_class::<language::AtomExp>()?;
    m.add_class::<language::Expression>()?;
    m.add_class::<language::IExpConfig>()?;
    m.add_class::<language::Intrinsic>()?;
    m.add_class::<language::MeasureType>()?;
    m.add_class::<Knowledge>()?;
    m.add_class::<KeyValueHashed>()?;
    m.add_function(wrap_pyfunction!(search_relations_ver1, m)?)?;
    m.add_function(wrap_pyfunction!(search_relations_ver2, m)?)?;
    m.add_function(wrap_pyfunction!(search_binary_relations, m)?)?;
    m.add_function(wrap_pyfunction!(search_trivial_relations, m)?)?;
    m.add_function(wrap_pyfunction!(search_relations_ver3, m)?)?;
    m.add_function(wrap_pyfunction!(struct_motion0, m)?)?;
    m.add_function(wrap_pyfunction!(struct_motion, m)?)?;
    m.add_function(wrap_pyfunction!(struct_collision, m)?)?;
    m.add_function(wrap_pyfunction!(struct_oscillation, m)?)?;
    m.add_function(wrap_pyfunction!(do_collision, m)?)?;
    m.add_function(wrap_pyfunction!(struct_stringmotion0, m)?)?;
    m.add_function(wrap_pyfunction!(struct_oscillationy, m)?)?;
    m.add_function(wrap_pyfunction!(struct_motion0y, m)?)?;
    m.add_function(wrap_pyfunction!(struct_motiony, m)?)?;
    register_experiment(m)?;
    register_data(m)?;
    Ok(())
}
