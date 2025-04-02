pub mod macros;
pub mod parser;
pub mod knowledge;
pub mod regression;
pub mod serialize;
pub mod semantic{
    pub mod evaluation;
    pub mod specialize;
    pub mod generalize;
    pub mod exprcharacter;
}
pub mod language;
pub mod expdata;
pub mod experiments;

mod impl_for_pyo3;
use expdata::register_data;
use experiments::register_experiment;
use parser::parsing::register_sentence;
use knowledge::Knowledge;
use semantic::exprcharacter::KeyValueHashed;
use regression::{
    search_relations_ver1,
    search_relations_ver2,
    search_binary_relations,
    search_trivial_relations,
    search_relations_ver3
};
use crate::experiments::simulation::{
    oscillation::struct_oscillation,
    collision::{struct_collision, do_collision},
    motion::struct_motion,
    motion0::struct_motion0,
    stringmotion0::struct_stringmotion0,
    oscillationy::struct_oscillationy,
    motion0y::struct_motion0y,
    motiony::struct_motiony
};
use pyo3::prelude::*;
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
