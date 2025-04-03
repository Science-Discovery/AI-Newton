pub mod constdata;
pub mod conversions;
/// This module provides various data structures and functions for handling experimental data.
///
/// # Modules
/// - `expdata`: A module for handling general experimental data.
/// - `normaldata`: A module for handling normal data.
/// - `constdata`: A module for handling constant data.
/// - `zerodata`: A module for handling zero data.
/// - `conversions`: A module for converting between different data types.
///
/// # Re-exports
/// - `ExpData`: Struct for handling general experimental data.
/// - `Diff`: Trait for calculating derivatives for experimental data.
/// - `ZeroData`: Struct for handling zero data.
/// - `NormalData`: Struct for handling normal data.
/// - `ConstData`: Enum for handling constant data.
/// - `is_conserved`: Function to check if a list of data points (with error) is judged to be conserved.
/// - `is_conserved_mean_and_std`: Same as `is_conserved`, but exposed to Python with PyO3.
///
/// # Functions
/// - `is_conserved_const_list`: Checks if a list of `ConstData` is judged to be conserved.
///
/// # PyO3 Integration
/// This module can be used with Python through PyO3. The following classes and functions are exposed to Python:
/// - `ExpData`
/// - `NormalData`
/// - `ConstData`
/// - `is_conserved_const_list`
/// - `is_conserved_mean_and_std`
///
pub mod expdata;
pub mod normaldata;
pub mod zerodata;

pub use constdata::ConstData;
pub use expdata::Diff;
pub use expdata::ExpData;
pub use normaldata::NormalData;
pub use normaldata::{is_conserved, is_conserved_mean_and_std};
pub use zerodata::ZeroData;

use ndarray::Array1;
use pyo3::prelude::*;
#[pyfunction]
fn is_conserved_const_list(data: Vec<ConstData>) -> bool {
    let mut mean_vec = vec![];
    let mut std_vec = vec![];
    for x in data {
        match x {
            ConstData::Data { mean, std } => {
                mean_vec.push(mean);
                std_vec.push(std);
            }
            ConstData::Exact { value: _ } => return false,
        };
    }
    // vec to arr
    is_conserved(&Array1::from(mean_vec), &Array1::from(std_vec), None)
}

#[pymodule]
pub fn register_data(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExpData>()?;
    m.add_class::<NormalData>()?;
    m.add_class::<ConstData>()?;
    m.add_function(wrap_pyfunction!(is_conserved_const_list, m)?)?;
    m.add_function(wrap_pyfunction!(is_conserved_mean_and_std, m)?)?;
    Ok(())
}
