use crate::experiments::*;
use crate::r;
use ndarray::Array1;
use pyo3::prelude::*;
use std::collections::HashMap;

pub fn builtin_stringmotion0(
    t_end: f64,
    t_num: usize,
    error: f64,
    exp_config: &ExpConfig,
) -> DataStructOfDoExperiment {
    let spa_length_value = exp_config.obj_para("SPa", &ATTR::freel());
    let step = (t_end - 0.0) / (t_num as f64);
    let t: Array1<f64> = Array1::range(0.0, t_end, step);
    let mut data_struct = exp_config.create_data_struct_of_do_experiment(t_num);
    data_struct.add_data(
        (DATA::time(), vec![r!("Clock")]),
        &add_errors(&t, error).unwrap(),
    );
    data_struct.add_data(
        (DATA::length(), vec![r!("SPa")]),
        &add_errors(&Array1::from_elem(t_num, spa_length_value), error).unwrap(),
    );
    data_struct
}

#[pyfunction]
pub fn struct_stringmotion0() -> ExpStructure {
    let default_spring_struct = Objstructure::spring((2.0, 2.2), (9.0, 11.0));
    let name = r!("stringmotion0");
    let spdim = 1 as usize;
    let exp_para = HashMap::from([]);
    let obj_info = HashMap::from([
        (r!("SPa"), default_spring_struct),
        (r!("Clock"), Objstructure::clock()),
    ]);
    let data_info = vec![
        (DATA::length(), vec![r!("SPa")]),
        (DATA::time(), vec![r!("Clock")]),
    ];
    let exp_config = ExpConfig::new(name, spdim, exp_para, obj_info, data_info);
    let do_experiment: DoExpType =
        DoExpType::new(r!("<builtin_stringmotion0>"), builtin_stringmotion0);
    ExpStructure::new(exp_config, do_experiment)
}
