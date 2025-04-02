use pyo3::prelude::*;
use crate::r;
use crate::experiments::*;
use ndarray::Array1;
use std::collections::HashMap;

pub fn builtin_motion0(t_end: f64, t_num: usize, error: f64, exp_config: &ExpConfig) -> DataStructOfDoExperiment {
    let x0 = exp_config.para("x0");
    let v0 = exp_config.para("v0");
    let step = (t_end - 0.0) / (t_num as f64);
    let t: Array1<f64> = Array1::range(0.0, t_end, step);
    let x: Array1<f64> = x0 + v0 * &t;
    // Generate y and z with all zeros
    let y: Array1<f64> = Array1::zeros(t_num);
    let z: Array1<f64> = Array1::zeros(t_num);
    let mut data_struct = exp_config.create_data_struct_of_do_experiment(t_num);
    data_struct.add_data((DATA::time(), vec![r!("Clock")]), &add_errors(&t, error).unwrap());
    data_struct.add_data((DATA::posx(), vec![r!("MPa")]), &add_errors(&x, error).unwrap());
    data_struct.add_data((DATA::posy(), vec![r!("MPa")]), &add_errors(&y, error).unwrap());
    data_struct.add_data((DATA::posz(), vec![r!("MPa")]), &add_errors(&z, error).unwrap());
    data_struct
}

#[pyfunction]
pub fn struct_motion0() -> ExpStructure {
    let default_particle_struct = Objstructure::particle((1.0, 1000.0));
    let name = r!("motion0");
    let spdim = 1 as usize;
    let exp_para = HashMap::from([
        (r!("x0"), Parastructure::new(Some((9.0, 11.0)))),
        (r!("v0"), Parastructure::new(Some((-2.0, 2.0)))),
    ]);
    let obj_info = HashMap::from([
        (r!("MPa"), default_particle_struct),
        (r!("Clock"), Objstructure::clock()),
    ]);
    let data_info = vec![
        (DATA::posx(), vec![r!("MPa")]),
        (DATA::posy(), vec![r!("MPa")]),
        (DATA::posz(), vec![r!("MPa")]),
        (DATA::time(), vec![r!("Clock")]),
    ];
    let mut exp_config = ExpConfig::new(name, spdim, exp_para, obj_info, data_info);
    exp_config.register_geometry_info(exp_config.gen_prop(r!("posz[MPa] is zero")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!("posy[MPa] is zero")));
    let do_experiment: DoExpType = DoExpType::new(
        r!("<builtin_motion0>"),
        builtin_motion0
    );
    ExpStructure::new(exp_config, do_experiment)
}