use crate::experiments::*;
use crate::r;
use ndarray::Array1;
use pyo3::prelude::*;
use std::collections::HashMap;

pub fn builtin_motiony(
    t_end: f64,
    t_num: usize,
    error: f64,
    exp_config: &ExpConfig,
) -> DataStructOfDoExperiment {
    let y0 = exp_config.para("y0");
    let z0 = exp_config.para("z0");
    let v0 = exp_config.para("v0");
    let theta = exp_config.para("theta");
    let step = (t_end - 0.0) / (t_num as f64);
    let t: Array1<f64> = Array1::range(0.0, t_end, step);
    let a = 9.801234567 * theta.sin();
    let dis = v0 * &t + 0.5 * a * t.mapv(|x| x.powi(2));
    let x: Array1<f64> = Array1::zeros(t_num);
    let y: Array1<f64> = y0 + &dis * theta.cos();
    let z: Array1<f64> = z0 - &dis * theta.sin();
    let mut data_struct = exp_config.create_data_struct_of_do_experiment(t_num);
    data_struct.add_data(
        (DATA::time(), vec![r!("Clock")]),
        &add_errors(&t, error).unwrap(),
    );
    data_struct.add_data(
        (DATA::posx(), vec![r!("MPa")]),
        &add_errors(&x, error).unwrap(),
    );
    data_struct.add_data(
        (DATA::posy(), vec![r!("MPa")]),
        &add_errors(&y, error).unwrap(),
    );
    data_struct.add_data(
        (DATA::posz(), vec![r!("MPa")]),
        &add_errors(&z, error).unwrap(),
    );
    data_struct.add_data(
        (DATA::cy(), vec![r!("Slope")]),
        &add_errors(&Array1::from_elem(t_num, theta.sin()), error).unwrap(),
    );
    data_struct.add_data(
        (DATA::cz(), vec![r!("Slope")]),
        &add_errors(&Array1::from_elem(t_num, theta.cos()), error).unwrap(),
    );
    data_struct
}

#[pyfunction]
pub fn struct_motiony() -> ExpStructure {
    let default_particle_struct = Objstructure::particle((1.0, 1000.0));
    let name = r!("motiony");
    let spdim = 2 as usize;
    let exp_para = HashMap::from([
        (r!("y0"), Parastructure::new(Some((9.0, 11.0)))),
        (r!("z0"), Parastructure::new(Some((9.0, 11.0)))),
        (r!("v0"), Parastructure::new(Some((-2.0, 2.0)))),
        (r!("theta"), Parastructure::new(Some((0.2, 0.4)))),
    ]);
    let obj_info = HashMap::from([
        (r!("MPa"), default_particle_struct),
        (r!("Clock"), Objstructure::clock()),
        (r!("Slope"), Objstructure::slope()),
    ]);
    let data_info = vec![
        (DATA::posx(), vec![r!("MPa")]),
        (DATA::posy(), vec![r!("MPa")]),
        (DATA::posz(), vec![r!("MPa")]),
        (DATA::cy(), vec![r!("Slope")]),
        (DATA::cz(), vec![r!("Slope")]),
        (DATA::time(), vec![r!("Clock")]),
    ];
    let mut exp_config = ExpConfig::new(name, spdim, exp_para, obj_info, data_info);
    exp_config.register_geometry_info(exp_config.gen_prop(r!("posx[MPa] is zero")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!("cy[Slope] is conserved")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!("cz[Slope] is conserved")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!(
        "cy[Slope] * posy[MPa] + cz[Slope] * posz[MPa] is conserved"
    )));
    let do_experiment: DoExpType = DoExpType::new(r!("<builtin_motiony>"), builtin_motiony);
    ExpStructure::new(exp_config, do_experiment)
}
