use pyo3::prelude::*;
use crate::r;
use crate::experiments::*;
use ndarray::Array1;
use std::collections::HashMap;

pub fn builtin_oscillationy(t_end: f64, t_num: usize, error: f64, exp_config: &ExpConfig) -> DataStructOfDoExperiment {
    let y1 = exp_config.para("posl");
    let y2 = exp_config.para("y2");
    let v2 = exp_config.para("v2");
    let mp1_mass_value = exp_config.obj_para("MPa", &ATTR::mass());
    let sp2_length_value = exp_config.obj_para("SPb", &ATTR::freel());
    let sp2_k_value = exp_config.obj_para("SPb", &ATTR::thickness()).powf(3.);
    let step = (t_end - 0.0) / (t_num as f64);
    let t: Array1<f64> = Array1::range(0.0, t_end, step);
    let sp2_l = Array1::from_elem(t_num, y1);
    let omega = (sp2_k_value / mp1_mass_value).sqrt();
    let length = if y2 > y1 {
        sp2_length_value + (v2 / omega) * (omega * &t).mapv(|y| y.sin()) + (y2 - y1 - sp2_length_value) * (omega * &t).mapv(|y| y.cos())
    } else {
        - sp2_length_value + (v2 / omega) * (omega * &t).mapv(|y| y.sin()) + (y2 - y1 + sp2_length_value) * (omega * &t).mapv(|y| y.cos())
    };
    let sp2_r = sp2_l + &length;
    // Generate x and z with all zeros
    let x: Array1<f64> = Array1::zeros(t_num);
    let z: Array1<f64> = Array1::zeros(t_num);
    let mut data_struct = exp_config.create_data_struct_of_do_experiment(t_num);
    data_struct.add_data((DATA::time(), vec![r!("Clock")]), &add_errors(&t, error).unwrap());
    data_struct.add_data((DATA::posx(), vec![r!("MPa")]), &add_errors(&x, error).unwrap());
    data_struct.add_data((DATA::posy(), vec![r!("MPa")]), &add_errors(&sp2_r, error).unwrap());
    data_struct.add_data((DATA::posz(), vec![r!("MPa")]), &add_errors(&z, error).unwrap());
    data_struct.add_data((DATA::length(), vec![r!("SPb")]), &add_errors(&length, error).unwrap());
    data_struct
}

#[pyfunction]
pub fn struct_oscillationy() -> ExpStructure {
    let default_particle_struct = Objstructure::particle((1.2, 5.0));
    let default_spring_struct = Objstructure::spring((2.0, 2.2), (9.0, 11.0));
    let name = r!("oscillationy");
    let spdim = 1 as usize;
    let exp_para = HashMap::from([
        (r!("posl"), Parastructure::new(Some((-1.0, 1.0)))),
        (r!("y2"), Parastructure::new(Some((9.0, 11.0)))),
        (r!("v2"), Parastructure::new(Some((-2.0, 2.0)))),
    ]);
    let obj_info = HashMap::from([
        (r!("MPa"), default_particle_struct),
        (r!("SPb"), default_spring_struct),
        (r!("Clock"), Objstructure::clock()),
    ]);
    let data_info = vec![
        (DATA::posx(), vec![r!("MPa")]),
        (DATA::posy(), vec![r!("MPa")]),
        (DATA::posz(), vec![r!("MPa")]),
        (DATA::length(), vec![r!("SPb")]),
        (DATA::time(), vec![r!("Clock")]),
    ];
    let mut exp_config = ExpConfig::new(name, spdim, exp_para, obj_info, data_info);
    let do_experiment: DoExpType = DoExpType::new(
        r!("<builtin_oscillationy>"),
        builtin_oscillationy
    );
    exp_config.register_geometry_info(exp_config.gen_prop(r!("posx[MPa] is zero")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!("posz[MPa] is zero")));
    exp_config.register_geometry_info(exp_config.gen_prop(r!("length[SPb] - posy[MPa] is conserved")));
    ExpStructure::new(exp_config, do_experiment)
}