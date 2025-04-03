use super::expstructure::{
    DataStruct, DataStructOfDoExperiment, DataStructOfExpData, DoExpType, ExpConfig, ExpStructure,
    Objstructure, Parastructure,
};
use super::objects::obj::{ObjType, ATTR, DATA};
use super::simulation::collision::do_collision;
use crate::expdata::expdata::ExpData;
use crate::language::{AtomExp, Concept, Exp, MeasureType, Proposition};
use crate::parser::FromStr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

#[pymethods]
impl ObjType {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    #[new]
    #[inline]
    pub fn from_string(obj: String) -> PyResult<Self> {
        obj.parse().map_err(|x| PyValueError::new_err(x))
    }
}

#[pymethods]
impl Parastructure {
    #[new]
    pub fn from_string(content: String) -> PyResult<Self> {
        (&content).parse().map_err(|e| PyValueError::new_err(e))
    }
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
}

#[pymethods]
impl Objstructure {
    #[new]
    pub fn from_string(content: String) -> PyResult<Self> {
        Self::from_str(&content).map_err(|e| PyValueError::new_err(e))
    }
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn random_settings(&mut self) {
        self.random_sample();
    }
    #[staticmethod]
    fn make_particle(m_low: f64, m_high: f64) -> Self {
        Objstructure::particle((m_low, m_high))
    }
    #[staticmethod]
    fn make_elec_particle(m_low: f64, m_high: f64, e_low: f64, e_high: f64) -> Self {
        Objstructure::elec_particle((m_low, m_high), (e_low, e_high))
    }
    #[staticmethod]
    fn make_spring(k_low: f64, k_high: f64, l_low: f64, l_high: f64) -> Self {
        Objstructure::spring((k_low, k_high), (l_low, l_high))
    }
    #[staticmethod]
    pub fn clock() -> Self {
        Objstructure::new(ObjType::clock(), HashMap::from([]))
    }
    #[staticmethod]
    fn make_slope() -> Self {
        Objstructure::slope()
    }
}

#[pymethods]
impl ExpStructure {
    #[new]
    #[pyo3(signature = (exp_config=None, do_experiment=None))]
    pub fn __new__(exp_config: Option<ExpConfig>, do_experiment: Option<DoExpType>) -> Self {
        if exp_config.is_none() || do_experiment.is_none() {
            ExpStructure::empty()
        } else {
            ExpStructure::new(exp_config.unwrap(), do_experiment.unwrap())
        }
    }
    fn __getstate__(&self) -> String {
        serde_yaml::to_string(self).unwrap()
    }
    fn __setstate__(&mut self, state: String) {
        *self = serde_yaml::from_str(&state).unwrap();
    }
    #[getter]
    #[inline]
    fn get_all_ids(&self) -> HashSet<i32> {
        self.get_ref_expconfig()
            .obj_name_map
            .keys()
            .cloned()
            .collect()
    }
    #[inline]
    fn get_obj_type(&self, id: i32) -> ObjType {
        self.get_ref_expconfig()
            .obj_name_map
            .get(&id)
            .unwrap()
            .0
            .clone()
    }
    #[inline]
    pub fn get_obj(&self, id: i32) -> Objstructure {
        self.get_ref_expconfig().get_obj(id).clone()
    }
    #[getter]
    #[inline]
    fn obj_info(&self) -> HashMap<String, (ObjType, i32)> {
        let expdata = self.get_ref_expconfig();
        expdata.obj_id_map.clone()
    }
    #[getter]
    #[inline]
    fn data_info(&self) -> DataStruct {
        if self.expdata_is_none() {
            panic!("The expdata has not been collected yet.");
        }
        let expdata = self.get_ref_expdata();
        // for (key, value) in expdata.data.iter() {
        //     println!("{}: {}", key.0, key.1);
        // }
        expdata.data.clone()
    }
    #[getter]
    #[inline]
    fn get_exp_name(&self) -> String {
        self.get_ref_expconfig().name.clone()
    }
    #[getter]
    #[inline]
    fn get_spdim(&self) -> usize {
        self.get_ref_expconfig().spdim
    }
    #[getter]
    #[inline]
    fn get_original_data(&self) -> Vec<AtomExp> {
        self.get_ref_expconfig().get_original_data()
    }
    #[getter]
    #[inline]
    fn get_original_concept(&self) -> HashSet<Concept> {
        self.get_ref_expconfig().get_original_concept()
    }
    fn random_settings(&mut self) {
        self.random_sample();
    }
    fn random_set_exp_para(&mut self) {
        self.random_sample_exp_para();
    }
    fn random_set_obj(&mut self, id: i32) {
        self.random_sample_obj(id);
    }
    pub fn set_obj(&mut self, id: i32, obj: Objstructure) {
        self._set_obj(id, obj);
    }
    pub fn collect_expdata(&mut self, measuretype: MeasureType) -> PyResult<DataStructOfExpData> {
        let res = self.get_expdata(measuretype);
        match res {
            Ok(data) => Ok(data.clone()),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn print_geometry_info(&self) {
        self.get_ref_expconfig().print_geometry_info();
    }
    #[getter]
    fn get_geometry_info(&self) -> Vec<Proposition> {
        self.get_ref_expconfig().geometry_info.clone()
    }
}

#[pymethods]
impl DataStruct {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    #[getter]
    #[inline]
    fn get_data_keys(&self) -> HashSet<AtomExp> {
        self.get_data().keys().cloned().collect()
    }
    #[inline]
    fn fetch_data_by_name_ids(&self, name: &str, ids: Vec<i32>) -> Option<ExpData> {
        self.get_data()
            .get(&AtomExp::new_variable_ids(name.to_string(), ids))
            .cloned()
    }
    #[inline]
    fn fetch_data_by_key(&self, atom: AtomExp) -> Option<ExpData> {
        self.get_data().get(&atom).cloned()
    }
    #[inline]
    fn fetch_data_by_str(&self, atom_name: &str) -> Option<ExpData> {
        self.get_data()
            .get(&AtomExp::from_str(atom_name).unwrap())
            .cloned()
    }
    #[staticmethod]
    fn empty() -> Self {
        DataStruct::new(HashMap::new())
    }
    fn add_data(&mut self, atom: AtomExp, expdata: ExpData) {
        self.set_data(atom, expdata);
    }
    fn remove_data(&mut self, atom: AtomExp) {
        self.reset_data(atom);
    }
}

#[pymethods]
impl DataStructOfExpData {
    #[getter]
    #[inline]
    fn get_datastruct(&self) -> DataStruct {
        self.data.clone()
    }
}

#[pymethods]
impl DATA {
    #[new]
    fn __new__(obj: ObjType, name: &str) -> Self {
        DATA::Mk {
            obj,
            name: name.to_string(),
        }
    }
    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    #[staticmethod]
    fn particle_posx() -> Concept {
        DATA::posx()
    }
    #[staticmethod]
    fn particle_posy() -> Concept {
        DATA::posy()
    }
    #[staticmethod]
    fn particle_posz() -> Concept {
        DATA::posz()
    }
    #[staticmethod]
    fn particle_dist() -> Concept {
        DATA::dist()
    }
    #[staticmethod]
    fn spring_length() -> Concept {
        DATA::length()
    }
    #[staticmethod]
    fn clock_time() -> Concept {
        DATA::time()
    }
    #[staticmethod]
    fn slope_cx() -> Concept {
        DATA::cx()
    }
    #[staticmethod]
    fn slope_cy() -> Concept {
        DATA::cy()
    }
    #[staticmethod]
    fn slope_cz() -> Concept {
        DATA::cz()
    }
}

#[pymethods]
impl ATTR {
    #[new]
    #[inline]
    fn __new__(obj: ObjType, name: &str) -> Self {
        ATTR::Mk {
            obj,
            name: name.to_string(),
        }
    }
    /// print name
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
}

#[pymethods]
impl ExpConfig {
    #[new]
    #[inline]
    fn __new__(
        name: &str,
        spdim: usize,
        exp_para: HashMap<String, Parastructure>,
        obj_info: HashMap<String, Objstructure>,
        data_info: Vec<(Concept, Vec<String>)>,
    ) -> Self {
        ExpConfig::new(name.to_string(), spdim, exp_para, obj_info, data_info)
    }
    #[inline]
    pub fn para(&self, para_name: &str) -> f64 {
        self.exp_para
            .get(para_name)
            .unwrap()
            .real_value()
            .map_err(|e| format!("Error in getting para `{}`: {}", para_name, e))
            .unwrap()
    }
    #[inline]
    pub fn obj_para(&self, obj_name: &str, para_name: &ATTR) -> f64 {
        self.obj_info
            .get(obj_name)
            .unwrap()
            .get_para_real_value(para_name)
            .unwrap()
    }
    #[inline]
    pub fn get_obj_para(&self, obj_name: &str, para_name: &str) -> f64 {
        let obj_type = self.obj_info.get(obj_name).unwrap().obj_type.clone();
        let para_attr = ATTR::new(obj_type, para_name);
        self.obj_info
            .get(obj_name)
            .unwrap()
            .get_para_real_value(&para_attr)
            .unwrap()
    }
    #[getter]
    #[inline]
    fn get_original_data(&self) -> Vec<AtomExp> {
        let original_data = self.original_data();
        original_data
            .iter()
            .map(|(concept, obj_ids)| concept.to_atomexp(obj_ids.clone()).unwrap())
            .collect()
    }
    #[getter]
    #[inline]
    fn get_original_concept(&self) -> HashSet<Concept> {
        let original_data = self.original_data();
        let mut res = HashSet::new();
        for (concept, _) in original_data.iter() {
            res.insert(concept.clone());
        }
        res
    }
    fn new_datastruct_of_doexperiment(&self, t_num: usize) -> DataStructOfDoExperiment {
        self.create_data_struct_of_do_experiment(t_num)
    }
    fn random_settings(&mut self) {
        self.random_sample();
    }
    pub fn gen_prop(&self, content: String) -> Proposition {
        let content = self._gen_exp_string(content);
        content.parse().unwrap()
    }
    pub fn gen_exp(&self, content: String) -> Exp {
        let content = self._gen_exp_string(content);
        (&content).parse().unwrap()
    }
    pub fn register_geometry_info(&mut self, prop: Proposition) {
        self.geometry_info.push(prop);
    }
    fn print_geometry_info(&self) {
        println!("Geometry Info in {}:", self.name);
        for prop in self.geometry_info.iter() {
            println!("{}", prop);
        }
    }
}
impl ExpConfig {
    fn _gen_exp_string(&self, content: String) -> String {
        let re = Regex::new(r"\[([a-zA-Z0-9, ]+)\]").unwrap();
        // find all positions of re in content
        let res: Vec<(usize, usize)> = re
            .find_iter(&content)
            .map(|m| (m.start(), m.end()))
            .collect();
        if res.len() == 0 {
            return content;
        }
        let mut new_content = String::new();
        let mut last = 0;
        let subst_dict = self
            .obj_id_map
            .iter()
            .map(|(key, value)| (key.clone(), value.1))
            .collect::<HashMap<String, i32>>();
        for i in 0..res.len() {
            new_content.push_str(&content[last..res[i].0]);
            // '[a1, a2, b, c]', transform it to ['a1', 'a2', 'b', 'c'], and do the substitution
            let vec_ids = &content[(res[i].0 + 1)..(res[i].1 - 1)]
                .split(", ")
                .map(|x| subst_dict.get(x).unwrap().to_string())
                .collect::<Vec<String>>();
            // transform vec_ids to a string
            let slice_str = format!("[{}]", vec_ids.join(", "));
            new_content.push_str(&slice_str);
            last = res[i].1;
        }
        new_content.push_str(&content[last..]);
        new_content
    }
}

#[pymodule]
pub fn register_experiment(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DATA>()?;
    m.add_class::<ATTR>()?;
    m.add_class::<DoExpType>()?;
    m.add_class::<Objstructure>()?;
    m.add_class::<Parastructure>()?;
    m.add_class::<ObjType>()?;
    m.add_class::<ExpConfig>()?;
    m.add_class::<ExpStructure>()?;
    m.add_class::<DataStruct>()?;
    m.add_class::<DataStructOfExpData>()?;
    m.add_class::<DataStructOfDoExperiment>()?;
    m.add_function(wrap_pyfunction!(do_collision, m)?)?;
    Ok(())
}
