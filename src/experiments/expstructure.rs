use crate::expdata::ExpData;
use crate::experiments::{ObjType, ATTR};
use crate::language::{AtomExp, Concept, MeasureType, Proposition};
use crate::r;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayBase, Dimension, OwnedRepr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};
use std::{collections::HashSet, fs};

#[pyclass]
#[derive(Clone)]
pub struct DoExpType {
    pub fun: Option<fn(f64, usize, f64, &ExpConfig) -> DataStructOfDoExperiment>,
    pub name: String,
}

fn get_relative_path(target_path: &std::path::Path) -> Option<std::path::PathBuf> {
    // 将路径转换为绝对路径
    let base_abs = std::env::current_dir().ok()?;
    let target_abs = target_path.canonicalize().ok()?;
    // 计算相对路径
    target_abs
        .strip_prefix(base_abs)
        .ok()
        .map(std::path::PathBuf::from)
}
fn get_relative_path_string(path: &str) -> Option<String> {
    let path = std::path::Path::new(path);
    get_relative_path(path).map(|x| x.to_str().unwrap().to_string())
}

#[pymethods]
impl DoExpType {
    #[new]
    pub fn __new__(name: String) -> PyResult<Self> {
        Self::new_with_name(name).map_err(|e| PyValueError::new_err(e))
    }
}

impl DoExpType {
    pub fn empty() -> Self {
        DoExpType {
            fun: None,
            name: "".to_string(),
        }
    }
    pub fn new(
        builtin_name: String,
        builtin_fun: fn(f64, usize, f64, &ExpConfig) -> DataStructOfDoExperiment,
    ) -> Self {
        DoExpType {
            fun: Some(builtin_fun),
            name: builtin_name,
        }
    }
    pub fn new_with_name(name: String) -> Result<Self, String> {
        use super::simulation::{
            collision::builtin_collision, motion::builtin_motion, motion0::builtin_motion0,
            motion0y::builtin_motion0y, motiony::builtin_motiony, oscillation::builtin_oscillation,
            oscillationy::builtin_oscillationy, stringmotion0::builtin_stringmotion0,
        };
        if name == r!("<builtin_oscillation>") {
            Ok(DoExpType::new(name, builtin_oscillation))
        } else if name == r!("<builtin_collision>") {
            Ok(DoExpType::new(name, builtin_collision))
        } else if name == r!("<builtin_stringmotion0>") {
            Ok(DoExpType::new(name, builtin_stringmotion0))
        } else if name == r!("<builtin_motion0>") {
            Ok(DoExpType::new(name, builtin_motion0))
        } else if name == r!("<builtin_motion>") {
            Ok(DoExpType::new(name, builtin_motion))
        } else if name == r!("<builtin_oscillationy>") {
            Ok(DoExpType::new(name, builtin_oscillationy))
        } else if name == r!("<builtin_motion0y>") {
            Ok(DoExpType::new(name, builtin_motion0y))
        } else if name == r!("<builtin_motiony>") {
            Ok(DoExpType::new(name, builtin_motiony))
        } else {
            DoExpType::new_with_filename(name)
        }
    }
    pub fn new_with_filename(filename: String) -> Result<Self, String> {
        get_relative_path_string(&filename).map_or(
            Err(format!(
                "DoExpType: Failed to get relative path of {}",
                filename
            )),
            |x| Ok(DoExpType { fun: None, name: x }),
        )
    }
    pub fn do_experiment(
        &self,
        t_end: f64,
        t_num: usize,
        error: f64,
        exp_config: &ExpConfig,
    ) -> Result<DataStructOfDoExperiment, String> {
        if let Some(fun) = self.fun {
            Ok(fun(t_end, t_num, error, exp_config))
        } else {
            let py_app = fs::read_to_string(self.name.clone()).unwrap();
            let res = Python::with_gil(|py| -> Result<DataStructOfDoExperiment, PyErr> {
                let app: Py<PyAny> = PyModule::from_code_bound(py, &py_app, "", "")?
                    .getattr("do_experiment")?
                    .into();
                // println!("DoExpType: Call Python function.");
                let res = app.call1(py, (t_end, t_num, error, exp_config.clone()))?;
                res.extract::<DataStructOfDoExperiment>(py)
            });
            // println!("DoExpType: Call Python function done.");
            match res {
                Ok(data) => Ok(data),
                Err(err) => Err(format!(
                    "DoExpType in experiment <{}>, Error in Python function: {}",
                    self.name, err
                )),
            }
        }
    }
}

// 刻画某个参数结构的抽象类
// range: 参数的取值范围
#[pyclass]
#[derive(Clone)]
pub struct Parastructure {
    value: Option<f64>,
    range: (f64, f64),
}

impl Parastructure {
    pub fn new(range: Option<(f64, f64)>) -> Self {
        Parastructure {
            value: None,
            range: range.unwrap_or((-1e10, 1e10)),
        }
    }
    pub fn new_with_value(value: Option<f64>, range: (f64, f64)) -> Self {
        Parastructure { value, range }
    }
    fn random_sample(&mut self) {
        let mut rng = rand::thread_rng();
        self.value = Some(if self.range.0 == self.range.1 {
            self.range.0
        } else {
            rng.gen_range(self.range.0..self.range.1)
        });
    }
    pub fn real_value(&self) -> Result<f64, String> {
        match self.value {
            Some(value) => Ok(value),
            None => Err("Parastructure: value is None.".to_string()),
        }
    }
}

impl fmt::Display for Parastructure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[Parastructure] value: {:?}, range: {:?}",
            self.value, self.range
        )
        .unwrap();
        Result::Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Objstructure {
    pub obj_type: ObjType,
    attribute: HashMap<ATTR, Parastructure>,
}
impl Objstructure {
    pub fn new(obj_type: ObjType, attribute: HashMap<ATTR, Parastructure>) -> Self {
        Objstructure {
            obj_type,
            attribute,
        }
    }
    pub fn random_sample(&mut self) {
        for (_, para) in self.attribute.iter_mut() {
            para.random_sample();
        }
    }
    // fn real_value(&self) -> HashMap<ATTR, f64> {
    //     let mut result = HashMap::new();
    //     for (name, para) in self.attribute.iter() {
    //         result.insert(name.clone(), para.real_value());
    //     }
    //     result
    // }
    pub fn get_para_real_value(&self, para_name: &ATTR) -> Result<f64, String> {
        self.attribute
            .get(para_name)
            .unwrap()
            .real_value()
            .map_err(|x| format!("Error in get para of `{}`: {}", para_name, x))
    }
    // fn set_value(&mut self, value_dict: HashMap<ATTR, f64>) {
    //     for (name, value) in value_dict.iter() {
    //         self.attribute.get_mut(name).unwrap().set_value(*value);
    //     }
    // }
}

struct ListATTR(Vec<ATTR>);

impl fmt::Display for ListATTR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let attr_list = self.0.iter().map(|x| x.to_string());
        let s = attr_list.collect::<Vec<String>>().join(", ");
        write!(f, "{}", s)?;
        write!(f, "]")
    }
}

impl fmt::Display for Objstructure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let attr_list = ListATTR(self.attribute.keys().cloned().collect());
        write!(
            f,
            "[Objstructure] obj_type: {}, attribute: {}",
            self.obj_type, attr_list
        )?;
        write!(f, "\nAttribute:")?;
        for (name, para) in self.attribute.iter() {
            write!(f, "\n| {}: {}", name, para)?;
        }
        write!(f, ".")
    }
}

#[pyclass]
#[derive(FromPyObject)]
pub struct DataStructOfDoExperiment {
    n: usize,
    obj_id_map: HashMap<String, (ObjType, i32)>,
    data: HashMap<AtomExp, Vec<f64>>,
}
impl DataStructOfDoExperiment {
    fn new(n: usize, obj_id_map: HashMap<String, (ObjType, i32)>) -> Self {
        DataStructOfDoExperiment {
            n,
            obj_id_map,
            data: HashMap::new(),
        }
    }
    pub fn add_data(&mut self, key: (Concept, Vec<String>), data: &Array1<f64>) {
        assert_eq!(data.len(), self.n);
        let mut obj_ids = vec![];
        for obj_name in key.1.iter() {
            let id = self.obj_id_map.get(obj_name).unwrap().1;
            obj_ids.push(id);
        }
        self.data
            .insert(key.0.to_atomexp(obj_ids).unwrap(), data.to_vec());
    }
}

#[pymethods]
impl DataStructOfDoExperiment {
    #[getter]
    pub fn get_n(&self) -> usize {
        self.n
    }
    #[getter]
    pub fn get_obj_id_map(&self) -> HashMap<String, (ObjType, i32)> {
        self.obj_id_map.clone()
    }
    #[getter]
    pub fn get_data(&self) -> HashMap<AtomExp, Vec<f64>> {
        self.data.clone()
    }
    pub fn insert_data(&mut self, key: (Concept, Vec<String>), data: Vec<f64>) {
        assert_eq!(data.len(), self.n);
        let mut obj_ids = vec![];
        for obj_name in key.1.iter() {
            let id = self.obj_id_map.get(obj_name).unwrap().1;
            obj_ids.push(id);
        }
        self.data.insert(key.0.to_atomexp(obj_ids).unwrap(), data);
    }
}

#[pymethods]
impl MeasureType {
    #[getter]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[getter]
    #[inline]
    pub fn repeat_time(&self) -> usize {
        self.repeat_time
    }
    #[getter]
    #[inline]
    pub fn error(&self) -> f64 {
        self.error
    }
    #[getter]
    #[inline]
    pub fn t_end(&self) -> f64 {
        self.t_end
    }
    #[staticmethod]
    pub fn default() -> Self {
        MeasureType {
            n: 100,
            repeat_time: 100,
            error: 1e-8,
            t_end: 2.0,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DataStruct {
    data: HashMap<AtomExp, ExpData>,
}
impl DataStruct {
    pub fn new(data: HashMap<AtomExp, ExpData>) -> Self {
        DataStruct { data }
    }
    #[inline]
    pub fn set_data(&mut self, atom: AtomExp, expdata: ExpData) {
        self.data.insert(atom, expdata);
    }
    #[inline]
    pub fn reset_data(&mut self, atom: AtomExp) {
        self.data.remove(&atom);
    }
    #[inline]
    pub fn get_data(&self) -> &HashMap<AtomExp, ExpData> {
        &self.data
    }
    #[inline]
    pub fn get_data_by_key(&self, atom: &AtomExp) -> Result<ExpData, String> {
        match self.get_data().get(&atom) {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Data {} not found", atom)),
        }
    }
    #[inline]
    pub fn iter(&self) -> std::collections::hash_map::Iter<AtomExp, ExpData> {
        self.data.iter()
    }
    #[inline]
    pub fn has_t(&self) -> bool {
        self.data.contains_key(&AtomExp::get_t())
    }
    #[inline]
    pub fn get_t(&self) -> ExpData {
        self.get_data_by_key(&AtomExp::get_t()).unwrap()
    }
}

impl fmt::Display for DataStruct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[DataStruct] data:").unwrap();
        for key in self.data.keys() {
            write!(f, "{},", key)?;
        }
        write!(f, ".")
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DataStructOfExpData {
    pub name: String,
    pub measuretype: MeasureType,
    pub data: DataStruct,
}
impl DataStructOfExpData {
    fn new(name: String, measuretype: MeasureType, data: DataStruct) -> Self {
        DataStructOfExpData {
            name,
            measuretype,
            data,
        }
    }
    #[inline]
    pub fn set_data(&mut self, atom: AtomExp, expdata: ExpData) {
        self.data.set_data(atom, expdata);
    }
    #[inline]
    pub fn get_data(&self) -> &DataStruct {
        &self.data
    }
    #[inline]
    pub fn get_t(&self) -> ExpData {
        self.data.get_t()
    }
    pub fn plot_expdata(&self, name: &str) {
        // plot the arr
        let mut plot = plotly::Plot::new();
        let n = self.measuretype.n;
        let repeat_time = self.measuretype.repeat_time;
        let t = self.get_t().to_normal_data(n, repeat_time);
        let repeat_time = t.repeat_time;
        for ith in 0..repeat_time {
            let t = t.data.row(ith).to_vec();
            for (key, value) in self.data.iter() {
                if key.get_name() == "t" {
                    continue;
                }
                let x = value.to_normal_data(n, repeat_time).data.row(ith).to_vec();
                let trace = plotly::Scatter::new(t.clone(), x.clone());
                plot.add_trace(trace);
            }
        }
        // plot.show();
        plot.write_html(format!("tmp/{}.html", name));
    }
}

// Store the configuration of an experiment.
// Use `original_data` to get the original measured data in the experiment.
// Such as '[t[0], posx[1]]' in 'motion' experiment.
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct ExpConfig {
    // provided
    pub name: String,
    pub spdim: usize,
    pub exp_para: HashMap<String, Parastructure>,
    pub obj_info: HashMap<String, Objstructure>,
    pub data_info: Vec<(Concept, Vec<String>)>,
    pub geometry_info: Vec<Proposition>,
    // auto generated
    pub obj_id_map: HashMap<String, (ObjType, i32)>,
    pub obj_name_map: HashMap<i32, (ObjType, String)>,
    obj_info_dict: HashMap<ObjType, HashMap<i32, String>>,
    pub data_atomexp: HashSet<AtomExp>,
}
impl ExpConfig {
    pub fn empty() -> Self {
        ExpConfig {
            name: "".to_string(),
            spdim: 0,
            exp_para: HashMap::new(),
            obj_info: HashMap::new(),
            data_info: vec![],
            obj_id_map: HashMap::new(),
            obj_name_map: HashMap::new(),
            obj_info_dict: HashMap::new(),
            data_atomexp: HashSet::new(),
            geometry_info: vec![],
        }
    }
    pub fn new(
        name: String,
        spdim: usize,
        exp_para: HashMap<String, Parastructure>,
        obj_info: HashMap<String, Objstructure>,
        data_info: Vec<(Concept, Vec<String>)>,
    ) -> Self {
        let mut obj_id_map: HashMap<String, (ObjType, i32)> = HashMap::new();
        let mut obj_info_dict: HashMap<ObjType, HashMap<i32, String>> = HashMap::new();
        let mut obj_name_map: HashMap<i32, (ObjType, String)> = HashMap::new();
        for (name, obj) in obj_info.iter() {
            if obj.obj_type == ObjType::clock() {
                obj_id_map.insert(name.clone(), (ObjType::clock(), 0));
                obj_name_map.insert(0, (ObjType::clock(), name.clone()));
            }
        }
        let mut hash_vec: Vec<(&String, &Objstructure)> = obj_info.iter().collect();
        hash_vec.sort_by(|a, b| (a.0).cmp(b.0));
        for (name, obj) in hash_vec.iter() {
            let obj_type = obj.obj_type.clone();
            let obj_id = if obj_type == ObjType::clock() {
                0
            } else {
                let id = obj_id_map.len() as i32;
                obj_id_map.insert((*name).clone(), (obj_type.clone(), id));
                obj_name_map.insert(id, (obj_type.clone(), (*name).clone()));
                id
            };
            if !obj_info_dict.contains_key(&obj.obj_type) {
                obj_info_dict.insert(obj_type, HashMap::new());
            }
            obj_info_dict
                .get_mut(&obj.obj_type)
                .unwrap()
                .insert(obj_id, (*name).clone());
        }
        let mut data_atomexp = HashSet::new();
        for (concept, obj_list) in data_info.iter() {
            let id_list = obj_list
                .iter()
                .map(|x| obj_id_map.get(x).unwrap().1)
                .collect();
            data_atomexp.insert(concept.to_atomexp(id_list).unwrap());
        }
        ExpConfig {
            name,
            spdim,
            exp_para,
            obj_info,
            data_info,
            obj_id_map,
            obj_name_map,
            obj_info_dict,
            data_atomexp,
            geometry_info: vec![],
        }
    }
    #[inline]
    pub fn obj(&self, obj_name: &str) -> &Objstructure {
        self.obj_info.get(obj_name).unwrap()
    }
    pub fn print_obj_info(&self) {
        println!("Name: {}; Object Info:", self.name);
        for (key, obj) in self.obj_info.iter() {
            println!("{}: {}", key, obj);
        }
    }
    #[inline]
    pub fn get_obj(&self, id: i32) -> &Objstructure {
        let name = &self.obj_name_map.get(&id).unwrap().1;
        self.obj_info.get(name).unwrap()
    }
    #[inline]
    fn get_mut_obj(&mut self, id: i32) -> &mut Objstructure {
        let name = &self.obj_name_map.get(&id).unwrap().1;
        self.obj_info.get_mut(name).unwrap()
    }
    #[inline]
    fn set_obj(&mut self, id: i32, obj: Objstructure) {
        let name = self
            .obj_info_dict
            .get(&obj.obj_type)
            .unwrap()
            .get(&id)
            .unwrap();
        self.obj_info.insert(name.clone(), obj);
    }
    #[inline]
    pub fn random_sample(&mut self) {
        for (_, para) in self.exp_para.iter_mut() {
            para.random_sample();
        }
        for (_, obj) in self.obj_info.iter_mut() {
            obj.random_sample();
        }
    }
    #[inline]
    fn random_set_exp_para(&mut self) {
        for (_, para) in self.exp_para.iter_mut() {
            para.random_sample();
        }
    }
    #[inline]
    fn random_set_obj(&mut self, id: i32) {
        self.get_mut_obj(id).random_sample();
    }
    pub fn create_data_struct_of_do_experiment(&self, t_num: usize) -> DataStructOfDoExperiment {
        for (data_concept, obj_names) in self.data_info.iter() {
            let mut concept_temp = data_concept;
            for obj_name in obj_names {
                let obj_type = self.obj_id_map.get(obj_name).unwrap().clone().0;
                match concept_temp {
                    Concept::Mksucc { objtype, concept, id:_ } => {
                        assert_eq!(*objtype, obj_type.to_string());
                        concept_temp = concept;
                    }
                    _ => panic!("DataStructOfDoExperiment: Concept not match, the data info dict has a wrong format."),
                }
            }
        }
        DataStructOfDoExperiment::new(t_num, self.obj_id_map.clone())
    }
    #[inline]
    pub fn original_data(&self) -> Vec<(Concept, Vec<i32>)> {
        let mut original_data = vec![];
        for key in self.data_info.iter() {
            let mut obj_ids = vec![];
            for obj_name in key.1.iter() {
                let id = self.obj_id_map.get(obj_name).unwrap().1;
                obj_ids.push(id);
            }
            original_data.push((key.0.clone(), obj_ids));
        }
        original_data
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ExpStructure {
    exp_config: ExpConfig,
    pub do_experiment: DoExpType,
    datastructofdata: Option<DataStructOfExpData>,
}
impl ExpStructure {
    pub fn empty() -> Self {
        ExpStructure::new(ExpConfig::empty(), DoExpType::empty())
    }
    pub fn new(exp_config: ExpConfig, do_experiment: DoExpType) -> Self {
        ExpStructure {
            exp_config,
            do_experiment,
            datastructofdata: None,
        }
    }
    #[inline]
    pub fn name(&self) -> &str {
        &self.exp_config.name
    }
    #[inline]
    pub fn print_obj_info(&self) {
        self.exp_config.print_obj_info();
    }
    #[inline]
    pub fn has_data(&self, atomexp: &AtomExp) -> bool {
        self.exp_config.data_atomexp.contains(atomexp)
    }
    #[inline]
    pub fn get_obj_ids(&self, obj_type: ObjType) -> HashSet<i32> {
        let ids = self.exp_config.obj_info_dict.get(&obj_type);
        match ids {
            None => return HashSet::new(),
            Some(ids) => {
                return ids.keys().cloned().collect();
            }
        }
    }
    pub fn _set_obj(&mut self, id: i32, obj: Objstructure) {
        if self.datastructofdata.is_some() {
            self.datastructofdata = None;
        }
        self.exp_config.set_obj(id, obj);
    }
    pub fn random_sample(&mut self) {
        if self.datastructofdata.is_some() {
            self.datastructofdata = None;
        }
        self.exp_config.random_sample();
    }
    pub fn random_sample_exp_para(&mut self) {
        if self.datastructofdata.is_some() {
            self.datastructofdata = None;
        }
        self.get_mut_expconfig().random_set_exp_para();
    }
    pub fn random_sample_obj(&mut self, id: i32) {
        if self.datastructofdata.is_some() {
            self.datastructofdata = None;
        }
        self.get_mut_expconfig().random_set_obj(id);
    }
    pub fn calc_expdata(&mut self, measuretype: MeasureType) -> Result<(), String> {
        let t_end = measuretype.t_end();
        let t_num = measuretype.n();
        let repeat_time = measuretype.repeat_time();
        let error = measuretype.error();
        let data_struct = self
            .do_experiment
            .do_experiment(t_end, t_num, 0.0, &self.exp_config)?;
        let data = data_struct.get_data();
        let mut multi_data: HashMap<AtomExp, ExpData> = HashMap::new();
        for (name, data) in data.iter() {
            let mut idata: Array2<f64> = Array2::zeros((repeat_time, t_num));
            for i in 0..repeat_time {
                idata.row_mut(i).assign(&Array1::from_vec(data.clone()));
            }
            idata = add_errors(&idata, error)?;
            assert_eq!(idata.shape(), [repeat_time, t_num]);
            multi_data.insert(name.clone(), ExpData::from_arr2(idata));
        }
        self.datastructofdata = Some(DataStructOfExpData::new(
            self.exp_config.name.clone(),
            measuretype,
            DataStruct::new(multi_data),
        ));
        Ok(())
    }
    pub fn get_expdata(
        &mut self,
        measuretype: MeasureType,
    ) -> Result<&DataStructOfExpData, String> {
        match self.datastructofdata.as_ref() {
            None => self.calc_expdata(measuretype)?,
            Some(datastructofdata) => {
                if datastructofdata.measuretype != measuretype {
                    self.calc_expdata(measuretype)?;
                }
            }
        }
        Ok(self.get_ref_expdata())
    }
    #[inline]
    pub fn expdata_is_none(&self) -> bool {
        self.datastructofdata.is_none()
    }
    #[inline]
    pub fn get_ref_expdata(&self) -> &DataStructOfExpData {
        self.datastructofdata.as_ref().unwrap()
    }
    #[inline]
    pub fn get_mut_expdata(&mut self) -> &mut DataStructOfExpData {
        self.datastructofdata.as_mut().unwrap()
    }
    #[inline]
    pub fn get_ref_expconfig(&self) -> &ExpConfig {
        &self.exp_config
    }
    #[inline]
    fn get_mut_expconfig(&mut self) -> &mut ExpConfig {
        &mut self.exp_config
    }
    pub fn get_all_possible_map(
        &self,
        objtype_id_map: &HashMap<String, HashSet<i32>>,
    ) -> Vec<HashMap<i32, i32>> {
        for (objtype, ids) in objtype_id_map.iter() {
            if self.get_obj_ids(objtype.parse().unwrap()).len() < ids.len() {
                return vec![];
            }
        }
        let mut vec_map: Vec<HashMap<i32, i32>> = vec![];
        vec_map.push(HashMap::new());
        for (objtype, ids) in objtype_id_map.iter() {
            let choose_ids = self.get_obj_ids(objtype.parse().unwrap());
            let perm = choose_ids.iter().permutations(ids.len());
            let mut vec_map_of_objtype = vec![];
            for p in perm {
                let dict: HashMap<i32, i32> =
                    ids.iter().zip(p).map(|(a, b)| (*a, *b as i32)).collect();
                vec_map_of_objtype.push(dict);
            }
            let mut vec_map_new: Vec<HashMap<i32, i32>> = vec![];
            for dict in vec_map.iter() {
                for dict_objtype in vec_map_of_objtype.iter() {
                    let mut new_dict = dict.clone();
                    new_dict.extend(dict_objtype.clone());
                    vec_map_new.push(new_dict);
                }
            }
            vec_map = vec_map_new;
        }
        vec_map
    }
}

pub fn add_errors<D: Dimension>(
    array: &ArrayBase<OwnedRepr<f64>, D>,
    error: f64,
) -> Result<ArrayBase<OwnedRepr<f64>, D>, String> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, error);
    normal
        .map(|normal| array.mapv(|x| normal.sample(&mut rng) + x))
        .map_err(|err| format!("Error in add_errors: {}", err))
}
