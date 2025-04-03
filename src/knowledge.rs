use crate::expdata::{ConstData, ExpData};
use crate::experiments::builtin_experiment;
use crate::experiments::{ExpStructure, Objstructure};
use crate::language::*;
use crate::semantic::exprcharacter::{KeyState, KeyValue, KeyValueHashed};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// A knowledge base about the concepts and theory of a set of experiments.
///
/// It contains a collection of experiments, concepts, objects, and conclusions.
/// It provides methods to manage the knowledge, update, retrieve, and remove knowledge elements,
/// and evaluate new expressions's value based on the current knowledge.
/// It also provides methods to generalize expressions to concepts,
/// and specialize concepts to expressions in specific experiments.
///
/// `concepts`:
///     only support two kinds of Expression:
///     `Intrinsic` and `Concept`.
///
/// `key`:
///     used to calculate the Concept's characteristic value.
///     to classify wheather two Concepts are the same.
///     for example, 1 + x and 2x + 1 - x are the same (under simplification).
///     for example, v[1] - x[2] and v[2] - x[1] are the same (under permutation of index).
///     for example, sum (m[i] + k[i]) and (sum m[i]) + (sum k[i]) are the same (under distribution).
///
/// `conclusions`:
///     used to store the conclusions that hold in some experiments.
///

#[pyclass]
pub struct Knowledge {
    pub experiments: HashMap<String, ExpStructure>,
    pub concepts: HashMap<String, Expression>,
    pub objects: HashMap<String, Objstructure>,
    pub key: KeyState,
    pub conclusions: HashMap<String, Proposition>,
}

#[pymethods]
impl Knowledge {
    /// This function create a empty Knowledge object.
    #[new]
    #[pyo3(signature = ())]
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            concepts: HashMap::new(),
            objects: HashMap::new(),
            key: KeyState::new(None),
            conclusions: HashMap::new(),
        }
    }

    /// This function create a Knowledge object with experiments implemented.
    #[staticmethod]
    #[pyo3(signature = (experiments))]
    pub fn new_with_experiments(experiments: HashMap<String, ExpStructure>) -> Self {
        Self {
            experiments,
            concepts: HashMap::new(),
            objects: HashMap::new(),
            key: KeyState::new(None),
            conclusions: HashMap::new(),
        }
    }

    /// This function create a default Knowledge object with default experiments implemented.
    #[staticmethod]
    #[pyo3(signature = ())]
    pub fn default() -> Self {
        Self {
            experiments: builtin_experiment(),
            concepts: HashMap::new(),
            objects: HashMap::new(),
            key: KeyState::new(None),
            conclusions: HashMap::new(),
        }
    }
    /// This function transform the Knowledge object to its string representation.
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    /// This function generate a Knowledge object from a string representation.
    #[staticmethod]
    pub fn parse_from(s: String) -> PyResult<Self> {
        s.parse().map_err(|x| PyValueError::new_err(x))
    }
    /// Print all experiments' name to the console.
    fn list_experiments(&self) {
        for (name, _) in self.experiments.iter() {
            println!("{}", name);
        }
    }
    /// Print all concepts' name to the console.
    fn list_concepts(&self) {
        // enumerate self.concepts by order of name
        let mut vec: Vec<_> = self.concepts.iter().collect();
        vec.sort_by(|a, b| a.0.cmp(b.0));
        for (name, expression) in vec.iter() {
            println!("{}: {}", name, expression);
        }
    }
    fn list_conclusions(&self) {
        // enumerate self.concepts by order of name
        let mut vec: Vec<_> = self.conclusions.iter().collect();
        vec.sort_by(|a, b| a.0.cmp(b.0));
        for (name, prop) in vec.iter() {
            println!("{}: {}", name, prop);
        }
    }
    #[getter]
    #[inline]
    fn fetch_experiments(&self) -> Vec<String> {
        let mut res = vec![];
        for (name, _) in self.experiments.iter() {
            res.push(name.clone());
        }
        res
    }
    #[getter]
    #[inline]
    fn fetch_concepts(&self) -> HashMap<String, Expression> {
        self.concepts.clone()
    }
    #[getter]
    #[inline]
    fn fetch_intrinsic_concepts(&self) -> HashMap<String, Intrinsic> {
        self.concepts
            .iter()
            .filter_map(|(name, exp)| match exp {
                Expression::Intrinsic { intrinsic } => {
                    Some((name.clone(), intrinsic.as_ref().clone()))
                }
                _ => None,
            })
            .collect()
    }
    #[inline]
    fn fetch_concept_by_name(&self, name: String) -> Expression {
        self.concepts.get(&name).unwrap().clone()
    }
    #[getter]
    #[inline]
    fn fetch_conclusions(&self) -> HashMap<String, Proposition> {
        self.conclusions.clone()
    }
    #[inline]
    fn fetch_conclusion_by_name(&self, name: String) -> Proposition {
        self.conclusions.get(&name).unwrap().clone()
    }
    #[getter]
    #[inline]
    fn fetch_object_keys(&self) -> Vec<String> {
        self.objects.keys().cloned().collect()
    }
    #[inline]
    fn fetch_object_by_name(&self, name: String) -> Objstructure {
        self.objects.get(&name).unwrap().clone()
    }
    #[inline]
    pub fn fetch_object_type_by_name(&self, name: String) -> String {
        self.objects.get(&name).unwrap().obj_type.to_string()
    }

    /// This function is used to register a new (obj: `Objstructure`) to the Knowledge object.
    #[inline]
    #[pyo3(signature = (name, obj))]
    pub fn register_object(&mut self, name: String, obj: Objstructure) {
        self.objects.insert(name, obj);
    }

    /// This function is used to register a new (exp: `ExpStructure`) to the Knowledge object.
    #[inline]
    #[pyo3(signature = (name, exp))]
    fn register_experiment(&mut self, name: String, exp: ExpStructure) {
        assert!(!self.experiments.contains_key(&name));
        self.experiments.insert(name, exp);
    }

    /// This function is used to update an existing (exp: `ExpStructure`) to the Knowledge object.
    #[inline]
    #[pyo3(signature = (name, exp))]
    fn update_experiment(&mut self, name: String, exp: ExpStructure) {
        assert!(self.experiments.contains_key(&name));
        self.experiments.insert(name, exp);
    }

    /// This function is used to register the basic concept to the Knowledge object.
    #[inline]
    #[pyo3(signature = (concept))]
    pub fn register_basic_concept(&mut self, concept: Concept) -> bool {
        let name = concept.get_atomexp_name();
        if self.concepts.contains_key(&name) {
            return false;
        }
        let (kv, kvh, _subs_dict) = self.eval_concept_keyvaluehashed(&concept).unwrap();
        self.key.insert_concept(name.clone(), kv, kvh);
        self.concepts.insert(
            name,
            Expression::Concept {
                concept: Box::new(concept),
            },
        );
        true
    }
    /// This function is used to register a new concept to the Knowledge object.
    /// The concept can be a Intrinsic or a Concept, and they must be wrapped to `Expression` type.
    #[inline]
    #[pyo3(signature = (name, exp))]
    pub fn register_expression(&mut self, name: String, exp: Expression) -> PyResult<bool> {
        self._register_expression(name, exp)
            .map_err(|x| PyValueError::new_err(x))
    }
    #[inline]
    #[pyo3(signature = (name, prop))]
    pub fn register_conclusion(&mut self, name: String, prop: Proposition) -> PyResult<bool> {
        self._register_conclusion(name, prop)
            .map_err(|x| PyValueError::new_err(x))
    }
    #[inline]
    fn remove_conclusion(&mut self, name: String) {
        self.conclusions.remove(&name);
        self.key.obliviate(HashSet::from([name]));
    }
    #[inline]
    fn remove_concept(&mut self, name: String) {
        self.concepts.remove(&name);
        self.key.obliviate(HashSet::from([name]));
    }
    #[inline]
    fn remove_object(&mut self, name: String) {
        self.objects.remove(&name);
    }
    #[inline]
    fn fetch_expstruct(&self, name: String) -> ExpStructure {
        self.experiments.get(&name).unwrap().clone()
    }
    #[inline]
    fn fetch_objstructure_in_expstruct(&self, expname: String, objid: i32) -> Objstructure {
        self.experiments.get(&expname).unwrap().get_obj(objid)
    }
    fn get_expstructure(
        &self,
        expconfig: &IExpConfig,
        objsettings: Vec<Objstructure>,
    ) -> PyResult<ExpStructure> {
        self._get_expstructure(expconfig, objsettings)
            .map_err(|x| PyValueError::new_err(x))
    }
    pub fn eval_intrinsic(
        &self,
        intrinsic: &Intrinsic,
        objsettings: Vec<Objstructure>,
    ) -> Option<ConstData> {
        match self._eval_intrinsic(intrinsic, objsettings) {
            Ok(x) => Some(x),
            Err(_) => None,
        }
    }
    pub fn eval_prop(&self, prop: Proposition, context: &mut ExpStructure) -> PyResult<bool> {
        let res = self._eval_prop(&prop, context);
        res.map_err(|x| PyValueError::new_err(x))
    }
    pub fn eval(&self, exp0: Exp, context: &mut ExpStructure) -> PyResult<ExpData> {
        let res = self._eval(&exp0, context);
        res.map_err(|x| PyValueError::new_err(x))
    }
    pub fn generalize_sexp(&self, sexp: &SExp) -> Option<Concept> {
        match sexp {
            SExp::Mk { expconfig, exp } => self.generalize(exp.as_ref(), expconfig.get_expname()),
        }
    }
    pub fn generalize(&self, expr: &Exp, exp_name: String) -> Option<Concept> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._generalize(expr, context)
    }
    pub fn extract_concepts(&self, expr: &Exp, exp_name: String) -> HashSet<Concept> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._extract_concepts(expr, context)
    }
    fn generalize_to_mksum(&self, expr: &Exp, exp_name: String) -> Option<Concept> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._generalize_to_mksum(expr, context)
    }
    fn generalize_to_normal_concept(&self, expr: &Exp, exp_name: String) -> Option<Concept> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._generalize_to_normal_concept(expr, context)
    }
    #[inline]
    pub fn has_data_in_exp(&self, exp: &Exp, exp_name: String) -> bool {
        let context = self.experiments.get(&exp_name).unwrap();
        self._has_data_in_exp(exp, context)
    }
    pub fn specialize(&self, concept: &Concept, exp_name: String) -> Vec<Exp> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._specialize(concept, context)
    }
    pub fn specialize_concept(&self, concept_name: String, exp_name: String) -> Vec<AtomExp> {
        let context = self.experiments.get(&exp_name).unwrap();
        self._specialize_concept(concept_name, context)
    }

    fn find_similar_concept(&mut self, concept: Concept) -> Option<String> {
        let (_kv, kvh, _subs_dict) = self.eval_concept_keyvaluehashed(&concept).unwrap();
        if kvh.is_none() || kvh.is_const() {
            None
        } else if self.key.contains_key_concept(&kvh) {
            return Some(self.key.get_key_concept(&kvh).unwrap());
        } else {
            None
        }
    }
    fn find_similar_intrinsic(&mut self, intrinsic: Intrinsic) -> Option<String> {
        let (_kv, kvh) = self.eval_intrinsic_keyvaluehashed(&intrinsic).unwrap();
        if kvh.is_none() || kvh.is_const() {
            None
        } else if self.key.contains_key_concept(&kvh) {
            return Some(self.key.get_key_concept(&kvh).unwrap());
        } else {
            None
        }
    }
    fn find_similar_conclusion(&mut self, prop: Proposition) -> Option<String> {
        let (_kv, kvh) = self.eval_proposition_keyvaluehashed(&prop).unwrap();
        if kvh.is_none() || kvh.is_const() {
            None
        } else if self.key.contains_key_concept(&kvh) {
            return Some(self.key.get_key_concept(&kvh).unwrap());
        } else {
            None
        }
    }

    #[inline]
    #[pyo3(signature = (exp, exp_name=None))]
    fn eval_exp_keyvalue(&mut self, exp: &Exp, exp_name: Option<String>) -> KeyValue {
        self.eval_keyvalue(exp, exp_name).unwrap()
    }
    #[inline]
    #[pyo3(signature = (exp, exp_name=None))]
    fn eval_exp_keyvaluehashed(&mut self, exp: &Exp, exp_name: Option<String>) -> KeyValueHashed {
        self.eval_keyvalue(exp, exp_name).unwrap().to_hashed()
    }
    #[inline]
    #[pyo3(signature = (expr))]
    fn eval_expr_key(&mut self, expr: &Expression) -> KeyValueHashed {
        match expr {
            Expression::Exp { exp } => self.eval_keyvalue(exp.as_ref(), None).unwrap().to_hashed(),
            Expression::Proposition { prop } => {
                self.eval_proposition_keyvaluehashed(prop.as_ref())
                    .unwrap()
                    .1
            }
            Expression::Concept { concept } => {
                self.eval_concept_keyvaluehashed(concept.as_ref())
                    .unwrap()
                    .1
            }
            Expression::Intrinsic { intrinsic } => {
                self.eval_intrinsic_keyvaluehashed(intrinsic.as_ref())
                    .unwrap()
                    .1
            }
            _ => unimplemented!(),
        }
    }
    #[pyo3(signature = (concept))]
    fn dependence_of_concept(&self, concept: &Concept) -> HashSet<String> {
        match concept {
            Concept::Mksum {
                objtype: _,
                concept,
            } => self.dependence_of_concept(concept.as_ref()),
            Concept::Mksucc {
                objtype: _,
                concept,
                id: _,
            } => self.dependence_of_concept(concept.as_ref()),
            Concept::Mk0 { exp } => self.dependence_of_exp(exp.as_ref()),
        }
    }
    #[pyo3(signature = (atom))]
    fn dependence_of_atomexp(&self, atom: &AtomExp) -> HashSet<String> {
        if let Some(expr) = self.concepts.get(&atom.get_name()) {
            match expr {
                Expression::Intrinsic { intrinsic } => {
                    let mut res = self.dependence_of_intrinsic(intrinsic.as_ref());
                    res.insert(atom.get_name());
                    res
                }
                Expression::Concept { concept } => {
                    if concept.atomexp_name() == Some(atom.get_name()) {
                        HashSet::from([atom.get_name()])
                    } else {
                        let mut res = self.dependence_of_concept(concept);
                        res.insert(atom.get_name());
                        res
                    }
                }
                _ => unimplemented!(),
            }
        } else {
            panic!("Atom not found in concepts")
        }
    }
    #[pyo3(signature = (exp))]
    fn dependence_of_exp(&self, exp: &Exp) -> HashSet<String> {
        match exp {
            Exp::Atom { atom } => self.dependence_of_atomexp(atom.as_ref()),
            Exp::Number { num: _ } => HashSet::new(),
            Exp::BinaryExp { left, op: _, right } => {
                let mut res = self.dependence_of_exp(left.as_ref());
                res.extend(self.dependence_of_exp(right.as_ref()));
                res
            }
            Exp::UnaryExp { op: _, exp } => self.dependence_of_exp(exp.as_ref()),
            Exp::DiffExp {
                left,
                right,
                ord: _,
            } => {
                let mut res = self.dependence_of_exp(left.as_ref());
                res.extend(self.dependence_of_exp(right.as_ref()));
                res
            }
            Exp::ExpWithMeasureType {
                exp,
                measuretype: _,
            } => self.dependence_of_exp(exp.as_ref()),
            Exp::Partial { left, right } => {
                let mut res = self.dependence_of_exp(left.as_ref());
                res.extend(self.dependence_of_atomexp(right.as_ref()));
                res
            }
        }
    }
    #[pyo3(signature = (intrinsic))]
    fn dependence_of_intrinsic(&self, intrinsic: &Intrinsic) -> HashSet<String> {
        match intrinsic {
            Intrinsic::From { sexp } => {
                let sexp = sexp.as_ref();
                match sexp {
                    SExp::Mk { expconfig, exp } => {
                        let mut res = self.dependence_of_exp(exp.as_ref());
                        res.extend(self.dependence_of_expconfig(expconfig));
                        res
                    }
                }
            }
        }
    }
    #[pyo3(signature = (expconfig))]
    fn dependence_of_expconfig(&self, expconfig: &IExpConfig) -> HashSet<String> {
        match expconfig {
            IExpConfig::From { name: _ } => HashSet::new(),
            IExpConfig::Mk {
                objtype: _,
                expconfig,
                id: _,
            } => self.dependence_of_expconfig(expconfig),
            IExpConfig::Mkfix {
                object,
                expconfig,
                id: _,
            } => {
                let mut res = self.dependence_of_expconfig(expconfig);
                res.insert(object.clone());
                res
            }
        }
    }
    #[pyo3(signature = (prop))]
    fn dependence_of_proposition(&self, prop: &Proposition) -> HashSet<String> {
        match prop {
            Proposition::Conserved { concept } => self.dependence_of_concept(concept.as_ref()),
            Proposition::Zero { concept } => self.dependence_of_concept(concept.as_ref()),
            Proposition::IsConserved { exp } => self.dependence_of_exp(exp.as_ref()),
            Proposition::IsZero { exp } => self.dependence_of_exp(exp.as_ref()),
            Proposition::Eq { left, right } => {
                let mut res = self.dependence_of_exp(left.as_ref());
                res.extend(self.dependence_of_exp(right.as_ref()));
                res
            }
            Proposition::Not { prop } => self.dependence_of_proposition(prop.as_ref()),
        }
    }

    #[pyo3(signature = (expression, exp_name=None))]
    fn raw_definition(&self, expression: &Expression, exp_name: Option<String>) -> Expression {
        match expression {
            Expression::Exp { exp } => Expression::Exp {
                exp: Box::new(self.raw_definition_exp(exp, exp_name)),
            },
            Expression::Proposition { prop } => Expression::Proposition {
                prop: Box::new(self.raw_definition_prop(prop, exp_name)),
            },
            Expression::Concept { concept } => Expression::Concept {
                concept: Box::new(self.raw_definition_concept(concept, exp_name)),
            },
            Expression::Intrinsic { intrinsic } => Expression::Intrinsic {
                intrinsic: Box::new(self.raw_definition_intrinsic(intrinsic)),
            },
            _ => unimplemented!(),
        }
    }
    #[pyo3(signature = (prop, exp_name=None))]
    fn raw_definition_prop(&self, prop: &Proposition, exp_name: Option<String>) -> Proposition {
        let context = exp_name.map(|x| self.experiments.get(&x).unwrap());
        self._raw_definition_prop(prop, context)
    }
    #[pyo3(signature = (exp, exp_name=None))]
    fn raw_definition_exp(&self, exp: &Exp, exp_name: Option<String>) -> Exp {
        let context = exp_name.map(|x| self.experiments.get(&x).unwrap());
        self._raw_definition_exp(exp, context)
    }
    #[pyo3(signature = (concept, exp_name=None))]
    fn raw_definition_concept(&self, concept: &Concept, exp_name: Option<String>) -> Concept {
        let context = exp_name.map(|x| self.experiments.get(&x).unwrap());
        self._raw_definition_concept(concept, context)
    }
    #[pyo3(signature = (intrinsic))]
    fn raw_definition_intrinsic(&self, intrinsic: &Intrinsic) -> Intrinsic {
        self._raw_definition_intrinsic(intrinsic)
    }

    pub fn parse_atomexp_to_sympy_str(&self, input: &AtomExp, argument: String) -> String {
        self._parse_atomexp_to_sympy_str(input, argument)
    }

    pub fn parse_exp_to_sympy_str(&self, input: &Exp, argument: String) -> String {
        self._parse_exp_to_sympy_str(input, argument)
    }

    fn obliviate(&mut self, useful_name: HashSet<String>) {
        let concept_names: Vec<String> = self.concepts.keys().map(|x| x.clone()).collect();
        let object_names: Vec<String> = self.objects.keys().map(|x| x.clone()).collect();
        let mut obliviate_names = HashSet::new();
        for name in concept_names.iter() {
            if !useful_name.contains(name) {
                self.concepts.remove(name);
                obliviate_names.insert(name.clone());
            }
        }
        for name in object_names.iter() {
            if !useful_name.contains(name) {
                self.objects.remove(name);
                obliviate_names.insert(name.clone());
            }
        }
        self.key.obliviate(obliviate_names);
    }

    fn gen_atom_concept_by_name(&self, name: String) -> PyResult<Concept> {
        let res = self.gen_atom_concept(name);
        res.map_err(|x| PyValueError::new_err(x))
    }
    fn check_geometry_info(&self, context: &mut ExpStructure) -> PyResult<bool> {
        let res = self._check_geometry_info(context);
        res.map_err(|x| PyValueError::new_err(x))
    }
}

impl Knowledge {
    fn gen_atom_concept(&self, name: String) -> Result<Concept, String> {
        match self.concepts.get(&name).unwrap() {
            Expression::Concept { concept } => {
                let preobjs = concept.get_pre_objtype_id_vec();
                let ids: Vec<i32> = (1..(preobjs.len() + 1) as i32).collect();
                let atomexp = AtomExp::new_variable_ids(name, ids);
                let mut concept_new = Concept::Mk0 {
                    exp: Box::new(Exp::from_atom(&atomexp)),
                };
                for i in 1..(preobjs.len() + 1) {
                    concept_new = Concept::Mksucc {
                        objtype: preobjs[i - 1].0.clone(),
                        concept: Box::new(concept_new),
                        id: i as i32,
                    };
                }
                Ok(concept_new)
            }
            Expression::Intrinsic { intrinsic } => {
                let preobjs = intrinsic.get_input_objtypes();
                let ids: Vec<i32> = (1..(preobjs.len() + 1) as i32).collect();
                let atomexp = AtomExp::new_variable_ids(name, ids);
                let mut concept_new = Concept::Mk0 {
                    exp: Box::new(Exp::from_atom(&atomexp)),
                };
                for i in 1..(preobjs.len() + 1) {
                    concept_new = Concept::Mksucc {
                        objtype: preobjs[i - 1].obj.clone(),
                        concept: Box::new(concept_new),
                        id: i as i32,
                    };
                }
                Ok(concept_new)
            }
            _ => Err(format!("Cannot generate a atom_concept for {}", name)),
        }
    }
}

impl Knowledge {
    pub fn _register_conclusion(
        &mut self,
        name: String,
        prop: Proposition,
    ) -> Result<bool, String> {
        let (kv, kvh) = self.eval_proposition_keyvaluehashed(&prop)?;
        // 不注册形如 1, m * k, m * (1 / m + 1) 这样显然守恒的结论
        if kvh.is_none() || kvh.is_const() {
            return Ok(false);
        }
        // 不重复注册结论
        if self.key.contains_key_concept(&kvh) || self.key.contains_key_concept(&kvh.neg()) {
            return Ok(false);
        }
        self.key.insert_concept(name.clone(), kv, kvh);
        self.conclusions.insert(name, prop);
        Ok(true)
    }
    pub fn _register_expression(&mut self, name: String, exp: Expression) -> Result<bool, String> {
        match &exp {
            Expression::Concept { concept } => {
                let (kv, kvh, _subs_dict) = self.eval_concept_keyvaluehashed(&concept)?;
                // 不注册形如 1, m * k, m * (1 / m + 1), v - v, x / x 这样的常量概念
                if kvh.is_none() || kvh.is_const() {
                    return Ok(false);
                }
                // 不重复注册概念
                if self.key.contains_key_concept(&kvh) || self.key.contains_key_concept(&kvh.neg())
                {
                    return Ok(false);
                }
                self.key.insert_concept(name.clone(), kv, kvh);
            }
            Expression::Intrinsic { intrinsic } => {
                let (kv, kvh) = self.eval_intrinsic_keyvaluehashed(&intrinsic)?;
                // 不注册形如 1, m * k, m * (1 / m + 1) 这样的内秉概念
                if kvh.is_none() || kvh.is_const() {
                    return Ok(false);
                }
                // 不重复注册概念
                if self.key.contains_key_concept(&kvh) || self.key.contains_key_concept(&kvh.neg())
                {
                    return Ok(false);
                }
                self.key.insert_concept(name.clone(), kv, kvh);
            }
            _ => (),
        };
        self.concepts.insert(name, exp);
        Ok(true)
    }
}
