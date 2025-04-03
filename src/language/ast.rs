use super::ast_arith::{Mul, Pow};
use super::complexity::Complexity;
/// This file defines the Abstract Syntax Tree (AST) for the language.
/// The AST is used to represent the physical concepts, laws, and intrinsic quantities.
///
/// `Concept`, `Exp`, `Proposition`, `SExp`,
/// `AtomExp`, `Intrinsic`, `IExpConfig` are the main components of the AST.
use crate::experiments::objects::obj::ObjType;
use crate::parser::FromStr;
use crate::r;
use core::panic;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Proposition defines the physical laws that holds in a specific Experiment or holds in general.
///
/// General Laws:
///     Conserved {concept: Concept}
///     Zero {concept: Concept}
///
/// Specific Laws:
///     IsConserved {exp: Exp}
///     IsZero {exp: Exp}
///     Eq {left: Exp, right: Exp}
#[pyclass(eq)]
#[derive(Clone, PartialEq, Hash, Eq)]
pub enum Proposition {
    // General Conclusion that can be applied to any Experiment
    Conserved { concept: Box<Concept> },
    Zero { concept: Box<Concept> },
    // Specific Conclusion in a specific Experiment
    IsConserved { exp: Box<Exp> },
    IsZero { exp: Box<Exp> },
    Eq { left: Box<Exp>, right: Box<Exp> },
    Not { prop: Box<Proposition> },
}
#[pymethods]
impl Proposition {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn __neg__(&self) -> Self {
        match self {
            Proposition::Conserved { concept } => Proposition::Conserved {
                concept: Box::new(-concept.as_ref()),
            },
            Proposition::Zero { concept } => Proposition::Zero {
                concept: Box::new(-concept.as_ref()),
            },
            Proposition::IsConserved { exp } => Proposition::IsConserved {
                exp: Box::new(-exp.as_ref()),
            },
            Proposition::IsZero { exp } => Proposition::IsZero {
                exp: Box::new(-exp.as_ref()),
            },
            _ => panic!("Error: Negation of Eq or Not is not supported"),
        }
    }
    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }
    #[new]
    #[inline]
    pub fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
    #[getter]
    #[inline]
    pub fn prop_type(&self) -> String {
        match self {
            Proposition::Conserved { concept: _ } => r!("Conserved"),
            Proposition::Zero { concept: _ } => r!("Zero"),
            Proposition::IsConserved { exp: _ } => r!("IsConserved"),
            Proposition::IsZero { exp: _ } => r!("IsZero"),
            Proposition::Eq { left: _, right: _ } => r!("Eq"),
            Proposition::Not { prop: _ } => r!("Not"),
        }
    }
    #[getter]
    #[inline]
    fn is_zero(&self) -> bool {
        match self {
            Proposition::Zero { concept: _ } | Proposition::IsZero { exp: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    fn is_conserved(&self) -> bool {
        match self {
            Proposition::Conserved { concept: _ } | Proposition::IsConserved { exp: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    fn get_complexity(&self) -> i32 {
        self.complexity()
    }
    #[getter]
    #[inline]
    fn unwrap_exp(&self) -> Exp {
        match self {
            Proposition::IsConserved { exp } => *exp.clone(),
            Proposition::IsZero { exp } => *exp.clone(),
            _ => panic!("Error: unwrap_exp failed"),
        }
    }
    #[getter]
    #[inline]
    fn unwrap_concept(&self) -> Concept {
        match self {
            Proposition::Conserved { concept } => *concept.clone(),
            Proposition::Zero { concept } => *concept.clone(),
            _ => panic!("Error: unwrap_concept failed"),
        }
    }
}

/// BinaryOp defines the binary operations in the language.
#[pyclass(eq, eq_int)]
#[derive(Eq, PartialEq, Clone, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}
impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "**"),
        }
    }
}

/// UnaryOp defines the unary operations in the language.
#[pyclass(eq, eq_int)]
#[derive(Eq, PartialEq, Clone, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Diff,
}
impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Diff => write!(f, "D"),
        }
    }
}

/// MeasureType defines the hyper parameters for the measurement in certain Experiment.
/// `t_end`: duration of the measurement
/// `n`: number of time points
/// `repeat_time`: number of independent repeated measurements
/// `error`: the overall noise level of the measurement
#[pyclass(eq)]
#[derive(Clone, PartialEq, Serialize, Debug)]
pub struct MeasureType {
    pub t_end: f64,
    pub n: usize,
    pub repeat_time: usize,
    pub error: f64,
}
impl MeasureType {
    pub fn new(t_end: f64, n: i32, repeat_time: i32, error: f64) -> Self {
        Self {
            t_end,
            n: n as usize,
            repeat_time: repeat_time as usize,
            error,
        }
    }
}

impl Hash for MeasureType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.t_end.to_bits().hash(state);
        self.n.hash(state);
        self.repeat_time.hash(state);
        self.error.to_bits().hash(state);
    }
}

impl Eq for MeasureType {}

/// AtomExp defines the atomic components in an physical expression.
///
/// Variable {name} -> a variable with a name,
/// such as `t`, `G`, `g`, ...
///
/// VariableIds {name, ids} -> a variable with a name and a list of indices,
/// such as `m[1]`, `x[2]`, `dist[3,4]`, ...
#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AtomExp {
    Variable { name: String },
    VariableIds { name: String, ids: Vec<i32> },
}

/// Exp defines the expression where each variable has specific indices.
/// Hence it's possible to evaluate an Exp in a specific Experiment.
#[pyclass(eq)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum Exp {
    Number {
        num: i32,
    },
    Atom {
        atom: Box<AtomExp>,
    },
    UnaryExp {
        op: UnaryOp,
        exp: Box<Exp>,
    },
    BinaryExp {
        left: Box<Exp>,
        op: BinaryOp,
        right: Box<Exp>,
    },
    DiffExp {
        left: Box<Exp>,
        right: Box<Exp>,
        ord: i32,
    },
    ExpWithMeasureType {
        exp: Box<Exp>,
        measuretype: Box<MeasureType>,
    },
    Partial {
        left: Box<Exp>,
        right: Box<AtomExp>,
    },
}
#[pymethods]
impl Exp {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    #[inline]
    #[getter]
    pub fn get_type(&self) -> String {
        match self {
            Exp::Number { num: _ } => r!("Number"),
            Exp::Atom { atom: _ } => r!("Atom"),
            Exp::UnaryExp { op, exp: _ } => match op {
                UnaryOp::Neg => r!("Neg"),
                UnaryOp::Diff => r!("Diff"),
            },
            Exp::BinaryExp {
                left: _,
                op,
                right: _,
            } => match op {
                BinaryOp::Add => r!("Add"),
                BinaryOp::Sub => r!("Sub"),
                BinaryOp::Mul => r!("Mul"),
                BinaryOp::Div => r!("Div"),
                BinaryOp::Pow => r!("Pow"),
            },
            Exp::DiffExp {
                left: _,
                right: _,
                ord: _,
            } => r!("DiffExp"),
            Exp::ExpWithMeasureType {
                exp: _,
                measuretype: _,
            } => r!("ExpWithMeasureType"),
            Exp::Partial { left: _, right: _ } => r!("Partial"),
        }
    }
    #[inline]
    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }
    #[getter]
    #[inline]
    fn unwrap_atom(&self) -> AtomExp {
        match self {
            Exp::Atom { atom } => *atom.clone(),
            _ => panic!("Error: unwrap_atom failed"),
        }
    }
    #[getter]
    #[inline]
    fn unwrap_partial(&self) -> (Exp, AtomExp) {
        match self {
            Exp::Partial { left, right } => (*left.clone(), *right.clone()),
            _ => panic!("Error: unwrap_partial failed"),
        }
    }
    #[getter]
    #[inline]
    fn get_complexity(&self) -> i32 {
        self.complexity()
    }
    #[staticmethod]
    pub fn from_atom(atom: &AtomExp) -> Self {
        Exp::Atom {
            atom: Box::new(atom.clone()),
        }
    }
    #[staticmethod]
    pub fn new_variable(name: String) -> Self {
        Exp::Atom {
            atom: Box::new(AtomExp::new_variable(name)),
        }
    }
    #[staticmethod]
    pub fn get_t() -> Self {
        Exp::Atom {
            atom: Box::new(AtomExp::get_t()),
        }
    }
    #[staticmethod]
    pub fn new_variable_ids(name: String, ids: Vec<i32>) -> Self {
        if ids.len() == 0 {
            Exp::Atom {
                atom: Box::new(AtomExp::Variable { name }),
            }
        } else {
            Exp::Atom {
                atom: Box::new(AtomExp::VariableIds { name, ids }),
            }
        }
    }
    #[getter]
    pub fn get_debug_str(&self) -> String {
        format!("{:?}", self)
    }
    #[getter]
    #[inline]
    pub fn get_all_atoms(&self) -> HashSet<AtomExp> {
        match self {
            Exp::Number { num: _ } => HashSet::new(),
            Exp::Atom { atom } => HashSet::from([*atom.clone()]),
            Exp::UnaryExp { op: _, exp } => exp.get_all_atoms(),
            Exp::BinaryExp { left, op: _, right } => {
                let left = left.get_all_atoms();
                let right = right.get_all_atoms();
                left.union(&right).cloned().collect()
            }
            Exp::DiffExp {
                left,
                right,
                ord: _,
            } => {
                let left = left.get_all_atoms();
                let right = right.get_all_atoms();
                left.union(&right).cloned().collect()
            }
            Exp::ExpWithMeasureType {
                exp,
                measuretype: _,
            } => exp.get_all_atoms(),
            Exp::Partial { left, right } => {
                let mut res = left.get_all_atoms();
                res.insert(*right.clone());
                res
            }
        }
    }
    #[getter]
    #[inline]
    pub fn get_all_partials(&self) -> HashSet<Exp> {
        match self {
            Exp::Number { num: _ } => HashSet::new(),
            Exp::Atom { atom: _ } => HashSet::new(),
            Exp::UnaryExp { op: _, exp } => exp.get_all_partials(),
            Exp::BinaryExp { left, op: _, right } => {
                let left = left.get_all_partials();
                let right = right.get_all_partials();
                left.union(&right).cloned().collect()
            }
            Exp::DiffExp {
                left,
                right,
                ord: _,
            } => {
                let left = left.get_all_partials();
                let right = right.get_all_partials();
                left.union(&right).cloned().collect()
            }
            Exp::ExpWithMeasureType {
                exp,
                measuretype: _,
            } => exp.get_all_partials(),
            Exp::Partial { left: _, right: _ } => HashSet::from([self.clone()]),
        }
    }
    #[getter]
    pub fn as_ordered_terms(&self) -> Vec<Exp> {
        match self {
            Exp::BinaryExp { left, op, right } => {
                let mut res = vec![];
                if op == &BinaryOp::Add {
                    res.extend(left.as_ordered_terms());
                    res.extend(right.as_ordered_terms());
                } else if op == &BinaryOp::Sub {
                    res.extend(left.as_ordered_terms());
                    res.extend(right.as_ordered_terms().iter().map(|x| -x.clone()));
                } else {
                    res.push(self.clone());
                }
                res
            }
            _ => vec![self.clone()],
        }
    }
    #[getter]
    pub fn remove_coeff(&self) -> Exp {
        match self {
            Exp::BinaryExp { left, op, right } => match op {
                BinaryOp::Mul => match left.unwrap_number() {
                    Some(_) => match right.unwrap_number() {
                        Some(_) => Exp::from_i32(1),
                        None => *right.clone(),
                    },
                    None => match right.unwrap_number() {
                        Some(_) => *left.clone(),
                        None => self.clone(),
                    },
                },
                BinaryOp::Div => match right.unwrap_number() {
                    Some(_) => *left.clone(),
                    None => match left.unwrap_number() {
                        Some(_) => Exp::from_i32(1) / *right.clone(),
                        None => self.clone(),
                    },
                },
                _ => self.clone(),
            },
            Exp::UnaryExp { op, exp } => match op {
                UnaryOp::Neg => exp.remove_coeff(),
                _ => self.clone(),
            },
            _ => self.clone(),
        }
    }
    pub fn subst(&self, oid: i32, nid: i32) -> Self {
        match self {
            Exp::Number { num } => Exp::Number { num: *num },
            Exp::Atom { atom } => Exp::Atom {
                atom: Box::new(atom.subst(oid, nid)),
            },
            Exp::UnaryExp { op, exp } => Exp::UnaryExp {
                op: op.clone(),
                exp: Box::new(exp.subst(oid, nid)),
            },
            Exp::BinaryExp { left, op, right } => Exp::BinaryExp {
                left: Box::new(left.subst(oid, nid)),
                op: op.clone(),
                right: Box::new(right.subst(oid, nid)),
            },
            Exp::DiffExp { left, right, ord } => Exp::DiffExp {
                left: Box::new(left.subst(oid, nid)),
                right: Box::new(right.subst(oid, nid)),
                ord: *ord,
            },
            Exp::ExpWithMeasureType { exp, measuretype } => Exp::ExpWithMeasureType {
                exp: Box::new(exp.subst(oid, nid)),
                measuretype: measuretype.clone(),
            },
            Exp::Partial { left, right } => Exp::Partial {
                left: Box::new(left.subst(oid, nid)),
                right: Box::new(right.subst(oid, nid)),
            },
        }
    }
    pub fn subst_by_dict(&self, sub_dict: HashMap<i32, i32>) -> Self {
        self.substs(&sub_dict)
    }
    #[getter]
    #[inline]
    pub fn obj_number(&self) -> usize {
        self.get_allids_not_t().len()
    }
    #[getter]
    #[inline]
    pub fn get_allids_not_t(&self) -> HashSet<i32> {
        let mut s = self.get_allids();
        s.remove(&0);
        s
    }
    #[getter]
    #[inline]
    pub fn get_allids(&self) -> HashSet<i32> {
        match self {
            Exp::Number { num: _ } => HashSet::new(),
            Exp::Atom { atom } => atom.get_allids(),
            Exp::UnaryExp { op: _, exp } => exp.get_allids(),
            Exp::BinaryExp { left, op: _, right } => {
                let left = left.get_allids();
                let right = right.get_allids();
                let res: HashSet<i32> = left.union(&right).cloned().collect();
                res
            }
            Exp::DiffExp {
                left,
                right,
                ord: _,
            } => {
                let left = left.get_allids();
                let right = right.get_allids();
                let res: HashSet<i32> = left.union(&right).cloned().collect();
                res
            }
            Exp::ExpWithMeasureType {
                exp,
                measuretype: _,
            } => exp.get_allids(),
            Exp::Partial { left, right } => {
                let mut res = left.get_allids();
                res.extend(right.get_allids());
                res
            }
        }
    }
    #[new]
    pub fn from_string(str: String) -> PyResult<Self> {
        (&str).parse().map_err(|e| PyValueError::new_err(e))
    }
    #[inline]
    fn __add__(&self, other: &Self) -> Self {
        self + other
    }
    #[inline]
    fn __sub__(&self, other: &Self) -> Self {
        self - other
    }
    #[inline]
    fn __mul__(&self, other: &Self) -> Self {
        self * other
    }
    #[inline]
    fn __rmul__(&self, other: i32) -> Self {
        Mul::mul(self.clone(), other)
    }
    #[inline]
    fn __truediv__(&self, other: &Self) -> Self {
        self / other
    }
    #[inline]
    fn __powi__(&self, i: i32) -> Self {
        self.clone().pow(Exp::from_i32(i))
    }
    #[inline]
    fn __neg__(&self) -> Self {
        (Exp::UnaryExp {
            op: UnaryOp::Neg,
            exp: Box::new(self.clone()),
        })
        .doit()
    }
    #[inline]
    pub fn __difft__(&self, ord: i32) -> Self {
        (Exp::DiffExp {
            left: Box::new(self.clone()),
            right: Box::new(Exp::get_t()),
            ord,
        })
        .doit()
    }
    #[inline]
    pub fn __diff__(&self, other: &Self) -> Self {
        (Exp::DiffExp {
            left: Box::new(self.clone()),
            right: Box::new(other.clone()),
            ord: 1,
        })
        .doit()
    }
    #[inline]
    pub fn __partial__(&self, other: &AtomExp) -> Self {
        (Exp::Partial {
            left: Box::new(self.clone()),
            right: Box::new(other.clone()),
        })
        .doit()
    }
    #[inline]
    fn replace_atom_by_exp(&self, atom: &AtomExp, exp: &Exp) -> Self {
        match self {
            Exp::Number { num } => Exp::Number { num: *num },
            Exp::Atom { atom: a } => {
                if **a == *atom {
                    exp.clone()
                } else {
                    self.clone()
                }
            }
            Exp::UnaryExp { op, exp: e } => Exp::UnaryExp {
                op: op.clone(),
                exp: Box::new(e.replace_atom_by_exp(atom, exp)),
            },
            Exp::BinaryExp {
                left: l,
                op,
                right: r,
            } => Exp::BinaryExp {
                left: Box::new(l.replace_atom_by_exp(atom, exp)),
                op: op.clone(),
                right: Box::new(r.replace_atom_by_exp(atom, exp)),
            },
            Exp::DiffExp {
                left: l,
                right: r,
                ord,
            } => Exp::DiffExp {
                left: Box::new(l.replace_atom_by_exp(atom, exp)),
                right: Box::new(r.replace_atom_by_exp(atom, exp)),
                ord: *ord,
            },
            Exp::ExpWithMeasureType {
                exp: e,
                measuretype: m,
            } => Exp::ExpWithMeasureType {
                exp: Box::new(e.replace_atom_by_exp(atom, exp)),
                measuretype: m.clone(),
            },
            Exp::Partial { left, right } => {
                if **right == *atom {
                    panic!("Error: replace_atom_by_exp cannot apply to the Partial Atom")
                }
                Exp::Partial {
                    left: Box::new(left.replace_atom_by_exp(atom, exp)),
                    right: right.clone(),
                }
            }
        }
    }
    #[inline]
    fn replace_partial_by_exp(&self, partial_exp: &Exp, exp: &Exp) -> Self {
        let origin = format!("{}", self);
        let replaced = origin.replace(&format!("{}", partial_exp), &format!("{}", exp));
        (&replaced).parse().unwrap()
    }
    pub fn doit(&self) -> Exp {
        // A trivial simplify function.
        match self {
            Exp::UnaryExp { op, exp } => match op {
                UnaryOp::Neg => -exp.doit(),
                UnaryOp::Diff => {
                    let exp = exp.doit();
                    match exp {
                        Exp::DiffExp { left, right, ord } => Exp::DiffExp {
                            left: right,
                            right: left,
                            ord,
                        },
                        _ => panic!("Error: Doit failed"),
                    }
                }
            },
            Exp::BinaryExp { left, op, right } => {
                let left = left.doit();
                let right = right.doit();
                match op {
                    BinaryOp::Add => left + right,
                    BinaryOp::Sub => left - right,
                    BinaryOp::Mul => left * right,
                    BinaryOp::Div => left / right,
                    BinaryOp::Pow => left.pow(right),
                }
            }
            Exp::DiffExp {
                left: l1,
                right: r1,
                ord,
            } => {
                let left = l1.doit();
                let right = r1.doit();
                match left {
                    Exp::Number { num: _ } => Exp::Number { num: 0 },
                    Exp::DiffExp {
                        left: l2,
                        right: r2,
                        ord: o2,
                    } => {
                        if &*r2 == &right {
                            Exp::DiffExp {
                                left: l2,
                                right: r2,
                                ord: o2 + ord,
                            }
                        } else {
                            Exp::DiffExp {
                                left: Box::new(Exp::DiffExp {
                                    left: l2,
                                    right: r2,
                                    ord: o2,
                                }),
                                right: Box::new(right),
                                ord: *ord,
                            }
                        }
                    }
                    Exp::UnaryExp {
                        op: UnaryOp::Neg,
                        exp,
                    } => Exp::UnaryExp {
                        op: UnaryOp::Neg,
                        exp: Box::new(Exp::DiffExp {
                            left: exp,
                            right: Box::new(right),
                            ord: *ord,
                        }),
                    },
                    _ => Exp::DiffExp {
                        left: Box::new(left),
                        right: Box::new(right),
                        ord: *ord,
                    },
                }
            }
            Exp::ExpWithMeasureType { exp, measuretype } => Exp::ExpWithMeasureType {
                exp: Box::new(exp.doit()),
                measuretype: measuretype.clone(),
            },
            Exp::Partial { left, right } => {
                let left = left.doit();
                match &left {
                    Exp::Number { num: _ } => Exp::Number { num: 0 },
                    Exp::UnaryExp { op, exp } => match op {
                        UnaryOp::Neg => -Exp::Partial {
                            left: exp.clone(),
                            right: right.clone(),
                        },
                        _ => Exp::Partial {
                            left: Box::new(left),
                            right: right.clone(),
                        },
                    },
                    _ => Exp::Partial {
                        left: Box::new(left),
                        right: right.clone(),
                    },
                }
            }
            _ => self.clone(),
        }
    }
}

impl Ord for Exp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let sc = self.complexity();
        let oc = other.complexity();
        let ord = sc < oc || sc == oc && format!("{}", self) < format!("{}", other);
        if ord {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }
}

impl PartialOrd for Exp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Exp {
    pub fn from_i32(i: i32) -> Self {
        Exp::Number { num: i }
    }
    pub fn unwrap_number(&self) -> Option<i32> {
        match self {
            Exp::Number { num } => Some(*num),
            _ => None,
        }
    }
    pub fn substs(&self, sub_dict: &HashMap<i32, i32>) -> Self {
        match self {
            Exp::Number { num } => Exp::Number { num: *num },
            Exp::Atom { atom } => Exp::Atom {
                atom: Box::new(atom.substs(sub_dict.clone())),
            },
            Exp::UnaryExp { op, exp } => Exp::UnaryExp {
                op: op.clone(),
                exp: Box::new(exp.substs(sub_dict)),
            },
            Exp::BinaryExp { left, op, right } => Exp::BinaryExp {
                left: Box::new(left.substs(sub_dict)),
                op: op.clone(),
                right: Box::new(right.substs(sub_dict)),
            },
            Exp::DiffExp { left, right, ord } => Exp::DiffExp {
                left: Box::new(left.substs(sub_dict)),
                right: Box::new(right.substs(sub_dict)),
                ord: *ord,
            },
            Exp::ExpWithMeasureType { exp, measuretype } => Exp::ExpWithMeasureType {
                exp: Box::new(exp.substs(sub_dict)),
                measuretype: measuretype.clone(),
            },
            Exp::Partial { left, right } => Exp::Partial {
                left: Box::new(left.substs(sub_dict)),
                right: Box::new(right.substs(sub_dict.clone())),
            },
        }
    }
}

/// SExp is useful to define an Intrinsic object,
/// It contains a specific experiment configuration and an Exp object.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum SExp {
    Mk {
        expconfig: Box<IExpConfig>,
        exp: Box<Exp>,
    },
}
#[pymethods]
impl SExp {
    #[inline]
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn __neg__(&self) -> Self {
        match self {
            SExp::Mk { expconfig, exp } => SExp::Mk {
                expconfig: expconfig.clone(),
                exp: Box::new(-*exp.clone()),
            },
        }
    }
    fn __inv__(&self) -> Self {
        match self {
            SExp::Mk { expconfig, exp } => SExp::Mk {
                expconfig: expconfig.clone(),
                exp: Box::new(exp.clone().__powi__(-1)),
            },
        }
    }
    #[new]
    #[inline]
    fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
    #[getter]
    #[inline]
    pub fn get_expconfig(&self) -> IExpConfig {
        match self {
            SExp::Mk { expconfig, .. } => (**expconfig).clone(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_exp(&self) -> Exp {
        match self {
            SExp::Mk { exp, .. } => (**exp).clone(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_objtype_id_map(&self) -> HashMap<String, HashSet<i32>> {
        self.get_expconfig().get_objtype_id_map()
    }
    #[getter]
    #[inline]
    fn get_relevant_objs(&self) -> HashSet<String> {
        self.get_expconfig().get_relevant_objs()
    }
}

/// Concept defines the expression where each variable has dummy indices that are not determined.
/// We can apply a concept in a specific experiment by mapping the dummy indices to specific values in the experiment.
/// This process is called `specialization`.
/// A Concept object can be specialized to an `Exp` object.
///
/// For example:
/// `(1 -> Particle) |- m[1]` can be specialized to `m[1], m[2]` in collision experiment.
#[pyclass(eq)]
#[derive(Clone, PartialEq, Hash, Eq)]
pub enum Concept {
    // `|- {exp}`
    Mk0 {
        exp: Box<Exp>,
    },
    // `({id} -> {objtype}) {concept}`
    Mksucc {
        objtype: String,
        concept: Box<Concept>,
        id: i32,
    },
    // `[Sum: {objtype}] {concept}`
    Mksum {
        objtype: String,
        concept: Box<Concept>,
    },
}
impl Concept {
    #[inline]
    pub fn partial_subst(&self, idlist: Vec<i32>) -> Result<Concept, String> {
        self._partial_subst(idlist, HashSet::new(), HashMap::new())
    }
    fn _partial_subst(
        &self,
        idlist: Vec<i32>,
        summed_objtype: HashSet<String>,
        sub_dict: HashMap<i32, i32>,
    ) -> Result<Concept, String> {
        match self {
            Concept::Mk0 { exp } => {
                if idlist.len() > 0 {
                    return Err(r!(
                        "Error: idlist is to long for Concept::partial_subst function"
                    ));
                }
                Ok(Concept::Mk0 {
                    exp: Box::new(exp.substs(&sub_dict)),
                })
            }
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => {
                if summed_objtype.contains(objtype) {
                    let x = concept._partial_subst(idlist, summed_objtype, sub_dict)?;
                    Ok(Concept::Mksucc {
                        objtype: objtype.clone(),
                        concept: Box::new(x),
                        id: *id,
                    })
                } else {
                    let mut jdlist = idlist;
                    if jdlist.len() == 0 {
                        Ok(Concept::Mksucc {
                            objtype: objtype.clone(),
                            concept: Box::new(concept._partial_subst(
                                jdlist,
                                summed_objtype,
                                sub_dict,
                            )?),
                            id: *id,
                        })
                    } else {
                        let jd = jdlist.pop().unwrap();
                        let mut sub_dict = sub_dict;
                        sub_dict.insert(*id, jd);
                        concept._partial_subst(jdlist, summed_objtype, sub_dict)
                    }
                }
            }
            Concept::Mksum { objtype, concept } => {
                let mut summed_objtype = summed_objtype;
                summed_objtype.insert(objtype.clone());
                Ok(Concept::Mksum {
                    objtype: objtype.clone(),
                    concept: Box::new(concept._partial_subst(idlist, summed_objtype, sub_dict)?),
                })
            }
        }
    }

    /// 非 Mksum 形式的 Concept，通过给哑指标赋值，得到一个具体的 Exp 对象
    #[inline]
    pub fn subst(&self, idlist: Vec<i32>) -> Result<Exp, String> {
        self._subst(idlist, HashMap::new())
    }
    fn _subst(&self, idlist: Vec<i32>, sub_dict: HashMap<i32, i32>) -> Result<Exp, String> {
        match self {
            Concept::Mk0 { exp } => {
                let ref exp = **exp;
                assert_eq!(idlist.len(), 0);
                Ok(exp.substs(&sub_dict))
            }
            Concept::Mksucc {
                objtype: _,
                concept,
                id,
            } => {
                let mut sub_dict = sub_dict;
                let mut idlist = idlist;
                let nid = idlist.pop().unwrap();
                let ref concept = **concept;
                sub_dict.insert(*id, nid);
                // println!("debug {} {}", id, nid);
                concept._subst(idlist, sub_dict)
            }
            Concept::Mksum {
                objtype: _,
                concept: _,
            } => Err(r!("Error: Concept::Mksum cannot be substed.")),
        }
    }
    pub fn substs(&self, sub_dict: &HashMap<i32, i32>) -> Result<Exp, String> {
        match self {
            Concept::Mk0 { exp } => Ok(exp.substs(&sub_dict)),
            Concept::Mksucc {
                objtype: _,
                concept,
                id: _,
            } => concept.substs(sub_dict),
            Concept::Mksum {
                objtype: _,
                concept: _,
            } => Err(r!("Error: Concept::Mksum cannot be substed.")),
        }
    }
    pub fn to_atomexp(&self, ids: Vec<i32>) -> Result<AtomExp, String> {
        let x = self.subst(ids);
        match x? {
            Exp::Atom { atom } => Ok(*atom),
            _ => Err(r!("Error: Concept to AtomExp Failed")),
        }
    }
    pub fn atomexp_name(&self) -> Option<String> {
        let x: Exp = self.get_exp();
        match x {
            Exp::Atom { atom } => Some(atom.get_name()),
            _ => None,
        }
    }
}
#[pymethods]
impl Concept {
    #[new]
    #[inline]
    pub fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
    #[inline]
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn __neg__(&self) -> Self {
        -self
    }
    #[inline]
    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }
    #[inline]
    #[getter]
    pub fn is_sum(&self) -> bool {
        match self {
            Concept::Mksum {
                objtype: _,
                concept: _,
            } => true,
            _ => false,
        }
    }
    fn subst_by_vec(&self, idlist: Vec<i32>) -> PyResult<Exp> {
        let res = self._subst(idlist, HashMap::new());
        match res {
            Ok(exp) => Ok(exp),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
    fn subst_by_dict(&self, sub_dict: HashMap<i32, i32>) -> PyResult<Exp> {
        let res = self.substs(&sub_dict);
        match res {
            Ok(exp) => Ok(exp),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
    #[getter]
    #[inline]
    pub fn get_exp(&self) -> Exp {
        match self {
            Concept::Mk0 { exp } => (**exp).clone(),
            Concept::Mksucc {
                objtype: _,
                concept,
                id: _,
            } => concept.get_exp(),
            Concept::Mksum {
                objtype: _,
                concept,
            } => concept.get_exp(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_complexity(&self) -> i32 {
        self.complexity()
    }
    /// 当前概念可以被 specialize 的哑指标序列
    #[getter]
    #[inline]
    pub fn get_preids(&self) -> Vec<i32> {
        self.get_pre_objtype_id_vec().iter().map(|x| x.1).collect()
    }
    #[getter]
    #[inline]
    pub fn get_pre_objtype_id_vec(&self) -> Vec<(String, i32)> {
        match self {
            Concept::Mk0 { exp: _ } => vec![],
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => {
                let mut s = concept.get_pre_objtype_id_vec();
                s.push((objtype.clone(), *id));
                s
            }
            Concept::Mksum { objtype, concept } => {
                let s = concept.get_pre_objtype_id_vec();
                s.iter().filter(|x| x.0 != *objtype).cloned().collect()
            }
        }
    }
    #[getter]
    #[inline]
    pub fn get_objtype_id_map(&self) -> HashMap<String, HashSet<i32>> {
        match self {
            Concept::Mk0 { exp: _ } => HashMap::new(),
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => {
                let mut res = concept.get_objtype_id_map();
                let res_objtype = res.entry(objtype.clone()).or_insert(HashSet::new());
                res_objtype.insert(*id);
                res
            }
            Concept::Mksum {
                objtype: _,
                concept,
            } => concept.get_objtype_id_map(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_atomexp_name(&self) -> String {
        let x: Exp = self.get_exp();
        match x {
            Exp::Atom { atom } => atom.get_name(),
            _ => panic!("Error: Concept to AtomExp Failed"),
        }
    }
    fn __add__(&self, other: &Concept) -> Option<Concept> {
        (self + other).ok()
    }
    fn __sub__(&self, other: &Concept) -> Option<Concept> {
        (self - other).ok()
    }
    fn __mul__(&self, other: &Concept) -> Option<Concept> {
        (self * other).ok()
    }
    fn __rmul__(&self, other: i32) -> Option<Concept> {
        Some(Mul::mul(self.clone(), other))
    }
    fn __truediv__(&self, other: &Concept) -> Option<Concept> {
        (self / other).ok()
    }
    fn __difft__(&self, ord: i32) -> Option<Concept> {
        self.difft(ord).ok()
    }
    fn __powi__(&self, other: i32) -> Concept {
        self.pow(other)
    }
    fn __partial__(&self, other: &Concept) -> Option<Concept> {
        self.partial_diff(other).ok()
    }
}

/// IExpConfig is a configuration of an experiment.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum IExpConfig {
    From {
        name: String,
    },
    Mk {
        objtype: String,
        expconfig: Box<IExpConfig>,
        id: i32,
    },
    Mkfix {
        object: String,
        expconfig: Box<IExpConfig>,
        id: i32,
    },
}
#[pymethods]
impl IExpConfig {
    #[inline]
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    #[new]
    #[inline]
    pub fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
    #[getter]
    #[inline]
    pub fn get_expname(&self) -> String {
        match self {
            IExpConfig::From { name } => name.clone(),
            IExpConfig::Mk {
                objtype: _,
                expconfig,
                id: _,
            } => expconfig.get_expname(),
            IExpConfig::Mkfix {
                object: _,
                expconfig,
                id: _,
            } => expconfig.get_expname(),
        }
    }
    #[getter]
    #[inline]
    fn get_objtype_id_map(&self) -> HashMap<String, HashSet<i32>> {
        match self {
            IExpConfig::From { name: _ } => HashMap::new(),
            IExpConfig::Mk {
                objtype,
                expconfig,
                id,
            } => {
                let mut res = expconfig.get_objtype_id_map();
                let res_objtype = res.entry(objtype.clone()).or_insert(HashSet::new());
                res_objtype.insert(*id);
                res
            }
            IExpConfig::Mkfix {
                object: _,
                expconfig,
                id: _,
            } => expconfig.get_objtype_id_map(),
        }
    }
    #[getter]
    #[inline]
    fn get_preids(&self) -> Vec<i32> {
        match self {
            IExpConfig::From { name: _ } => vec![],
            IExpConfig::Mk {
                objtype: _,
                expconfig,
                id,
            } => {
                let mut s = expconfig.get_preids();
                s.push(*id);
                s
            }
            IExpConfig::Mkfix {
                object: _,
                expconfig,
                id: _,
            } => expconfig.get_preids(),
        }
    }
    #[getter]
    #[inline]
    fn get_preobjtypes(&self) -> Vec<ObjType> {
        match self {
            IExpConfig::From { name: _ } => vec![],
            IExpConfig::Mk {
                objtype,
                expconfig,
                id: _,
            } => {
                let mut s = expconfig.get_preobjtypes();
                s.push(ObjType::new(&objtype));
                s
            }
            IExpConfig::Mkfix {
                object: _,
                expconfig,
                id: _,
            } => expconfig.get_preobjtypes(),
        }
    }
    #[getter]
    #[inline]
    fn get_relevant_objs(&self) -> HashSet<String> {
        match self {
            IExpConfig::From { name: _ } => HashSet::new(),
            IExpConfig::Mk {
                objtype: _,
                expconfig,
                id: _,
            } => expconfig.get_relevant_objs(),
            IExpConfig::Mkfix {
                object,
                expconfig,
                id: _,
            } => {
                let mut s = expconfig.get_relevant_objs();
                s.insert(object.clone());
                s
            }
        }
    }
}

/// Intrinsic is a special Exp object which defines a measurement in an experiment.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum Intrinsic {
    From { sexp: Box<SExp> },
}
impl Intrinsic {
    pub fn get_sexp(&self) -> SExp {
        match self {
            Intrinsic::From { sexp } => (**sexp).clone(),
        }
    }
    pub fn get_objtype_id_map(&self) -> HashMap<String, HashSet<i32>> {
        self.get_sexp().get_objtype_id_map()
    }
    pub fn get_preids(&self) -> Vec<i32> {
        self.get_sexp().get_expconfig().get_preids()
    }
}
#[pymethods]
impl Intrinsic {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn __neg__(&self) -> Self {
        Intrinsic::From {
            sexp: Box::new(self.get_sexp().__neg__()),
        }
    }
    fn __inv__(&self) -> Self {
        Intrinsic::From {
            sexp: Box::new(self.get_sexp().__inv__()),
        }
    }
    #[new]
    #[inline]
    pub fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
    #[getter]
    #[inline]
    fn get_relevant_objs(&self) -> HashSet<String> {
        self.get_sexp().get_relevant_objs()
    }
    #[getter]
    #[inline]
    fn get_input_objids(&self) -> Vec<i32> {
        self.get_sexp().get_expconfig().get_preids()
    }
    #[getter]
    #[inline]
    pub fn get_input_objtypes(&self) -> Vec<ObjType> {
        self.get_sexp().get_expconfig().get_preobjtypes()
    }
    #[getter]
    #[inline]
    pub fn measure_experiment(&self) -> String {
        let sexp = self.get_sexp();
        sexp.get_expconfig().get_expname()
    }
}

/// Expression is a general object that can be a Exp, SExp, Concept, Intrinsic, or Proposition.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum Expression {
    Exp { exp: Box<Exp> },
    SExp { sexp: Box<SExp> },
    Concept { concept: Box<Concept> },
    Intrinsic { intrinsic: Box<Intrinsic> },
    Proposition { prop: Box<Proposition> },
}

#[pymethods]
impl AtomExp {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }
    #[staticmethod]
    pub fn get_t() -> Self {
        AtomExp::VariableIds {
            name: r!("t"),
            ids: vec![0],
        }
    }
    #[getter]
    #[inline]
    pub fn get_name(&self) -> String {
        match self {
            AtomExp::Variable { name } => name.clone(),
            AtomExp::VariableIds { name, ids: _ } => name.clone(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_vec_ids(&self) -> Vec<i32> {
        match self {
            AtomExp::Variable { name: _ } => vec![],
            AtomExp::VariableIds { name: _, ids } => ids.clone(),
        }
    }
    #[getter]
    #[inline]
    pub fn get_allids(&self) -> HashSet<i32> {
        match self {
            AtomExp::Variable { name: _ } => HashSet::new(),
            AtomExp::VariableIds { name: _, ids } => {
                let mut res = HashSet::new();
                for id in ids.iter() {
                    res.insert(*id);
                }
                res
            }
        }
    }
    pub fn subst(&self, oid: i32, nid: i32) -> Self {
        match self {
            AtomExp::Variable { name } => AtomExp::Variable { name: name.clone() },
            AtomExp::VariableIds { name, ids } => {
                let ids = ids.clone();
                let mut res = Vec::new();
                for id in ids.iter() {
                    if *id == oid {
                        res.push(nid);
                    } else {
                        res.push(*id);
                    }
                }
                AtomExp::VariableIds {
                    name: name.clone(),
                    ids: res,
                }
            }
        }
    }
    pub fn substs(&self, sub_dict: HashMap<i32, i32>) -> Self {
        match self {
            AtomExp::Variable { name } => AtomExp::Variable { name: name.clone() },
            AtomExp::VariableIds { name, ids } => {
                let ids = ids.clone();
                let mut res = Vec::new();
                for id in ids.iter() {
                    match sub_dict.get(id) {
                        Some(nid) => res.push(*nid),
                        None => res.push(*id),
                    }
                }
                AtomExp::VariableIds {
                    name: name.clone(),
                    ids: res,
                }
            }
        }
    }
    #[new]
    pub fn from_string(str: String) -> PyResult<Self> {
        Self::from_str(&str).map_err(|e| PyValueError::new_err(e))
    }
}
impl AtomExp {
    #[inline]
    pub fn new_variable_ids(name: String, ids: Vec<i32>) -> Self {
        if ids.len() == 0 {
            AtomExp::Variable { name }
        } else {
            AtomExp::VariableIds { name, ids }
        }
    }
    #[inline]
    pub fn new_variable(name: String) -> Self {
        AtomExp::Variable { name }
    }
}
