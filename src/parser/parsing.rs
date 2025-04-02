/// This file contains the functions for parsing a string into an expression object.

use crate::language::UnaryOp;
use crate::experiments::ObjType;
use crate::knowledge::Knowledge;
use crate::r;
pub use std::str::FromStr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_yaml::from_str;
use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub expr);
lalrpop_mod!(pub sympy);

use crate::experiments::{
    Objstructure, Parastructure
};

impl FromStr for Objstructure {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ObjstructureParser::new().parse(s).map_err(
            |e| e.to_string()
        )
    }
}

// Function for parsing a string into an expression
#[pyfunction]
fn parse(input: &str) -> PyResult<Expression> {
    Expression::from_str(input).map_err(|e| PyErr::new::<PyValueError, _>(e))
}

pub fn parse_knowledge(input: &str) -> Result<(Vec<(String, Objstructure)>, Vec<(String, Expression)>, Vec<(String, Proposition)>), String> {
    expr::KnowledgeParser::new().parse(input).map_err(
        |e| e.to_string()
    )
}

#[pyfunction]
pub fn parse_sympy_to_exp(input: &str) -> PyResult<Exp> {
    sympy::ExpParser::new().parse(input).map(|e| *e).map_err(
        |e| PyValueError::new_err(
            input.to_string() + &e.to_string()
        )
    )
}


use crate::language::ast::{Proposition, AtomExp, Exp, SExp, Concept, IExpConfig, Intrinsic, Expression};

#[pymethods]
impl Expression {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    #[new]
    fn from_string(input: &str) -> PyResult<Self> {
        input.parse().map_err(|e| PyValueError::new_err(e))
    }
    #[getter]#[inline]
    pub fn expr_type(&self) -> String {
        match self {
            Expression::Exp {exp: _} => r!("Exp"),
            Expression::SExp {sexp: _} => r!("SExp"),
            Expression::Concept {concept: _} => r!("Concept"),
            Expression::Intrinsic {intrinsic: _} => r!("Intrinsic"),
            Expression::Proposition {prop: _} => r!("Proposition"),
        }
    }
    #[getter]#[inline]
    fn unwrap_exp(&self) -> Exp {
        match self {
            Expression::Exp {exp} => *exp.clone(),
            _ => panic!("Not an Exp"),
        }
    }
    #[getter]#[inline]
    fn unwrap_concept(&self) -> Concept {
        match self {
            Expression::Concept {concept} => *concept.clone(),
            _ => panic!("Not a Concept"),
        }
    }
    #[getter]#[inline]
    fn unwrap_sexp(&self) -> SExp {
        match self {
            Expression::SExp {sexp} => *sexp.clone(),
            _ => panic!("Not a SExp"),
        }
    }
    #[getter]#[inline]
    fn unwrap_intrinsic(&self) -> Intrinsic {
        match self {
            Expression::Intrinsic {intrinsic} => *intrinsic.clone(),
            _ => panic!("Not an Intrinsic"),
        }
    }
    #[getter]#[inline]
    fn unwrap_proposition(&self) -> Proposition {
        match self {
            Expression::Proposition {prop} => *prop.clone(),
            _ => panic!("Not a Proposition"),
        }
    }
}

impl FromStr for Expression {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ExpressionParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for Exp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ExpParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for SExp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::SExpParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for Concept {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ConceptParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for Intrinsic {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::IntrinsicParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for AtomExp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::AtomExpParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for Proposition {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::PropositionParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for IExpConfig {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ExpConfigParser::new().parse(s).map_err(
            |e| e.to_string()
        ).map(|e| *e)
    }
}

impl FromStr for ObjType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ObjType { obj: s.to_string() })
    }
}

impl FromStr for Parastructure {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expr::ParastructureParser::new().parse(s).map_err(
            |e| e.to_string()
        )
    }
}

impl FromStr for Knowledge {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let end = s.find("[end]").unwrap();
        let s1 = &s[..(end+5)];
        let s2 = &s[(end+6)..];
        assert_eq!(s2.find("[Experiments]"), Some(0));
        assert_eq!(s2.find("[end]"), Some(s2.len()-5));
        let yaml_str = &s2[14..(s2.len()-5)];
        let experiments = from_str(&yaml_str).unwrap();
        let knowledge_item = parse_knowledge(&s1)?;
        let mut knowledge = Knowledge::new_with_experiments(experiments);
        for (name, obj) in knowledge_item.0 {
            knowledge.register_object(name, obj);
        }
        for (name, exp) in knowledge_item.1 {
            knowledge._register_expression(name, exp)?;
        }
        for (name, prop) in knowledge_item.2 {
            knowledge._register_conclusion(name, prop)?;
        }
        Ok(knowledge)
    }
}


impl Knowledge {
    pub fn _parse_atomexp_to_sympy_str(&self, input: &AtomExp, argument: String) -> String {
        let res = match input {
            AtomExp::Variable { name } => format!("{}", name),
            AtomExp::VariableIds { name, ids } => {
                if ids.len() == 0 {
                    format!("{}", name)
                } else {
                    format!("{}_{}", name, ids.iter().map(|x| format!("{}", x)).collect::<Vec<String>>().join("_"))
                }
            }
        };
        let not_with_argument = self._made_of_obj_attr(&Exp::Atom { atom: Box::new(input.clone()) }).unwrap();
        if res == argument || not_with_argument {
            res
        } else {
            format!("{}({})", res, argument)
        }
    }

    pub fn _parse_exp_to_sympy_str(&self, input: &Exp, argument: String) -> String {
        match input {
            Exp::Number { num } => format!("{}", num),
            Exp::Atom { atom } => self.parse_atomexp_to_sympy_str(atom.as_ref(), argument),
            Exp::UnaryExp { op, exp } => {
                match op {
                    UnaryOp::Neg => format!("(-{})", self.parse_exp_to_sympy_str(exp.as_ref(), argument)),
                    UnaryOp::Diff => {
                        let s = self.parse_exp_to_sympy_str(exp.as_ref(), argument.clone());
                        if s == argument { r!("1") } else { 
                            format!("Derivative({}, {})", s, argument)
                        }
                    },
                }
            }
            Exp::BinaryExp { left, op, right } => {
                format!(
                    "({} {} {})",
                    self.parse_exp_to_sympy_str(left.as_ref(), argument.clone()),
                    op,
                    self.parse_exp_to_sympy_str(right.as_ref(), argument)
                )
            }
            Exp::DiffExp { left, right, ord } => {
                let left = self.parse_exp_to_sympy_str(left.as_ref(), argument.clone());
                let right = self.parse_exp_to_sympy_str(right.as_ref(), argument.clone());
                if right == argument { return format!("Derivative({}, {}, {})", left, right, *ord); }
                let mut res = if left == argument {
                    format!("(1 / Derivative({}, {}))", right, argument)
                } else {
                    format!("(Derivative({}, {}) / Derivative({}, {}))", left, argument, right, argument)
                };
                for _ in 1..*ord {
                    res = format!(
                        "(Derivative({}, {}) / Derivative({}, {}))",
                        res, argument, right, argument
                    );
                }
                res
            }
            Exp::ExpWithMeasureType { exp, measuretype:_ } => {
                self.parse_exp_to_sympy_str(exp.as_ref(), argument)
            }
            Exp::Partial { left:_, right:_ } => {
                panic!("Partial not supported yet");
            }
        }
    }
}


#[pymodule]
pub fn register_sentence(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(m.py(), "sentence")?;
    child_module.add_function(wrap_pyfunction!(parse, m)?)?;
    child_module.add_function(wrap_pyfunction!(parse_sympy_to_exp, m)?)?;
    m.add_submodule(&child_module)?;
    Ok(())
}
