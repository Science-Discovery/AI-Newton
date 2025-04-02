/// This file defines the complexity trait for the language.
/// The complexity trait is used to estimate the complexity of an expression.

use crate::r;
use super::{
    AtomExp, BinaryOp, Concept, Exp, Proposition
};

pub trait Complexity<T: PartialOrd> {
    fn complexity(&self) -> T;
}

impl Complexity<i32> for AtomExp {
    fn complexity(&self) -> i32 {
        match self {
            AtomExp::Variable { name:_ } => 5,
            AtomExp::VariableIds { name, ids } => {
                if *name == r!("t") {
                    5
                } else {
                    ids.len() as i32 * 10
                }
            },
        }
    }
}

impl Complexity<i32> for BinaryOp {
    fn complexity(&self) -> i32 {
        match self {
            BinaryOp::Add => 2,
            BinaryOp::Sub => 2,
            BinaryOp::Mul => 4,
            BinaryOp::Div => 5,
            BinaryOp::Pow => 6,
        }
    }
}

impl Complexity<i32> for Exp {
    fn complexity(&self) -> i32 {
        match self {
            Exp::Atom { atom } => atom.complexity(),
            Exp::Number { num: _ } => 5,
            Exp::BinaryExp { left, op, right } => {
                let lids = i32::max(left.obj_number() as i32 - 1, 0);
                let rids = i32::max(right.obj_number() as i32 - 1, 0);
                left.complexity() + right.complexity() + op.complexity() + (lids + rids) * 2
            },
            Exp::UnaryExp { op: _, exp } => exp.complexity() + 5,
            Exp::DiffExp { left, right, ord } => {
                let lids = i32::max(left.obj_number() as i32 - 1, 0);
                let rids = i32::max(right.obj_number() as i32 - 1, 0);
                left.complexity() + right.complexity() * 2 + ord * 8 + lids * 2 + rids * 4
            },
            Exp::ExpWithMeasureType { exp, measuretype: _ } => exp.complexity(),
            Exp::Partial { left, right } => {
                let lids = i32::max(left.obj_number() as i32 - 1, 0);
                left.complexity() + right.complexity() + lids * 2
            }
        }
    }
}

impl Complexity<i32> for Concept {
    fn complexity(&self) -> i32 {
        match self {
            Concept::Mk0 { exp } => exp.complexity(),
            Concept::Mksucc { objtype: _, concept, id: _ } => {
                concept.complexity()
            },
            Concept::Mksum { objtype: _, concept } => {
                concept.complexity()
            }
        }
    }
}

impl Complexity<i32> for Proposition {
    fn complexity(&self) -> i32 {
        match self {
            Proposition::Conserved { concept } => {
                concept.complexity() + 20
            }
            Proposition::Zero { concept } => {
                concept.complexity()
            }
            Proposition::IsConserved { exp } => {
                exp.complexity() + 20
            }
            Proposition::IsZero { exp } => {
                exp.complexity()
            }
            Proposition::Eq { left, right } => {
                left.complexity() + right.complexity()
            }
            Proposition::Not { prop } => {
                prop.complexity()
            }
        }
    }
}