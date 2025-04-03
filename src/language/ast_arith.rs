use super::{BinaryOp, Concept, Exp, UnaryOp};
use crate::r;
/// This file defines the arithmetic operations for the AST.
pub use num_traits::Pow;
use std::collections::{HashMap, HashSet};
pub use std::ops::{Add, Div, Mul, Neg, Sub};

impl Concept {
    fn make_from_exp(exp: Exp, objids: Vec<(String, i32)>) -> Self {
        let mut concept0 = Concept::Mk0 { exp: Box::new(exp) };
        for i in 0..objids.len() {
            concept0 = Concept::Mksucc {
                objtype: objids[i].0.clone(),
                concept: Box::new(concept0),
                id: objids[i].1,
            }
        }
        concept0
    }
}

impl Pow<Exp> for Exp {
    type Output = Self;
    #[inline]
    fn pow(self, exp: Exp) -> Self::Output {
        match exp {
            Exp::Number { num } => self.pow(num),
            _ => unimplemented!(),
        }
    }
}

impl Pow<i32> for Exp {
    type Output = Self;
    #[inline]
    fn pow(self, exp: i32) -> Self::Output {
        match exp {
            0 => Exp::Number { num: 1 },
            1 => self,
            _ => Exp::BinaryExp {
                left: Box::new(self),
                op: BinaryOp::Pow,
                right: Box::new(Exp::Number { num: exp }),
            },
        }
    }
}

impl Add for Exp {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) {
            return rhs;
        }
        if rhs.unwrap_number() == Some(0) {
            return self;
        }
        if self > rhs {
            Exp::BinaryExp {
                left: Box::new(rhs),
                op: BinaryOp::Add,
                right: Box::new(self),
            }
        } else {
            Exp::BinaryExp {
                left: Box::new(self),
                op: BinaryOp::Add,
                right: Box::new(rhs),
            }
        }
    }
}

impl Add for &Exp {
    type Output = Exp;
    fn add(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) {
            return rhs.clone();
        }
        if rhs.unwrap_number() == Some(0) {
            return self.clone();
        }
        if self > rhs {
            Exp::BinaryExp {
                left: Box::new(rhs.clone()),
                op: BinaryOp::Add,
                right: Box::new(self.clone()),
            }
        } else {
            Exp::BinaryExp {
                left: Box::new(self.clone()),
                op: BinaryOp::Add,
                right: Box::new(rhs.clone()),
            }
        }
    }
}

impl Sub for Exp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.unwrap_number() == Some(0) {
            return self;
        }
        Exp::BinaryExp {
            left: Box::new(self),
            op: BinaryOp::Sub,
            right: Box::new(rhs),
        }
    }
}

impl Sub for &Exp {
    type Output = Exp;
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.unwrap_number() == Some(0) {
            return self.clone();
        }
        Exp::BinaryExp {
            left: Box::new(self.clone()),
            op: BinaryOp::Sub,
            right: Box::new(rhs.clone()),
        }
    }
}

impl Neg for Exp {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match &self {
            Exp::Number { num } => Exp::Number { num: -num },
            Exp::UnaryExp { op, exp } => match op {
                UnaryOp::Neg => *exp.clone(),
                UnaryOp::Diff => self,
            },
            // TODO, BinaryExp case
            _ => Exp::UnaryExp {
                op: UnaryOp::Neg,
                exp: Box::new(self),
            },
        }
    }
}

impl Neg for &Exp {
    type Output = Exp;
    fn neg(self) -> Self::Output {
        match self {
            Exp::Number { num } => Exp::Number { num: -num },
            Exp::UnaryExp { op, exp } => match op {
                UnaryOp::Neg => *exp.clone(),
                UnaryOp::Diff => self.clone(),
            },
            _ => Exp::UnaryExp {
                op: UnaryOp::Neg,
                exp: Box::new(self.clone()),
            },
        }
    }
}

impl Mul for Exp {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) || rhs.unwrap_number() == Some(0) {
            return Exp::Number { num: 0 };
        }
        if self.unwrap_number() == Some(1) {
            return rhs;
        }
        if rhs.unwrap_number() == Some(1) {
            return self;
        }
        if self > rhs {
            Exp::BinaryExp {
                left: Box::new(rhs),
                op: BinaryOp::Mul,
                right: Box::new(self),
            }
        } else {
            Exp::BinaryExp {
                left: Box::new(self),
                op: BinaryOp::Mul,
                right: Box::new(rhs),
            }
        }
    }
}

impl Mul for &Exp {
    type Output = Exp;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) || rhs.unwrap_number() == Some(0) {
            return Exp::Number { num: 0 };
        }
        if self.unwrap_number() == Some(1) {
            return rhs.clone();
        }
        if rhs.unwrap_number() == Some(1) {
            return self.clone();
        }
        if self > rhs {
            Exp::BinaryExp {
                left: Box::new(rhs.clone()),
                op: BinaryOp::Mul,
                right: Box::new(self.clone()),
            }
        } else {
            Exp::BinaryExp {
                left: Box::new(self.clone()),
                op: BinaryOp::Mul,
                right: Box::new(rhs.clone()),
            }
        }
    }
}

impl Mul<i32> for Exp {
    type Output = Exp;
    fn mul(self, rhs: i32) -> Self::Output {
        self * Exp::Number { num: rhs }
    }
}

impl Div for Exp {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) {
            return Exp::Number { num: 0 };
        }
        if rhs.unwrap_number() == Some(1) {
            return self;
        }
        Exp::BinaryExp {
            left: Box::new(self),
            op: BinaryOp::Div,
            right: Box::new(rhs),
        }
    }
}

impl Div for &Exp {
    type Output = Exp;
    fn div(self, rhs: Self) -> Self::Output {
        if self.unwrap_number() == Some(0) {
            return Exp::Number { num: 0 };
        }
        if rhs.unwrap_number() == Some(1) {
            return self.clone();
        }
        Exp::BinaryExp {
            left: Box::new(self.clone()),
            op: BinaryOp::Div,
            right: Box::new(rhs.clone()),
        }
    }
}

fn union_preids(
    vecids: Vec<(String, i32)>,
    vecids2: Vec<(String, i32)>,
) -> Option<(Vec<(String, i32)>, Vec<i32>, Vec<i32>)> {
    let set1: HashSet<_> = vecids.iter().cloned().collect();
    let set2: HashSet<_> = vecids2.iter().cloned().collect();
    // if set1 subset set2 or set2 subset set1
    if set1.is_subset(&set2) {
        let res2: Vec<i32> = (1..(vecids2.len() + 1) as i32).collect();
        let map: HashMap<_, i32> = res2
            .iter()
            .map(|x| (vecids2[*x as usize - 1].1, *x))
            .collect();
        let res1 = vecids.iter().map(|x| *map.get(&x.1).unwrap()).collect();
        return Some((vecids2, res1, res2));
    }
    if set2.is_subset(&set1) {
        let res1: Vec<i32> = (1..(vecids.len() + 1) as i32).collect();
        let map: HashMap<_, i32> = res1
            .iter()
            .map(|x| (vecids[*x as usize - 1].1, *x))
            .collect();
        let res2 = vecids2.iter().map(|x| *map.get(&x.1).unwrap()).collect();
        return Some((vecids, res1, res2));
    }
    return None;
}

impl Concept {
    fn unwrap_exp(&self) -> Result<Exp, String> {
        match self {
            Concept::Mk0 { exp } => Ok(*exp.clone()),
            Concept::Mksucc {
                objtype: _,
                concept: _,
                id: _,
            } => Err(r!("Error: Concept unwrap_exp failed")),
            Concept::Mksum {
                objtype: _,
                concept: _,
            } => Err(r!("Error: Concept unwrap_exp failed")),
        }
    }
}

impl Add for Concept {
    type Output = Result<Concept, String>;
    fn add(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Add failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 + exp2, vecids))
            },
        )
    }
}

impl Add for &Concept {
    type Output = Result<Concept, String>;
    fn add(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Add failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 + exp2, vecids))
            },
        )
    }
}

impl Sub for Concept {
    type Output = Result<Concept, String>;
    fn sub(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Sub failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 - exp2, vecids))
            },
        )
    }
}

impl Sub for &Concept {
    type Output = Result<Concept, String>;
    fn sub(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Sub failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 - exp2, vecids))
            },
        )
    }
}

impl Mul for Concept {
    type Output = Result<Concept, String>;
    fn mul(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Mul failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 * exp2, vecids))
            },
        )
    }
}

impl Mul for &Concept {
    type Output = Result<Concept, String>;
    fn mul(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Mul failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 * exp2, vecids))
            },
        )
    }
}

impl Div for Concept {
    type Output = Result<Concept, String>;
    fn div(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Div failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 / exp2, vecids))
            },
        )
    }
}

impl Div for &Concept {
    type Output = Result<Concept, String>;
    fn div(self, rhs: Self) -> Self::Output {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept Div failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let exp2 = rhs.partial_subst(ids2)?.unwrap_exp()?;
                Ok(Concept::make_from_exp(exp1 / exp2, vecids))
            },
        )
    }
}

impl Concept {
    pub fn partial_diff(&self, rhs: &Concept) -> Result<Self, String> {
        let vecids = union_preids(self.get_pre_objtype_id_vec(), rhs.get_pre_objtype_id_vec());
        vecids.map_or(
            Err(r!("Error: Concept partial_diff failed")),
            |(vecids, ids1, ids2)| {
                let exp1 = self.partial_subst(ids1)?.unwrap_exp()?;
                let atomexp2 = rhs.partial_subst(ids2)?.to_atomexp(vec![])?;
                Ok(Concept::make_from_exp(exp1.__partial__(&atomexp2), vecids))
            },
        )
    }
}

impl Concept {
    pub fn difft(&self, ord: i32) -> Result<Self, String> {
        let vecids = self.get_pre_objtype_id_vec();
        let ids: Vec<i32> = (1..(vecids.len() + 1) as i32).collect();
        let exp1 = self.partial_subst(ids.clone())?.unwrap_exp()?;
        Ok(Concept::make_from_exp(exp1.__difft__(ord), vecids))
    }
}

impl Mul<i32> for Concept {
    type Output = Concept;
    fn mul(self, rhs: i32) -> Self::Output {
        match self {
            Concept::Mk0 { exp } => Concept::Mk0 {
                exp: Box::new(*exp * Exp::Number { num: rhs }),
            },
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => Concept::Mksucc {
                objtype,
                concept: Box::new(*concept * rhs),
                id,
            },
            Concept::Mksum { objtype, concept } => Concept::Mksum {
                objtype,
                concept: Box::new(*concept * rhs),
            },
        }
    }
}

impl Neg for Concept {
    type Output = Concept;
    fn neg(self) -> Self::Output {
        match self {
            Concept::Mk0 { exp } => Concept::Mk0 {
                exp: Box::new(-*exp.clone()),
            },
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => Concept::Mksucc {
                objtype: objtype.clone(),
                concept: Box::new(-*concept),
                id: id,
            },
            Concept::Mksum { objtype, concept } => Concept::Mksum {
                objtype: objtype.clone(),
                concept: Box::new(-*concept),
            },
        }
    }
}

impl Neg for &Concept {
    type Output = Concept;
    fn neg(self) -> Self::Output {
        match self {
            Concept::Mk0 { exp } => Concept::Mk0 {
                exp: Box::new(-*exp.clone()),
            },
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => Concept::Mksucc {
                objtype: objtype.clone(),
                concept: Box::new(-concept.as_ref()),
                id: *id,
            },
            Concept::Mksum { objtype, concept } => Concept::Mksum {
                objtype: objtype.clone(),
                concept: Box::new(-concept.as_ref()),
            },
        }
    }
}

impl Pow<i32> for Concept {
    type Output = Concept;
    fn pow(self, other: i32) -> Self::Output {
        match self {
            Concept::Mk0 { exp } => Concept::Mk0 {
                exp: Box::new(exp.pow(other)),
            },
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => Concept::Mksucc {
                objtype: objtype.clone(),
                concept: Box::new(concept.pow(other)),
                id: id,
            },
            Concept::Mksum { objtype, concept } => Concept::Mksum {
                objtype: objtype.clone(),
                concept: Box::new(concept.pow(other)),
            },
        }
    }
}

impl Pow<i32> for &Concept {
    type Output = Concept;
    fn pow(self, other: i32) -> Self::Output {
        match self {
            Concept::Mk0 { exp } => Concept::Mk0 {
                exp: Box::new(exp.clone().pow(other)),
            },
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => Concept::Mksucc {
                objtype: objtype.clone(),
                concept: Box::new(concept.as_ref().pow(other)),
                id: *id,
            },
            Concept::Mksum { objtype, concept } => Concept::Mksum {
                objtype: objtype.clone(),
                concept: Box::new(concept.as_ref().pow(other)),
            },
        }
    }
}
