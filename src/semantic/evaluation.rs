/// This file defines the evaluation functions that calculate the value of an expression under a given context.
/// The context contains the experimental data, the object settings, and physical concepts (including Intrinsics).
/// So we can transform an Exp object into a ExpData object.
///
/// Given the `concepts` dictionary, we can expand a normal Exp object into its `raw definition` form.
/// For example, the expression `momentum[1]` can be expanded into `mass[1] * D[posx[1]]/D[t[0]]`.

use crate::r;
use std::collections::HashMap;

use crate::expdata::{Diff, ExpData, ConstData};
use crate::knowledge::Knowledge;
use crate::experiments::{ExpStructure, Objstructure};
use crate::language::*;



pub fn apply_binary_op<T>(op: &BinaryOp, valuei: T, valuej: T) -> T
where T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + num_traits::Pow<T, Output = T> {
    match op {
        BinaryOp::Add => valuei + valuej,
        BinaryOp::Sub => valuei - valuej,
        BinaryOp::Mul => valuei * valuej,
        BinaryOp::Div => valuei / valuej,
        BinaryOp::Pow => valuei.pow(valuej),
    }
}



pub fn _concept_apply_ids(concept: &Concept, ids: Vec<i32>, context: Option<&ExpStructure>,
    concepts: &HashMap<String, Expression>) -> Result<Exp, String> {
    let con = concept.partial_subst(ids.clone())?;
    match con {
        Concept::Mk0 { exp } => Ok(*exp),
        _ => match context {
            None => Err(format!("exp_name is None in concept_apply_ids({}, {:?})", concept, ids)),
            Some(context) => {
                let res = context._specialize(&con, concepts);
                match res.len() {
                    0 => Err(format!("No data found for {}", con)),
                    1 => Ok(res[0].clone()),
                    _ => Err(format!("More than one result in concept_apply_ids({}, {:?}", concept, ids))
                }
            }
        }
    }
}

pub fn _raw_definition_exp(exp: &Exp, context: Option<&ExpStructure>, concepts: &HashMap<String, Expression>) -> Exp {
    match exp {
        Exp::Number { num: _ } => {
            exp.clone()
        }
        Exp::Atom { atom } => {
            let atom = atom.as_ref();
            if let Some(expr) = concepts.get(&atom.get_name()) {
                match expr {
                    Expression::Intrinsic { intrinsic: _ } => {
                        exp.clone()
                    }
                    Expression::Concept { concept } => {
                        if concept.atomexp_name() == Some(atom.get_name()) {
                            return exp.clone();  // 基本概念返回自己
                        }
                        let concept_new = _concept_apply_ids(concept, atom.get_vec_ids(), context, concepts);
                        match concept_new {
                            Ok(concept_new) => _raw_definition_exp(&concept_new, context, concepts),
                            Err(_) => exp.clone()
                        }
                    }
                    _ => unimplemented!()
                }
            } else {
                exp.clone()
            }
        }
        Exp::BinaryExp { left, op, right } => {
            let left = _raw_definition_exp(&*left, context, concepts);
            let right = _raw_definition_exp(&*right, context, concepts);
            Exp::BinaryExp { left: Box::new(left), op: op.clone(), right: Box::new(right) }
        }
        Exp::UnaryExp { op, exp } => {
            let exp = _raw_definition_exp(&*exp, context, concepts);
            Exp::UnaryExp { op: op.clone(), exp: Box::new(exp) }
        }
        Exp::DiffExp { left, right, ord } => {
            let left = _raw_definition_exp(&*left, context, concepts);
            let right = _raw_definition_exp(&*right, context, concepts);
            Exp::DiffExp { left: Box::new(left), right: Box::new(right), ord: *ord }
        }
        Exp::ExpWithMeasureType { exp, measuretype } => {
            let exp = _raw_definition_exp(&*exp, context, concepts);
            Exp::ExpWithMeasureType { exp: Box::new(exp), measuretype: measuretype.clone() }
        }
        Exp::Partial { left, right } => {
            let left = _raw_definition_exp(&*left, context, concepts);
            Exp::Partial { left: Box::new(left), right: right.clone() }
        }
    }
}


impl Knowledge {
    pub fn _get_expstructure(&self, expconfig: &IExpConfig, objsettings: Vec<Objstructure> ) -> Result<ExpStructure, String> {
        match expconfig {
            IExpConfig::From { name } => {
                if objsettings.len() != 0 {
                    return Err(format!("length of objsettings does not match with {}", expconfig));
                }
                let mut exp = (*self.experiments.get(name).unwrap()).clone();
                exp.random_sample();
                Ok(exp)
            }
            IExpConfig::Mk { objtype, expconfig, id } => {
                let mut objsettings = objsettings;
                let obj = objsettings.pop().unwrap();
                if obj.obj_type.to_string() != *objtype {
                    return Err(format!("objtype {} in objsettings does not match with {}", obj.obj_type, expconfig));
                }
                let mut exp = self._get_expstructure(expconfig, objsettings)?;
                exp.set_obj(*id, obj);
                Ok(exp)
            }
            IExpConfig::Mkfix { object, expconfig, id } => {
                let obj = self.objects.get(object).unwrap();
                let mut exp = self._get_expstructure(expconfig, objsettings)?;
                exp.set_obj(*id, obj.clone());
                Ok(exp)
            }
        }
    }
    pub fn _eval_intrinsic(&self, intrinsic: &Intrinsic, objsettings: Vec<Objstructure>) -> Result<ConstData, String> {
        match intrinsic {
            Intrinsic::From { sexp } => {
                let sexp = sexp.as_ref();
                match sexp {
                    SExp::Mk { expconfig, exp } => {
                        let expconfig = expconfig.as_ref();
                        let mut total_time = 0;
                        loop {
                            total_time += 1;
                            if total_time > 5 {
                                return Err(format!("failed to evaluate intrinsic concept {}", intrinsic));
                            }
                            let mut context = self._get_expstructure(expconfig, objsettings.clone())?;
                            let exp = exp.as_ref();
                            if let Some(res) = self._eval(exp, &mut context)?.force_to_const_data() {
                                return Ok(res);
                            }
                        };
                    }
                }
            }
        }
    }
    pub fn _eval_prop(&self, prop: &Proposition, context: &mut ExpStructure) -> Result<bool, String> {
        match prop {
            Proposition::Conserved { concept } => {
                let exp_list = self._specialize(&concept, context);
                for exp in exp_list.iter() {
                    if !self._eval(exp, context)?.is_conserved() {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Proposition::Zero { concept } => {
                let exp_list = self._specialize(&concept, context);
                for exp in exp_list.iter() {
                    if !self._eval(exp, context)?.is_zero() {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Proposition::IsConserved { exp } => {
                Ok(self._eval(&exp, context)?.is_conserved())
            }
            Proposition::IsZero { exp } => {
                Ok(self._eval(&exp, context)?.is_zero())
            }
            _ => unimplemented!()
        }
    }
    pub fn _eval(&self, exp0: &Exp, context: &mut ExpStructure) -> Result<ExpData, String> {
        if context.expdata_is_none() {
            match exp0 {
                Exp::ExpWithMeasureType { exp: _, measuretype } => {
                    context.calc_expdata((**measuretype).clone())?;
                }
                _ => {
                    context.calc_expdata(MeasureType::default())?;
                }
            }
        }
        assert!(!context.expdata_is_none());
        let data = context.get_ref_expdata();
        match exp0 {
            Exp::ExpWithMeasureType { exp, measuretype } => {
                assert!(**measuretype == data.measuretype);
                let exp = exp.as_ref();
                self.__eval(exp, context)
            }
            _ => {
                self.__eval(exp0, context)
            }
        }
    }
    fn __eval(&self, exp0: &Exp, context: &mut ExpStructure) -> Result<ExpData, String> {
        assert!(!context.expdata_is_none());
        let data = context.get_ref_expdata();
        match exp0 {
            Exp::ExpWithMeasureType { exp: _, measuretype: _ } => {
                Err(r!("ExpWithMeasureType should be handled in eval"))
            }
            Exp::Number { num } => {
                Ok(ExpData::from_exact_const(*num))
            }
            Exp::Atom { atom } => {
                let atom = atom.as_ref();
                // println!("atom = {}", atom);
                let expdata = data.get_data().get_data_by_key(atom);
                match expdata {
                    Ok(_) => expdata,
                    Err(_) => {
                        // println!("{} {}", atom, atom.get_name());
                        let expr = match self.concepts.get(&atom.get_name()) {
                            Some(expr) => expr,
                            None => return Err(format!("No concept found for atom {}", atom))
                        };
                        match expr {
                            Expression::Intrinsic { intrinsic } => {
                                let mut objs = vec![];
                                for id in atom.get_allids().iter() {
                                    objs.push(context.get_obj(*id).clone());
                                }
                                let expdata = {
                                    if let Some(constdata) = self.eval_intrinsic(intrinsic, objs) {
                                        // println!("constdata = {}", constdata);
                                        constdata.into()
                                    } else {
                                        ExpData::Err { }
                                    }
                                };
                                context.get_mut_expdata().set_data(atom.clone(), expdata.clone());
                                Ok(expdata)
                            }
                            Expression::Concept { concept } => {
                                if concept.atomexp_name() == Some(atom.get_name()) {
                                    // 基本概念且实验数据中没有这个概念的数据
                                    Err(format!("No expdata found for basic concept {}", atom))
                                }
                                else {
                                    let concept_new = self._concept_apply_ids(
                                        concept.as_ref(), atom.get_vec_ids(), 
                                        Some(context))?;
                                    let expdata = self.__eval(&concept_new, context);
                                    context.get_mut_expdata().set_data(atom.clone(), expdata.clone()?);
                                    expdata
                                }
                            }
                            _ => unimplemented!()
                        }
                    }
                }
            }
            Exp::UnaryExp { op: UnaryOp::Neg, ref exp } => Ok(-self.__eval(&*exp, context)?),
            Exp::UnaryExp { op: UnaryOp::Diff, ref exp } => Ok(self.__eval(&*exp, context)?.diff_tau()),
            Exp::BinaryExp { op, ref left, ref right } => 
                Ok(apply_binary_op(op, self.__eval(&*left, context)?, self.__eval(&*right, context)?)),
            Exp::DiffExp { ref left, ref right, ord} =>
                Ok((&self.__eval(&*left, context)?).diff_n(&self.__eval(&*right, context)?, *ord as usize)),
            Exp::Partial { left:_, right:_ } => {
                unimplemented!()
            }
        }
    }
    pub fn _check_geometry_info(&self, context: &mut ExpStructure) -> Result<bool, String> {
        let geometry_info: Vec<Proposition> = context.get_ref_expconfig().geometry_info.clone();
        for prop in geometry_info.iter() {
            if !self._eval_prop(prop, context)? {
                println!("Geometry info not satisfied: {}", prop);
                return Ok(false);
            }
            println!("Geometry info satisfied: {}", prop);
        }
        Ok(true)
    }
    pub fn _raw_definition_prop(&self, prop: &Proposition, context: Option<&ExpStructure>) -> Proposition {
        match prop {
            Proposition::Conserved { concept } => {
                let concept = self._raw_definition_concept(concept, None);
                Proposition::Conserved { concept: Box::new(concept) }
            }
            Proposition::Zero { concept } => {
                let concept = self._raw_definition_concept(concept, None);
                Proposition::Zero { concept: Box::new(concept) }
            }
            Proposition::IsConserved { exp } => {
                let exp = self._raw_definition_exp(exp, context).doit();
                Proposition::IsConserved { exp: Box::new(exp) }
            }
            Proposition::IsZero { exp } => {
                let exp = self._raw_definition_exp(exp, context).doit();
                Proposition::IsZero { exp: Box::new(exp) }
            }
            Proposition::Eq { left, right } => {
                let left = self._raw_definition_exp(left, context).doit();
                let right = self._raw_definition_exp(right, context).doit();
                Proposition::Eq { left: Box::new(left), right: Box::new(right) }
            }
            Proposition::Not { prop } => {
                let prop = self._raw_definition_prop(prop, context);
                Proposition::Not { prop: Box::new(prop) }
            }
        }
    }
    #[inline]
    pub fn _raw_definition_exp(&self, exp: &Exp, context: Option<&ExpStructure>) -> Exp {
        _raw_definition_exp(exp, context, &self.concepts)
    }
    #[inline]
    pub fn _raw_definition_concept(&self, concept: &Concept, context: Option<&ExpStructure>) -> Concept {
        match concept {
            Concept::Mk0 { exp } => {
                Concept::Mk0 { exp: Box::new(self._raw_definition_exp(exp, context).doit()) }
            }
            Concept::Mksucc { objtype, concept, id } => {
                Concept::Mksucc { objtype: objtype.clone(), concept: Box::new(self._raw_definition_concept(concept, context)), id: *id }
            }
            Concept::Mksum { objtype, concept } => {
                Concept::Mksum { objtype: objtype.clone(), concept: Box::new(self._raw_definition_concept(concept, context)) }
            }
        }
    }
    #[inline]
    pub fn _raw_definition_intrinsic(&self, intrinsic: &Intrinsic) -> Intrinsic {
        match intrinsic {
            Intrinsic::From { sexp } => {
                let sexp = sexp.as_ref();
                match sexp {
                    SExp::Mk { expconfig, exp } => {
                        let expconfig = expconfig.as_ref();
                        let context = self.experiments.get(&expconfig.get_expname());
                        let exp = self._raw_definition_exp(exp, context);
                        Intrinsic::From {
                            sexp: Box::new(SExp::Mk {
                                expconfig: Box::new(expconfig.clone()),
                                exp: Box::new(exp) 
                            })
                        }
                    }
                }
            }
        }
    }
    #[inline]
    pub fn _has_data_in_exp(&self, exp: &Exp, context: &ExpStructure) -> bool {
        context._has_data_in_exp(exp, &self.concepts)
    }
    #[inline]
    pub fn _specialize(&self, concept: &Concept, context: &ExpStructure) -> Vec<Exp> {
        context._specialize(concept, &self.concepts)
    }
    #[inline]
    pub fn _specialize_concept(&self, concept_name: String, context: &ExpStructure) -> Vec<AtomExp> {
        context._specialize_concept(concept_name, &self.concepts)
    }
    // 一个表达式它只由 Intrinsic（内禀概念） 和 Number 构成，可以用于判断它显然是守恒的。
    pub fn _made_of_obj_attr(&self, exp: &Exp) -> Result<bool, String> {
        match exp {
            Exp::Number { num: _ } => Ok(true),
            Exp::Atom { atom } => {
                let atom = atom.as_ref();
                if let Some(expr) = self.concepts.get(&atom.get_name()) {
                    match expr {
                        Expression::Intrinsic { intrinsic: _ } => Ok(true),
                        Expression::Concept { concept } => {
                            if concept.atomexp_name() == Some(atom.get_name()) {
                                return Ok(false); // 基本概念返回 false
                            }
                            let concept_new = concept.subst(atom.get_vec_ids())?;
                            self._made_of_obj_attr(&concept_new)
                        }
                        _ => unimplemented!()
                    }
                } else {
                    // 没在概念集合里出现过的 atom exp，默认为一个用户自定义的守恒符号。
                    Ok(true)
                }
            }
            Exp::UnaryExp { op:_, exp } => self._made_of_obj_attr(&*exp),
            Exp::BinaryExp { left, op:_, right } => {
                Ok(self._made_of_obj_attr(&*left)? && self._made_of_obj_attr(&*right)?)
            }
            Exp::DiffExp { left, right, ord:_ } => {
                Ok(self._made_of_obj_attr(&*left)? && self._made_of_obj_attr(&*right)?)
            }
            Exp::ExpWithMeasureType { exp, measuretype: _ } => {
                self._made_of_obj_attr(&*exp)
            }
            Exp::Partial { left:_, right:_ } => {
                unimplemented!()
            }
        }
    }
    #[inline]
    pub fn _concept_apply_ids(&self, concept: &Concept, ids: Vec<i32>, context: Option<&ExpStructure>) -> Result<Exp, String> {
        _concept_apply_ids(concept, ids, context, &self.concepts)
    }
}
