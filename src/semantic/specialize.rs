/// This file defines the specialization functions that transform a Concept object into an Exp object, given a context.
/// The context contains the experiment configuration and the concepts that are already defined.
/// The specialization functions are used to generate the possible Exp object
/// that can be derived from a Concept object in certain experiment configurations.
/// For example,
/// `[Sum: Particle] (1->Particle) |- m[1] * v[1]` can be specialized into `m[1] * v[1] + m[2] * v[2]` in collision experiment.
/// `(1->Particle) |- posx[1]` can be specialized into `posx[1], posx[2]` in collision experiment.

use std::collections::HashMap;
use super::evaluation::_raw_definition_exp;

use crate::language::{AtomExp, BinaryOp, Concept, Exp, Expression};
use crate::experiments::ExpStructure;

impl ExpStructure {
    pub fn _specialize(&self, concept: &Concept, context: &HashMap<String, Expression>) -> Vec<Exp> {
        let objtype_id_map = concept.get_objtype_id_map();
        let vec_map = self.get_all_possible_map(&objtype_id_map);
        match concept {
            Concept::Mksum { objtype, concept } => {
                if vec_map.len() == 0 {
                    return vec![Exp::Number { num: 0 }];
                }
                let mut result: HashMap<Vec<(i32, i32)>, Exp> = HashMap::new();
                for dict in vec_map.iter() {
                    let new_exp = concept.substs(dict).unwrap();
                    let mut tmp: Vec<(i32, i32)> = vec![];
                    for (i_objtype, idset) in objtype_id_map.iter() {
                        if i_objtype != objtype {
                            for i in idset {
                                tmp.push((*i, *dict.get(i).unwrap()));
                            }
                        }
                    }
                    tmp.sort_by_key(|x| x.0);
                    let res = result.get_mut(&tmp);
                    if let Some(x) = res {
                        let w = x.clone();
                        *x = Exp::BinaryExp { left: Box::new(w), op: BinaryOp::Add, right: Box::new(new_exp) };
                    } else {
                        result.insert(tmp, new_exp);
                    }
                }
                let mut res: Vec<Exp> = vec![];
                for (_, exp) in result.iter() {
                    if self._has_data_in_exp(&exp, context) {
                        res.push(exp.clone());
                    }
                }
                res
            },
            _ => {
                let mut res: Vec<Exp> = vec![];
                for dict in vec_map.iter() {
                    let new_exp = concept.substs(dict).unwrap();
                    if self._has_data_in_exp(&new_exp, context) {
                        res.push(new_exp);
                    }
                }
                res
            }
        }
    }
    pub fn _specialize_concept(&self, concept_name: String, context: &HashMap<String, Expression>) -> Vec<AtomExp> {
        let res = context.get(&concept_name);
        if res.is_none() {
            return vec![];
        }
        let concept = res.unwrap();
        match concept {
            Expression::Intrinsic { intrinsic } => {
                let vec_map = self.get_all_possible_map(&intrinsic.get_objtype_id_map());
                let preids = intrinsic.get_preids();
                let mut exp_list = vec![];
                for dict in vec_map.iter() {
                    let mut ids = vec![];
                    for id in preids.iter() {
                        ids.push(*dict.get(id).unwrap());
                    }
                    exp_list.push(AtomExp::new_variable_ids(concept_name.clone(), ids));
                }
                exp_list
            }
            Expression::Concept { concept } => {
                match concept.as_ref() {
                    Concept::Mksum { objtype, concept: _ } => {
                        let mut objtype_id_map = concept.get_objtype_id_map();
                        objtype_id_map.remove(objtype);
                        let vec_map = self.get_all_possible_map(&objtype_id_map);
                        let preids = concept.get_preids();
                        let mut exp_list = vec![];
                        for dict in vec_map.iter() {
                            let mut ids = vec![];
                            for id in preids.iter() {
                                ids.push(*dict.get(id).unwrap());
                            }
                            let atom_exp = AtomExp::new_variable_ids(concept_name.clone(), ids);
                            if self._has_data_in_exp(&Exp::Atom{ atom: Box::new(atom_exp.clone()) }, context) {
                                exp_list.push(atom_exp);
                            }
                        }
                        exp_list
                    },
                    _ => {
                        let vec_map = self.get_all_possible_map(&concept.get_objtype_id_map());
                        let preids = concept.get_preids();
                        let mut exp_list = vec![];
                        for dict in vec_map.iter() {
                            let mut ids = vec![];
                            for id in preids.iter() {
                                ids.push(*dict.get(id).unwrap());
                            }
                            let atom_exp = AtomExp::new_variable_ids(concept_name.clone(), ids);
                            if self._has_data_in_exp(&Exp::Atom{ atom: Box::new(atom_exp.clone()) }, context) {
                                exp_list.push(atom_exp);
                            }
                        }
                        exp_list
                    }
                }
            }
            _ => unimplemented!()
        }
    }
    pub fn _has_data_in_exp(&self, exp: &Exp, context: &HashMap<String, Expression>) -> bool {
        let raw_exp = _raw_definition_exp(exp, Some(self), context);
        let atoms = raw_exp.get_all_atoms();
        for atom in atoms.iter() {
            if !self.has_data(atom) {
                match context.get(&atom.get_name()) {
                    None | Some(Expression::Concept { concept: _ }) => return false,
                    _ => {}
                }
            }
        }
        true
    }
}
