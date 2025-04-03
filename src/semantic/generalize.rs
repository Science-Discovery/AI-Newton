/// This file defines the generalization functions that transform an Exp object into a Concept object.
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use crate::experiments::{ExpStructure, ObjType};
use crate::knowledge::Knowledge;
use crate::language::{Concept, Exp};

impl Knowledge {
    pub fn _generalize(&self, expr: &Exp, context: &ExpStructure) -> Option<Concept> {
        match self._generalize_to_mksum(expr, context) {
            Some(concept) => Some(concept),
            None => self._generalize_to_normal_concept(expr, context),
        }
    }
    pub fn _extract_concepts(&self, expr: &Exp, context: &ExpStructure) -> HashSet<Concept> {
        let mut res = HashSet::new();
        let ordered_terms = expr.as_ordered_terms();
        match self._generalize_to_mksum(expr, context) {
            Some(concept) => {
                res.insert(Some(concept));
                res.insert(
                    self._generalize_to_normal_concept(&ordered_terms[0].remove_coeff(), context),
                );
            }
            None => {
                res.insert(self._generalize_to_normal_concept(&expr.remove_coeff(), context));
                // if ordered_terms.len() > 1 {
                //     for term in ordered_terms.iter() {
                //         res.insert(self._generalize_to_normal_concept(&term.remove_coeff(), context));
                //     }
                // }
                // match expr {
                //     Exp::DiffExp { left, right, ord: _ } => {
                //         if format!("{}", right) != r!("t[0]") {
                //             res.insert(self._generalize_to_normal_concept(&left.__difft__(1), context));
                //             res.insert(self._generalize_to_normal_concept(&right.__difft__(1), context));
                //         }
                //     }
                //     _ => {}
                // }
            }
        }
        res.into_iter().filter_map(|x| x).collect()
    }
    pub fn _generalize_to_normal_concept(
        &self,
        expr: &Exp,
        context: &ExpStructure,
    ) -> Option<Concept> {
        let vec = expr.get_allids_not_t();
        let n = vec.len();
        if n == 0 {
            return None;
        }
        let perm = (1..(n + 1)).permutations(n);
        let mut nexp = expr.clone();
        let mut nexp_subs_dict: HashMap<i32, i32> = HashMap::new();
        for p in perm {
            let mut subst_dict: HashMap<i32, i32> = HashMap::new();
            for (i, j) in vec.iter().zip(p) {
                subst_dict.insert(*i, j as i32);
            }
            let new_exp = expr.substs(&subst_dict);
            if nexp_subs_dict.is_empty() || format!("{}", new_exp) < format!("{}", nexp) {
                nexp = new_exp;
                nexp_subs_dict = subst_dict;
            }
        }
        // println!("nexp = {}", nexp);
        // println!("nexp_subs_dict = {:?}", nexp_subs_dict);
        let mut id_objtype_map: HashMap<i32, String> = HashMap::new();
        for (i, j) in nexp_subs_dict.iter() {
            let obj = context.get_obj(*i);
            id_objtype_map.insert(*j, obj.obj_type.to_string());
        }
        let mut concept_res = Concept::Mk0 {
            exp: Box::new(nexp),
        };
        for i in 1..(n + 1) {
            let objtype = id_objtype_map.get(&(i as i32)).unwrap();
            // println!("--({}->{}), ", i, objtype);
            concept_res = Concept::Mksucc {
                objtype: objtype.clone(),
                concept: Box::new(concept_res),
                id: i as i32,
            };
        }
        Some(concept_res)
    }
    pub fn _generalize_to_mksum(&self, expr: &Exp, context: &ExpStructure) -> Option<Concept> {
        let ordered_terms = expr.as_ordered_terms();
        if ordered_terms.len() > 1 {
            let concept0 = self._generalize_to_normal_concept(&ordered_terms[0], context);
            if concept0.is_none() {
                return None;
            }
            let concept0 = concept0.unwrap();
            let preids = concept0.get_preids();
            if preids.len() != 1 {
                return None;
            }
            let id = *ordered_terms[0].get_allids_not_t().iter().next().unwrap();
            let objtype = context.get_obj(id).obj_type.obj;
            let mut ids = HashSet::from([id]);
            for i in 1..ordered_terms.len() {
                let concepti = self._generalize_to_normal_concept(&ordered_terms[i], context);
                if concepti.is_none_or(|x| x != concept0) {
                    return None;
                }
                let id = *ordered_terms[i].get_allids_not_t().iter().next().unwrap();
                if ids.contains(&id) {
                    return None;
                }
                ids.insert(id);
            }
            if ids != context.get_obj_ids(ObjType::new(&objtype)) {
                return None;
            }
            Some(Concept::Mksum {
                objtype,
                concept: Box::new(concept0.clone()),
            })
        } else {
            None
        }
    }
}
