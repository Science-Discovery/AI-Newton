/// This module defines the basic components of the object language.
///
/// `DATA`: Represent a basic concept attached to an object (or a list of objects).
///    the concept that applies to specific objects in specific experiment can be measured by experimentalists.
///
/// `ATTR`: Represent a basic attribute attached to an object (or a list of objects).
///    the attributes has no special meaning in physics, but can be used to describe the freedom of the object.
///    it can be accessed by the experimentalists, but not exposed to the theorist.
///    In a specific experiment setting, the attributes can be freely adjusted within a pre-defined range.
///
/// `ObjType`: Represent a type of object.
use pyo3::prelude::*;
use std::fmt;

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ObjType {
    pub obj: String,
}
impl ObjType {
    pub fn new(obj: &str) -> ObjType {
        ObjType {
            obj: obj.to_string(),
        }
    }
}

impl fmt::Display for ObjType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.obj)
    }
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum DATA {
    Mk { obj: ObjType, name: String },
}
impl DATA {
    pub fn new(obj: ObjType, name: &str) -> DATA {
        DATA::Mk {
            obj,
            name: name.to_string(),
        }
    }
    pub fn name(&self) -> &String {
        match self {
            DATA::Mk { name, .. } => name,
        }
    }
    pub fn obj(&self) -> &ObjType {
        match self {
            DATA::Mk { obj, .. } => obj,
        }
    }
}
impl fmt::Display for DATA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DATA::Mk { obj: _, name } => write!(f, "{}", name),
        }
    }
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ATTR {
    Mk { obj: ObjType, name: String },
}
impl ATTR {
    pub fn new(obj: ObjType, name: &str) -> ATTR {
        ATTR::Mk {
            obj,
            name: name.to_string(),
        }
    }
    pub fn name(&self) -> &String {
        match self {
            ATTR::Mk { name, .. } => name,
        }
    }
    pub fn obj(&self) -> &ObjType {
        match self {
            ATTR::Mk { obj, .. } => obj,
        }
    }
}
impl fmt::Display for ATTR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ATTR::Mk { name, .. } => write!(f, "{}", name),
        }
    }
}

use crate::language::{Concept, Exp};
impl DATA {
    pub fn data_global(name: String) -> Concept {
        Concept::Mk0 {
            exp: Box::new(Exp::new_variable(name)),
        }
    }
    pub fn data(obj_types: Vec<String>, name: String) -> Concept {
        let n = obj_types.len();
        let atom = Exp::new_variable_ids(name, (1..(n + 1)).map(|x| x as i32).collect());
        let mut concept = Concept::Mk0 {
            exp: Box::new(atom),
        };
        for i in 0..n {
            concept = Concept::Mksucc {
                objtype: obj_types[i].clone(),
                concept: Box::new(concept),
                id: i as i32 + 1,
            }
        }
        concept
    }
}
