use crate::r;
use crate::language::Concept;
use super::obj::{ObjType, DATA};

impl ObjType {
    pub fn clock() -> Self { ObjType::new("Clock") }
}

impl DATA {
    /// `time` is concept related to `Clock` object.
    pub fn time() -> Concept { DATA::data(vec![r!("Clock")], r!("t")) }
}
