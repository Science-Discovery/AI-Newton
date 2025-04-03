use crate::knowledge::Knowledge;
/// This file defines the display trait for the language.
/// The display trait is used to convert an expression to a string,
/// which is used to display the expression in the console, or write to a file.
/// The string representation of an expression contains all the information of the expression,
/// so that the expression object can be restored from the string.
use itertools::Itertools;

use crate::language::{
    AtomExp, Concept, Exp, Expression, IExpConfig, Intrinsic, MeasureType, Proposition, SExp,
    UnaryOp,
};
use std::fmt::{self, Display, Formatter, Result};

impl Display for AtomExp {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            AtomExp::Variable { name } => write!(f, "{}", name),
            AtomExp::VariableIds { name, ids } => {
                if ids.len() == 0 {
                    write!(f, "{}", name)
                } else {
                    let str_list = ids.iter().map(|x| x.to_string()).collect::<Vec<String>>();
                    write!(f, "{}[{}]", name, str_list.join(", "))
                }
            }
        }
    }
}
impl Display for Exp {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Exp::Number { num } => write!(f, "{}", num),
            Exp::Atom { atom } => write!(f, "{}", atom),
            Exp::UnaryExp { op, exp } => match op {
                UnaryOp::Neg => write!(f, "(-{})", exp),
                UnaryOp::Diff => write!(f, "D.{}", exp), // Do not recommend to use this, use DiffExp instead
            },
            Exp::BinaryExp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Exp::DiffExp { left, right, ord } => match ord {
                1 => write!(f, "D[{}]/D[{}]", left, right),
                _ => write!(f, "D^{}[{}]/D[{}]^{}", ord, left, right, ord),
            },
            Exp::ExpWithMeasureType { exp, measuretype } => {
                write!(f, "{} with {}", exp, measuretype)
            }
            Exp::Partial { left, right } => write!(f, "Partial[{}]/Partial[{}]", left, right),
        }
    }
}
impl Display for SExp {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            SExp::Mk { expconfig, exp } => write!(f, "{} |- {}", expconfig, exp),
        }
    }
}

impl Concept {
    fn _aux_print(&self, f: &mut Formatter) -> Result {
        match self {
            Concept::Mk0 { exp: _ } => Ok(()),
            Concept::Mksucc {
                objtype,
                concept,
                id,
            } => {
                concept._aux_print(f)?;
                write!(f, "({}->{}) ", id, objtype)
            }
            Concept::Mksum { objtype, concept } => {
                write!(f, "[Sum:{}] ", objtype)?;
                concept._aux_print(f)
            }
        }
    }
}
impl Display for Concept {
    fn fmt(&self, f: &mut Formatter) -> Result {
        self._aux_print(f)?;
        write!(f, "|- {}", self.get_exp())
    }
}
impl Display for IExpConfig {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            IExpConfig::From { name } => write!(f, "#{}", name),
            IExpConfig::Mk {
                objtype,
                expconfig,
                id,
            } => write!(f, "{} ({}->{})", expconfig, id, objtype),
            IExpConfig::Mkfix {
                object,
                expconfig,
                id,
            } => write!(f, "{} [{}->{}]", expconfig, id, object),
        }
    }
}
impl Display for Intrinsic {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Intrinsic::From { sexp } => write!(f, "[{}]", sexp),
        }
    }
}
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Expression::Exp { exp } => write!(f, "{}", exp),
            Expression::SExp { sexp } => write!(f, "{}", sexp),
            Expression::Concept { concept } => write!(f, "{}", concept),
            Expression::Intrinsic { intrinsic } => write!(f, "{}", intrinsic),
            Expression::Proposition { prop } => write!(f, "{}", prop),
        }
    }
}
impl Display for MeasureType {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "[t_end={}, n={}, repeat_time={}, error={}]",
            self.t_end, self.n, self.repeat_time, self.error
        )
    }
}
impl Display for Proposition {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Proposition::Conserved { concept } => write!(f, "{} is conserved", concept),
            Proposition::Zero { concept } => write!(f, "{} is zero", concept),
            Proposition::IsConserved { exp } => write!(f, "{} is conserved", exp),
            Proposition::IsZero { exp } => write!(f, "{} is zero", exp),
            Proposition::Eq { left, right } => write!(f, "{} = {}", left, right),
            Proposition::Not { prop } => write!(f, "not ({})", prop),
        }
    }
}

impl fmt::Display for Knowledge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[Knowledge]\n")?;
        for (name, obj) in self.objects.iter() {
            write!(f, "{} := {}\n", name, obj)?;
        }
        let name_list = self.concepts.keys().cloned().collect::<Vec<String>>();
        let name_list = name_list.iter().sorted_by_key(|x|
            // 将 x_123 分割为 (x, 123)，取 123 为 key
            match x.split('_').last() {
                Some(x) => {
                    x.parse::<i32>().unwrap_or(0)
                },
                None => 0
            });
        for name in name_list {
            write!(f, "{} := {}\n", name, self.concepts.get(name).unwrap())?;
        }
        for (name, prop) in self.conclusions.iter() {
            write!(f, "{} := {}\n", name, prop)?;
        }
        write!(f, "[end]\n")?;
        let yaml_str = serde_yaml::to_string(&self.experiments).unwrap();
        write!(f, "[Experiments]\n")?;
        write!(f, "{}", yaml_str)?;
        write!(f, "[end]")?;
        Ok(())
    }
}
