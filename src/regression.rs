/// This file defines several regression algorithms.
/// They are used to find simple relations between the data.

use fraction::Ratio;
use pyo3::prelude::*;
use crate::r;
use crate::expdata::{ExpData, Diff};
use crate::experiments::DataStruct;
use crate::language::{AtomExp, BinaryOp, Exp};
use crate::semantic::evaluation::apply_binary_op;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;


// 对于给定的数据 f(t), ... ， 提取出所有形如 f(t), f'(t) 的守恒量
#[pyfunction]
#[pyo3(signature = (fn_list, debug=false))]
pub fn search_trivial_relations(fn_list: &DataStruct, debug: bool) -> Vec<(Exp, ExpData)> {
    let mut list: Vec<(Exp, ExpData)> = vec![];

    let tdata = if fn_list.has_t() { Some(fn_list.get_t()) } else { None };
    for (atom, value) in fn_list.iter() {
        if value.is_conserved() {
            list.push((Exp::from_atom(atom), value.clone()));
        } else 
        if atom.get_name() != r!("t") {
            if let Some(ref tdata) = tdata {
                let valuedt = value.diff(&tdata);
                if valuedt.is_conserved() {
                    let exp = Exp::from_atom(atom).__difft__(1);
                    list.push((exp, valuedt));
                }
            }
        }
    }
    for i in 0..list.len() {
        for j in 0..i {
            if list[i].1.is_const() && list[j].1.is_const() {
                let value = &list[i].1 - &list[j].1;
                if value.is_zero() {
                    let exp = &list[i].0 - &list[j].0;
                    list.push((exp, value));
                }
                let value = &list[i].1 + &list[j].1;
                if value.is_zero() {
                    let exp = &list[i].0 + &list[j].0;
                    list.push((exp, value));
                }
            }
        }
    }
    if debug {
        println!("Trivial relations: ");
        for (exp, _) in list.iter() {
            print!("{}, ", exp);
        }
    }
    list
}


/// 对于给定的数据 f(t), g(t), ... ，
/// 生成所有不超过二次的 （形如 f(t), f(t)g(t) ） 的非守恒单项式
fn gen_monomials(fn_list: &DataStruct, including_div: bool) -> Vec<(Exp, ExpData)> {
    let ref origin_list = fn_list.iter().collect::<Vec<_>>();
    let mut list: Vec<(Exp, ExpData)> = vec![];
    for (atom, value) in origin_list {
        if !value.is_conserved() {
            list.push((Exp::from_atom(*atom), (*value).clone()));
        }
    }
    for id1 in 0..origin_list.len() {
        for id2 in 0..(id1+1) {
            let (atom1, value1) = origin_list[id1];
            let (atom2, value2) = origin_list[id2];
            if value1.is_err() || value2.is_err() {
                continue;
            }
            if value1.is_conserved() && value2.is_conserved() {
                continue;
            }
            let value = value1 * value2;
            if value.is_normal() {
                let exp = Exp::from_atom(atom1) * Exp::from_atom(atom2);
                list.push((exp, value));
            }
            if including_div && id2 < id1 {
                let value = value1 / value2;
                if value.is_normal() {
                    let exp = Exp::from_atom(atom1) / Exp::from_atom(atom2);
                    list.push((exp, value));
                }
                let value = value2 / value1;
                if value.is_normal() {
                    let exp = Exp::from_atom(atom2) / Exp::from_atom(atom1);
                    list.push((exp, value));
                }
            }
        }
    }
    list
}


// 对于给定的数据 f(t), g(t), ... ，
// 提取出所有形如 h1(t) o h2(t), f1(t)g1(t) o h(t) 或 f1(t)/g1(t) o f2(t)/g2(t) 的守恒量
// 这里的 o 表示二元运算符， 包括加减乘除和求导
#[pyfunction]
#[pyo3(signature = (fn_list, debug=false, cpu_num=50))]
pub fn search_relations_ver2(fn_list: &DataStruct, debug: bool, cpu_num: usize) -> Vec<(Exp, ExpData)> {
    let list = gen_monomials(fn_list, true);
    if debug {
        println!("debug in search_relations_ver2, Monomials:");
        for (exp, _) in list.iter() {
            print!("{}, ", exp);
        }
    }
    let tdata = if fn_list.has_t() { Some(fn_list.get_t()) } else { None };
    let mut res = search_relations_aux(&list, tdata, true, cpu_num);
    res.extend(search_trivial_relations(fn_list, debug));
    res
}


fn search_relations_ver3_closed(id1: usize, origin_list: &Vec<(&AtomExp, &ExpData)>, list: &Vec<(Exp, ExpData)>) -> Vec<(Exp, ExpData)> {
    let mut result = vec![];
    for id2 in 0..origin_list.len() {
        let (atom1, value1) = origin_list[id1];
        let (atom2, value2) = origin_list[id2];
        if value1.is_err() || value2.is_err() {
            continue;
        }
        if value1.is_conserved() && value2.is_conserved() {
            continue;
        }
        for op0 in vec![BinaryOp::Mul, BinaryOp::Div] {
            let value = apply_binary_op(&op0, value2.powi(2), value1.clone());
            if value.is_normal() {
                let exp = apply_binary_op(&op0, Exp::from_atom(atom2).pow(2), Exp::from_atom(atom1).clone());
                for (exp0, value0) in list.iter() {
                    for op in vec![BinaryOp::Add, BinaryOp::Sub] {
                        let valuenew = apply_binary_op(&op, value0.clone(), value.clone());
                        if valuenew.is_conserved() {
                            let expnew = apply_binary_op(&op, exp0.clone(), exp.clone());
                            result.push((expnew, valuenew));
                        }
                    }
                }
            }
        }
    }
    result
}

// 对于给定的数据 f(t), g(t), ... ，
// 提取出所有形如
// h1(t) o h2(t), f1(t)g1(t) o h(t), f1(t)g1(t) o f2(t)g2(t),
// f1(t)g1(t)^2 +/- f2(t)g2(t)^2 的守恒量
// 这里的 o 表示二元运算符， 包括加减乘除和求导
#[pyfunction]
#[pyo3(signature = (fn_list, debug=false, cpu_num=50))]
pub fn search_relations_ver3(fn_list: &DataStruct, debug: bool, cpu_num: usize) -> Vec<(Exp, ExpData)> {
    let ref origin_list = fn_list.iter().collect::<Vec<_>>();
    let mut list = gen_monomials(fn_list, false);
    let tdata = if fn_list.has_t() { Some(fn_list.get_t()) } else { None };
    let mut result = search_relations_aux(&list, tdata, true, cpu_num);
    if debug {
        println!("debug in search_relations_ver3");
    }
    let range = 0..origin_list.len();
    let indexes: Vec<usize> = range.collect();
    for id1 in 0..origin_list.len() {
        for id2 in 0..origin_list.len() {
            let (atom1, value1) = origin_list[id1];
            let (atom2, value2) = origin_list[id2];
            if value1.is_err() || value2.is_err() {
                continue;
            }
            if value1.is_conserved() && value2.is_conserved() {
                continue;
            }
            for op0 in vec![BinaryOp::Mul, BinaryOp::Div] {
                let value = apply_binary_op(&op0, value2.powi(2), value1.clone());
                if value.is_normal() {
                    let exp = apply_binary_op(&op0, Exp::from_atom(atom2).pow(2), Exp::from_atom(atom1).clone());
                    list.push((exp, value));
                }
            }
        }
    }

    let pool = ThreadPoolBuilder::new().num_threads(cpu_num).build().unwrap();
    let mut result_list: Vec<Vec<(Exp, ExpData)>> = Vec::new();
    
    pool.install(|| {
        result_list = indexes.par_iter().map(|&x| search_relations_ver3_closed(x, origin_list, &list)).collect();
    });

    let res: Vec<(Exp, ExpData)> = result_list.par_iter().flatten().cloned().collect();
    result.extend(res);
    result.extend(search_trivial_relations(fn_list, debug));
    if debug {
        println!("Monomials: ");
        let mut id = 0;
        for (exp, _) in list.iter() {
            print!("{}, ", exp);
            id = id + 1;
            if id % 5 == 0 {
                println!("")
            }
        }
        println!("Result: ");
        id = 0;
        for (exp, _) in result.iter() {
            print!("{},", exp);
            if id % 5 == 0 {
                println!("")
            }
        }
    }
    result
}


// 对于给定的数据 f(t), g(t), ... ，
// 提取出所有形如 f(t) o g(t) 的守恒量（这里的 f(t) 和 g(t) 被要求是非守恒的）
// o 不包括求导
#[pyfunction]
#[pyo3(signature = (fn_list, debug=false, cpu_num=50))]
pub fn search_binary_relations(fn_list: &DataStruct, debug: bool, cpu_num: usize) -> Vec<(Exp, ExpData)> {
    let mut list: Vec<(Exp, ExpData)> = vec![];
    if debug {
        println!("debug in search_binary_relations");
    }
    for (atom, value) in fn_list.iter() {
        if !value.is_conserved() {
            list.push((Exp::from_atom(atom), value.clone()));
        }
    }
    let tdata = if fn_list.has_t() { Some(fn_list.get_t()) } else { None };
    search_relations_aux(&list, tdata, false, cpu_num)
}


// 对于给定的数据 f(t), g(t), ... ，
// 提取出所有形如 f(t) o g(t) 的守恒量（这里的 f(t) 和 g(t) 被要求是非守恒的）
// 包括求导 D[f(t)]/D[g(t)] 形式的非平凡守恒量 （ D[f(t)]/D[t] != const )
#[pyfunction]
#[pyo3(signature = (fn_list, debug=false, cpu_num=50))]
pub fn search_relations_ver1(fn_list: &DataStruct, debug: bool, cpu_num: usize) -> Vec<(Exp, ExpData)> {
    let mut list: Vec<(Exp, ExpData)> = vec![];
    for (atom, value) in fn_list.iter() {
        if !value.is_conserved() {
            list.push((Exp::from_atom(atom), value.clone()));
        }
    }
    if debug {
        println!("debug in search_relations");
        for (exp, _) in list.iter() {
            print!("{}, ", exp);
        }
    }
    let tdata = if fn_list.has_t() { Some(fn_list.get_t()) } else { None };
    let mut res = search_relations_aux(&list, tdata, true, cpu_num);
    res.extend(search_trivial_relations(fn_list, debug));
    res
}


fn search_relations_aux0(i: usize, list: &Vec<(Exp, ExpData)>, tdata: Option<ExpData>, with_diff: bool) -> Vec<(Exp, ExpData)> {
    let mut relation_list = vec![];
    for j in 0..list.len() {
        if i == j {
            continue;
        }
        let (ref id, ref valuei) = list[i];
        let (ref jd, ref valuej) = list[j];
        if valuei.is_err() || valuej.is_err() {
            continue;
        }
        if j < i {
            let mut flag = false;
            for op in vec![BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
                let value = apply_binary_op(&op, valuei.clone(), valuej.clone());
                if value.is_conserved() {
                    let exp = apply_binary_op(&op, id.clone(), jd.clone());
                    relation_list.push((exp, value));
                    flag = true;
                    break;
                }
            }
            if flag {
                continue;
            }
            // check linear relation
            let valueii = valuei.diff_tau();
            let valuejj = valuej.diff_tau();
            if valueii.is_conserved() || valuejj.is_conserved() {
                continue;
            }
            let value_div = &valueii / &valuejj;
            if value_div.is_const() {
                let constdata = value_div.unwrap_const_data();
                let mean = constdata.mean();
                // estimate a f64 with rational number under precision
                if let Some((numer, denom)) = approximate_float(mean, 0.1, 10) {
                    // println!("calculate relation between {} and {}, expect ratio is {} / {}", id, jd, numer, denom);
                    let value_res = valuei * &ExpData::from(denom) - valuej * &ExpData::from(numer);
                    if value_res.is_conserved() {
                        let exp = id * &Exp::from_i32(denom) - jd * &Exp::from_i32(numer);
                        // println!("relation is {}", exp);
                        relation_list.push((exp, value_res));
                        continue;
                    }
                    if let Some(tdata) = &tdata {
                        let difft: ExpData = tdata.diff_tau();
                        let value_res = (&valueii / &difft) * denom.into()
                            - (&valuejj / &difft) * numer.into();
                        if value_res.is_conserved() {
                            let exp = id.__difft__(1) * Exp::from_i32(denom) - jd.__difft__(1) * Exp::from_i32(numer);
                            // println!("relation is {}", exp);
                            relation_list.push((exp, value_res));
                            continue;
                        }
                    }
                }
                if id.get_type() != r!("Div") && jd.get_type() != r!("Div") {
                    let value = valuei / valuej;
                    if value.is_conserved() {
                        let exp = id / jd;
                        relation_list.push((exp, value));
                    }
                    else if with_diff {
                        let exp = id.__diff__(jd);
                        relation_list.push((exp, value_div));
                    }
                }
            }
        }
    }
    relation_list
}

fn search_relations_aux(list: &Vec<(Exp, ExpData)>, tdata: Option<ExpData>, with_diff: bool, cpu_num: usize) -> Vec<(Exp, ExpData)> {
    // all data in list must be non-conserved!
    for (_, expdata) in list.iter() {
        if expdata.is_conserved() {
            panic!("Error: data in list must be non-conserved!");
        }
    }
    let range = 0..list.len();
    let indexes: Vec<usize> = range.collect();

    let pool = ThreadPoolBuilder::new().num_threads(cpu_num).build().unwrap();
    let mut relation_list: Vec<Vec<(Exp, ExpData)>> = Vec::new();

    pool.install(|| {
        relation_list = indexes.par_iter().map(|&x| search_relations_aux0(x, list, tdata.clone(), with_diff)).collect();
    });

    // 拼接好多好多个 vec
    let res  = relation_list.par_iter().flatten().cloned().collect();
    res
}

use num_integer::Integer;
use num_traits::float::FloatCore;
use num_traits::{Bounded, NumCast, Pow};

fn approximate_float(val: f64, max_error: f64, max_iterations: usize) -> Option<(i32, i32)> {
    let negative = val.is_sign_negative();
    let abs_val = val.abs();
    let r: Ratio<i32> = approximate_float_unsigned(abs_val, max_error, max_iterations)?;
    Some(if negative { (-*r.numer(), *r.denom()) } else { (*r.numer(), *r.denom()) })
}
fn approximate_float_unsigned<T, F>(val: F, max_error: F, max_iterations: usize) -> Option<Ratio<T>>
where
    T: Integer + Bounded + NumCast + Clone,
    F: FloatCore + NumCast,
{
    // Continued fractions algorithm
    // https://web.archive.org/web/20200629111319/http://mathforum.org:80/dr.math/faq/faq.fractions.html#decfrac

    if val < F::zero() || val.is_nan() {
        return None;
    }

    let mut q = val;
    let mut n0 = T::zero();
    let mut d0 = T::one();
    let mut n1 = T::one();
    let mut d1 = T::zero();

    let t_max = T::max_value();
    let t_max_f = <F as NumCast>::from(t_max.clone())?;

    // 1/epsilon > T::MAX
    let epsilon = t_max_f.recip();

    // Overflow
    if q > t_max_f {
        return None;
    }

    for _ in 0..max_iterations {
        let a = match <T as NumCast>::from(q) {
            None => break,
            Some(a) => a,
        };

        let a_f = match <F as NumCast>::from(a.clone()) {
            None => break,
            Some(a_f) => a_f,
        };
        let f = q - a_f;

        // Prevent overflow
        if !a.is_zero()
            && (n1 > t_max.clone() / a.clone()
                || d1 > t_max.clone() / a.clone()
                || a.clone() * n1.clone() > t_max.clone() - n0.clone()
                || a.clone() * d1.clone() > t_max.clone() - d0.clone())
        {
            break;
        }

        let n = a.clone() * n1.clone() + n0.clone();
        let d = a.clone() * d1.clone() + d0.clone();

        n0 = n1;
        d0 = d1;
        n1 = n.clone();
        d1 = d.clone();

        // Simplify fraction. Doing so here instead of at the end
        // allows us to get closer to the target value without overflows
        let g = Integer::gcd(&n1, &d1);
        if !g.is_zero() {
            n1 = n1 / g.clone();
            d1 = d1 / g.clone();
        }

        // Close enough?
        let (n_f, d_f) = match (<F as NumCast>::from(n), <F as NumCast>::from(d)) {
            (Some(n_f), Some(d_f)) => (n_f, d_f),
            _ => break,
        };
        if (n_f / d_f - val).abs() < max_error {
            break;
        }

        // Prevent division by ~0
        if f < epsilon {
            break;
        }
        q = f.recip();
    }

    // Overflow
    if d1.is_zero() {
        return None;
    }

    Some(Ratio::new(n1, d1))
}
