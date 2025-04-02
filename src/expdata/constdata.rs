use num_traits::Pow;
use pyo3::prelude::*;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

use super::ZeroData;


/// ConstData is a struct that represents a constant value (estimated by `mean`)
/// with uncertainty (estimated by `std`).
#[pyclass]
#[derive(Debug, Clone)]
pub enum ConstData {
    Data { mean: f64, std: f64 },
    Exact { value: i32 },
}

#[pymethods]
impl ConstData {
    #[inline]
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    #[inline]
    #[new]
    fn __new__(mean: f64, std: f64) -> PyResult<Self> {
        Ok(ConstData::Data { mean, std })
    }
    #[inline]
    #[staticmethod]
    fn exact(value: i32) -> PyResult<Self> {
        Ok(ConstData::Exact { value })
    }
    #[inline]
    #[getter]
    pub fn mean(&self) -> f64 {
        match self {
            ConstData::Data { mean, std: _ } => *mean,
            ConstData::Exact { value } => *value as f64,
        }
    }
    #[inline]
    #[getter]
    pub fn std(&self) -> f64 {
        match self {
            ConstData::Data { mean: _, std } => *std,
            ConstData::Exact { value: _ } => 0.0,
        }
    }
    #[inline]
    fn __add__(&self, rhs: &Self) -> Self {
        self + rhs
    }
    #[inline]
    fn __sub__(&self, rhs: &Self) -> Self {
        self - rhs
    }
    #[inline]
    fn __mul__(&self, rhs: &Self) -> Self {
        self * rhs
    }
    #[inline]
    fn __rmul__(&self, rhs: f64) -> Self {
        self.clone() * ConstData::new(rhs, 0.0)
    }
    #[inline]
    fn __truediv__(&self, rhs: &Self) -> Self {
        self / rhs
    }
    #[inline]
    fn __neg__(&self) -> Self {
        -self
    }
    #[inline]
    fn __powi__(&self, n: i32) -> Self {
        self.powi(n)
    }
}


impl ConstData {
    #[inline]
    pub fn to_zero_data(&self) -> ZeroData {
        ZeroData::new(fold_gaussian(self.mean(), self.std()))
    }
}


impl fmt::Display for ConstData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ExpData.ConstData =\n")?;
        match self {
            ConstData::Data { mean, std } => {
                write!(f, "mean = {}, std = {}", mean, std)
            }
            ConstData::Exact { value } => {
                write!(f, "value = {}", value)
            }
        }
    }
}

impl Add for ConstData {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match self {
            ConstData::Data { mean: m1, std: s1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: m1 + m2,
                    std: fold_gaussian(s1, s2),
                },
                ConstData::Exact { value: v2 } => ConstData::Data {
                    mean: m1 + v2 as f64,
                    std: s1,
                },
            },
            ConstData::Exact { value: v1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: v1 as f64 + m2,
                    std: s2,
                },
                ConstData::Exact { value: v2 } => ConstData::Exact {
                    value: v1 + v2,
                },
            },
        }
    }
}

impl Add for &ConstData {
    type Output = ConstData;

    fn add(self, other: Self) -> ConstData {
        self.clone() + other.clone()
    }
}

impl Add<&ZeroData> for &ConstData {
    type Output = ConstData;

    fn add(self, other: &ZeroData) -> ConstData {
        self + &ConstData::new(0., other.err)
    }
}

impl Sub for ConstData {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match self {
            ConstData::Data { mean: m1, std: s1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: m1 - m2,
                    std: fold_gaussian(s1, s2),
                },
                ConstData::Exact { value: v2 } => ConstData::Data {
                    mean: m1 - v2 as f64,
                    std: s1,
                },
            },
            ConstData::Exact { value: v1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: v1 as f64 - m2,
                    std: s2,
                },
                ConstData::Exact { value: v2 } => ConstData::Exact {
                    value: v1 - v2,
                },
            },
        }
    }
}

impl Sub for &ConstData {
    type Output = ConstData;

    fn sub(self, other: Self) -> ConstData {
        self.clone() - other.clone()
    }
}

impl Sub<&ZeroData> for &ConstData {
    type Output = ConstData;

    fn sub(self, other: &ZeroData) -> ConstData {
        self + &ConstData::new(0., other.err)
    }
}


impl Mul for ConstData {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match self {
            ConstData::Data { mean: m1, std: s1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: m1 * m2,
                    std: fold_gaussian(m1 * s2, m2 * s1),
                },
                ConstData::Exact { value: v2 } => ConstData::Data {
                    mean: m1 * v2 as f64,
                    std: s1 * v2 as f64,
                },
            },
            ConstData::Exact { value: v1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: v1 as f64 * m2,
                    std: s2 * v1 as f64,
                },
                ConstData::Exact { value: v2 } => ConstData::Exact {
                    value: v1 * v2,
                },
            },
        }
    }
}

impl Mul for &ConstData {
    type Output = ConstData;

    fn mul(self, other: Self) -> ConstData {
        self.clone() * other.clone()
    }
}

impl Div for ConstData {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match self {
            ConstData::Data { mean: m1, std: s1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: m1 / m2,
                    std: fold_gaussian(s1 / m2, s2 * m1 / m2.powi(2)),
                },
                ConstData::Exact { value: v2 } => ConstData::Data {
                    mean: m1 / v2 as f64,
                    std: s1 / v2 as f64,
                },
            },
            ConstData::Exact { value: v1 } => match other {
                ConstData::Data { mean: m2, std: s2 } => ConstData::Data {
                    mean: v1 as f64 / m2,
                    std: s2 * v1 as f64 / m2.powi(2),
                },
                ConstData::Exact { value: _ } => {
                    unimplemented!()
                    // ConstData::Exact {
                    //     // TODO: rational
                    //     value: v1 / v2,
                    // }
                },
            },
        }
    }
}

impl Div for &ConstData {
    type Output = ConstData;

    fn div(self, other: Self) -> ConstData {
        self.clone() / other.clone()
    }
}

impl Neg for ConstData {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            ConstData::Data { mean, std } => ConstData::Data {
                mean: -mean,
                std,
            },
            ConstData::Exact { value } => ConstData::Exact {
                value: -value,
            },
        }
    }
}

impl Neg for &ConstData {
    type Output = ConstData;

    fn neg(self) -> ConstData {
        self.clone().neg()
    }
}

impl Pow<i32> for &ConstData {
    type Output = ConstData;

    fn pow(self, n: i32) -> ConstData {
        self.powi(n)
    }
}


impl ConstData {
    pub fn new(mean: f64, std: f64) -> Self {
        ConstData::Data { mean, std }
    }
    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        match self {
            ConstData::Exact { value } => ConstData::Exact {
                value: value.pow(n as u32),
            },
            ConstData::Data { mean, std } => ConstData::Data {
                mean: mean.powi(n),
                std: n as f64 * mean.powi(n - 1) * std,
            },
        }
    }
}

// fold two gaussian distributions
#[inline]
fn fold_gaussian(a: f64, b: f64) -> f64 {
    (a.powi(2) + b.powi(2)).sqrt()
}
