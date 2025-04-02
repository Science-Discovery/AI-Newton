use std::ops::{Add, Div, Mul, Sub};

use num_traits::Pow;
use pyo3::prelude::*;


/// ZeroData is a struct that represents a value of zero with uncertainty.
/// The uncertainty is estimated by `err`.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ZeroData {
    pub err: f64,
}

impl ZeroData {
    pub fn new(err: f64) -> Self {
        Self { err }
    }
}

impl std::fmt::Display for ZeroData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ExpData.ZeroData: err = {}", self.err)
    }
}

impl Add for ZeroData {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            err: (self.err.powi(2) + other.err.powi(2)).sqrt(),
        }
    }
}

impl Add for &ZeroData {
    type Output = ZeroData;
    fn add(self, other: Self) -> ZeroData {
        ZeroData {
            err: (self.err.powi(2) + other.err.powi(2)).sqrt()
        }
    }
}

impl Sub for ZeroData {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            err: (self.err.powi(2) - other.err.powi(2)).sqrt(),
        }
    }
}

impl Sub for &ZeroData {
    type Output = ZeroData;

    fn sub(self, other: &ZeroData) -> ZeroData {
        ZeroData {
            err: (self.err.powi(2) - other.err.powi(2)).sqrt(),
        }
    }
}

impl Mul<f64> for ZeroData {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            err: self.err * other,
        }
    }
}

impl Mul<f64> for &ZeroData {
    type Output = ZeroData;

    fn mul(self, other: f64) -> ZeroData {
        ZeroData {
            err: self.err * other,
        }
    }
}

impl Div<f64> for ZeroData {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            err: self.err / other,
        }
    }
}

impl Div<f64> for &ZeroData {
    type Output = ZeroData;

    fn div(self, other: f64) -> ZeroData {
        assert_ne!(other, 0.0);
        ZeroData {
            err: self.err / other,
        }
    }
}

impl Mul<ZeroData> for ZeroData {
    type Output = ZeroData;

    fn mul(self, other: ZeroData) -> ZeroData {
        ZeroData {
            err: self.err * other.err,
        }
    }
}

impl Mul<&ZeroData> for &ZeroData {
    type Output = ZeroData;

    fn mul(self, other: &ZeroData) -> ZeroData {
        ZeroData {
            err: self.err * other.err,
        }
    }
}

impl Pow<i32> for &ZeroData {
    type Output = ZeroData;

    fn pow(self, exp: i32) -> ZeroData {
        ZeroData {
            err: self.err.powi(exp),
        }
    }
}
