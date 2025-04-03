use crate::r;
/// This file contains the conversion functions between different types of data.
use ndarray::Array2;

use super::{ConstData, ExpData, NormalData, ZeroData};

impl From<ZeroData> for ExpData {
    fn from(content: ZeroData) -> Self {
        match content.err {
            f64::INFINITY => ExpData::Err {},
            _ => ExpData::Zero { content },
        }
    }
}

impl From<ConstData> for ExpData {
    fn from(content: ConstData) -> Self {
        match content {
            ConstData::Data { mean, std } => {
                if mean == f64::INFINITY || std == f64::INFINITY {
                    ExpData::Err {}
                } else if std > mean.abs() * 1.61344 {
                    // 等于 0 的置信度非常高
                    ExpData::Zero {
                        content: content.to_zero_data(),
                    }
                }
                // else
                // if std > mean.abs() / 10.0 {
                //     // 处于 0 和 const 之间的模糊地带的数据，不予考虑
                //     ExpData::Err { }
                // }
                else {
                    ExpData::Const { content }
                }
            }
            ConstData::Exact { value } => {
                if value == 0 {
                    ExpData::Zero {
                        content: ZeroData::new(0.),
                    }
                } else {
                    ExpData::Const { content }
                }
            }
        }
    }
}

impl From<i32> for ExpData {
    fn from(value: i32) -> Self {
        ExpData::Const {
            content: ConstData::Exact { value },
        }
    }
}

impl From<f64> for ExpData {
    fn from(value: f64) -> Self {
        match value {
            f64::INFINITY => ExpData::Err {},
            value => ExpData::Const {
                content: ConstData::Data {
                    mean: value,
                    std: 0.,
                },
            },
        }
    }
}

impl From<NormalData> for ExpData {
    fn from(content: NormalData) -> Self {
        if content.badpts.len() > content.n / 4 {
            ExpData::Err {}
        } else if content.is_zero() {
            ExpData::Zero {
                content: content.to_zero_data(),
            }
        } else if content.is_conserved() {
            ExpData::Const {
                content: content.to_const_data(),
            }
        } else {
            ExpData::Normal { content }
        }
    }
}

impl From<Array2<f64>> for ExpData {
    fn from(arr: Array2<f64>) -> Self {
        NormalData::new(arr).into()
    }
}

impl From<Array2<f64>> for NormalData {
    fn from(arr: Array2<f64>) -> Self {
        NormalData::new(arr)
    }
}

impl From<i32> for ConstData {
    fn from(value: i32) -> Self {
        ConstData::Exact { value }
    }
}

impl From<f64> for ConstData {
    fn from(value: f64) -> Self {
        ConstData::Data {
            mean: value,
            std: 0.,
        }
    }
}

impl TryFrom<ExpData> for ZeroData {
    type Error = String;

    fn try_from(data: ExpData) -> Result<Self, Self::Error> {
        match data {
            ExpData::Zero { content } => Ok(content),
            _ => Err(r!("Not a ZeroData")),
        }
    }
}

impl TryFrom<ExpData> for ConstData {
    type Error = &'static str;

    fn try_from(data: ExpData) -> Result<Self, Self::Error> {
        match data {
            ExpData::Const { content } => Ok(content),
            _ => Err("Not a ConstData"),
        }
    }
}

impl TryFrom<ExpData> for NormalData {
    type Error = &'static str;

    fn try_from(data: ExpData) -> Result<Self, Self::Error> {
        match data {
            ExpData::Normal { content } => Ok(content),
            _ => Err("Not a NormalData"),
        }
    }
}
