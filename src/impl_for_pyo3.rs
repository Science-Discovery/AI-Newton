use crate::language::{
    AtomExp, Concept, Exp, IExpConfig, Intrinsic, MeasureType, Proposition, SExp,
};
use pyo3::callback::IntoPyCallbackOutput;
use pyo3::prelude::*;

impl FromPyObject<'_> for Box<Exp> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<Exp>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<Exp> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}
impl FromPyObject<'_> for Box<Proposition> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<Proposition>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<Proposition> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}
impl FromPyObject<'_> for Box<IExpConfig> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<IExpConfig>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<IExpConfig> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}
impl FromPyObject<'_> for Box<SExp> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<SExp>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<SExp> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}

impl FromPyObject<'_> for Box<Intrinsic> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<Intrinsic>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<Intrinsic> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}

impl FromPyObject<'_> for Box<MeasureType> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<MeasureType>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<MeasureType> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}
impl FromPyObject<'_> for Box<Concept> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<Concept>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}
impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<Concept> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}

impl FromPyObject<'_> for Box<AtomExp> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = ob.extract::<AtomExp>()?;
        // println!("Extracted: {}", x);
        Ok(Box::new(x))
    }
}

impl IntoPyCallbackOutput<*mut pyo3::ffi::PyObject> for Box<AtomExp> {
    #[inline]
    fn convert(self, py: Python<'_>) -> PyResult<*mut pyo3::ffi::PyObject> {
        Ok(self.into_py(py).as_ptr())
    }
}
