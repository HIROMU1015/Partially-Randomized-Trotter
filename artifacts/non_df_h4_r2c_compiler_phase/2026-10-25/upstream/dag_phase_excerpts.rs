// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
// Excerpts only; origin line numbers are in the section labels below.

// Original lines 817-832
    ///
    /// Args:
    ///     angle (float, :class:`.ParameterExpression`): The phase angle.
    #[setter]
    fn set_global_phase(&mut self, angle: Param) -> PyResult<()> {
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * PI));
            }
            Param::ParameterExpression(angle) => {
                self.global_phase = Param::ParameterExpression(angle);
            }
            Param::Obj(_) => return Err(PyTypeError::new_err("Invalid type for global phase")),
        }
        Ok(())
    }

// Original lines 6310-6320
    pub fn add_global_phase(&mut self, py: Python, value: &Param) -> PyResult<()> {
        match value {
            Param::Obj(_) => {
                return Err(PyTypeError::new_err(
                    "Invalid parameter type, only float and parameter expression are supported",
                ))
            }
            _ => self.set_global_phase(add_global_phase(py, &self.global_phase, value)?)?,
        }
        Ok(())
    }

// Original lines 7010-7032
/// Add to global phase. Global phase can only be Float or ParameterExpression so this
/// does not handle the full possibility of parameter values.
pub(crate) fn add_global_phase(py: Python, phase: &Param, other: &Param) -> PyResult<Param> {
    Ok(match [phase, other] {
        [Param::Float(a), Param::Float(b)] => Param::Float(a + b),
        [Param::Float(a), Param::ParameterExpression(b)] => Param::ParameterExpression(
            b.clone_ref(py)
                .call_method1(py, intern!(py, "__radd__"), (*a,))?,
        ),
        [Param::ParameterExpression(a), Param::Float(b)] => Param::ParameterExpression(
            a.clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (*b,))?,
        ),
        [Param::ParameterExpression(a), Param::ParameterExpression(b)] => {
            Param::ParameterExpression(a.clone_ref(py).call_method1(
                py,
                intern!(py, "__add__"),
                (b,),
            )?)
        }
        _ => panic!("Invalid global phase"),
    })
}
