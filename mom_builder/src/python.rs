use crate::build_tree::build_tree;
use crate::state::MinMaxMeanState;
use crate::state::{MinMaxMeanStateMerger, MinMaxMeanStateValidator, StateBuilder};
use std::convert::Infallible;

use crate::tree_config::TreeConfig;
use itertools::Itertools;
use numpy::ndarray::{Array1 as NdArray, ArrayView1 as NdArrayView, NdFloat};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction(name = "mom_from_array")]
fn py_mom_from_array<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    max_norder: usize,
    thereshold: f64,
) -> Vec<(&'py PyArray1<usize>, &'py PyArray1<f64>)> {
    mom_from_array(a.as_array(), max_norder, thereshold)
        .into_iter()
        .map(|(indexes, values)| (indexes.into_pyarray(py), values.into_pyarray(py)))
        .collect()
}

fn mom_from_array<T>(
    values: NdArrayView<T>,
    max_norder: usize,
    thereshold: T,
) -> Vec<(NdArray<usize>, NdArray<T>)>
where
    T: NdFloat,
    MinMaxMeanState<T>: Into<T>,
{
    let it = values.iter().map(|&x| -> Result<T, Infallible> { Ok(x) });

    mom_from_it(it, max_norder, thereshold).expect("Should not fail with infallible error")
}

#[pyfunction(name = "mom_from_batch_it")]
fn py_mom_from_batch_it<'py>(
    py: Python<'py>,
    it: &'py PyAny,
    max_norder: usize,
    thereshold: f64,
) -> PyResult<Vec<(&'py PyArray1<usize>, &'py PyArray1<f64>)>> {
    let it = it
        .iter()?
        .map(|batch| -> PyResult<_> {
            let py_array = batch?.downcast::<PyArray1<f64>>()?;
            let py_ro = py_array.readonly();
            Ok(py_ro.to_owned_array())
        })
        .flatten_ok();

    Ok(mom_from_it(it, max_norder, thereshold)?
        .into_iter()
        .map(|(indexes, values)| (indexes.into_pyarray(py), values.into_pyarray(py)))
        .collect())
}

fn mom_from_it<'a, T, E>(
    it: impl Iterator<Item = Result<T, E>>,
    max_norder: usize,
    thereshold: T,
) -> Result<Vec<(NdArray<usize>, NdArray<T>)>, E>
where
    T: NdFloat,
    MinMaxMeanState<T>: Into<T>,
{
    let it_states = it.map(|x| x.map(MinMaxMeanState::from));

    let state_builder = StateBuilder::new(
        MinMaxMeanStateMerger::new(),
        MinMaxMeanStateValidator::new(thereshold),
    );

    let tree_config = TreeConfig::new(12usize, 4usize, max_norder);

    let tree = build_tree(&state_builder, &tree_config, it_states)?;

    let output = tree
        .into_iter()
        .map(|tiles| {
            let (indexes, values) = tiles
                .into_inner()
                .expect("Cannot consume RwLock to get tree norder-storage")
                .into_tuple();
            (
                indexes.into_iter().collect::<NdArray<_>>(),
                values.into_iter().map(|x| x.into()).collect::<NdArray<_>>(),
            )
        })
        .collect::<Vec<_>>();
    Ok(output)
}

/// A Python module implemented in Rust.
#[pymodule]
fn mom_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_mom_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_mom_from_batch_it, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mom_from_array() {
        let a = NdArray::linspace(0.0, 47.0, 48);
        let max_norder = 1;
        let thereshold = 0.5;
        let mom = mom_from_array(a.view(), max_norder, thereshold);

        assert_eq!(mom.len(), 2);

        assert_eq!(mom[0].0.len(), mom[0].0.len());
        assert_eq!(
            mom[0].0.as_slice_memory_order().unwrap(),
            (1usize..=11).collect::<Vec<_>>().as_slice(),
        );
        approx::assert_relative_eq!(
            mom[0].1.as_slice_memory_order().unwrap(),
            vec![5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5].as_slice()
        );

        assert_eq!(mom[1].0.len(), mom[1].0.len());
        assert_eq!(
            mom[1].0.as_slice_memory_order().unwrap(),
            (0usize..=3).collect::<Vec<_>>().as_slice(),
        );
        approx::assert_relative_eq!(
            mom[1].1.as_slice_memory_order().unwrap(),
            vec![0.0, 1.0, 2.0, 3.0].as_slice()
        );
    }
}
