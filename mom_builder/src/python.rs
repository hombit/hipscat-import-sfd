use crate::build_tree::{build_tree, TreeBuildIterator, TreeBuildStage, TreeBuilder};
use crate::state::{
    MinMaxMeanState, MinMaxMeanStateMerger, MinMaxMeanStateValidator, StateBuilder,
};
use std::convert::Infallible;
use std::ops::DerefMut;

use crate::tree::Tree;
use crate::tree_config::TreeConfig;
use itertools::Itertools;
use numpy::ndarray::{s, Array1 as NdArray, ArrayView1 as NdArrayView, NdFloat};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyIterator;

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

    let tree = build_tree(state_builder, tree_config, it_states)?;

    let output = tree
        .into_iter()
        .map(|tiles| {
            let (indexes, values) = tiles.into_tuple();
            (
                indexes.into_iter().collect::<NdArray<_>>(),
                values.into_iter().map(|x| x.into()).collect::<NdArray<_>>(),
            )
        })
        .collect::<Vec<_>>();
    Ok(output)
}

#[pyclass(unsendable)]
struct MomBuilderIterator {
    stage: TreeBuildStage<
        MinMaxMeanStateMerger<f64>,
        MinMaxMeanStateValidator<f64>,
        MinMaxMeanState<f64>,
    >,
    input_it: PyObject,
    array: NdArray<f64>,
    cursor: usize,
}

#[pymethods]
impl MomBuilderIterator {
    #[new]
    fn __new__<'py>(
        py: Python<'py>,
        it: &'py PyAny,
        max_norder: usize,
        thereshold: f64,
        batch_size: usize,
    ) -> PyResult<Self> {
        let state_builder = StateBuilder::new(
            MinMaxMeanStateMerger::new(),
            MinMaxMeanStateValidator::new(thereshold),
        );
        let tree_builder = TreeBuilder {
            state_builder,
            config: TreeConfig::new(12usize, 4usize, max_norder),
        };
        let stage = TreeBuildStage::new(tree_builder, batch_size);
        let input_it = it.iter()?.into_py(py);
        Ok(Self {
            stage,
            input_it,
            array: NdArray::zeros(0),
            cursor: 0,
        })
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyObject>> {
        let py = slf.py();
        let slf: &mut Self = slf.deref_mut();

        let array_slice = slf.array.slice(s![slf.cursor..]);
        let mut new_array = NdArray::zeros(0);

        let mut it_states = array_slice
            .iter()
            .map(|&x| Ok(x))
            .chain(
                slf.input_it
                    .as_ref(py)
                    .iter()?
                    .map(|batch| {
                        let py_array = batch?.downcast::<PyArray1<f64>>()?;
                        let py_ro = py_array.readonly();
                        new_array = py_ro.to_owned_array();
                        Ok(&new_array)
                    })
                    .flatten_ok()
                    .map(|x: PyResult<&f64>| -> PyResult<f64> { x.copied() }),
            )
            .map(|x| x.map(|x| MinMaxMeanState::from(x)));
        slf.stage
            .next_with_iter(&mut it_states)
            .transpose()
            .map(|option| option.map(|(norder, norder_tiles)| py.None()))
    }
}

// #[pymethods]
// impl MomBuilderIterator {
//     #[new]
//     fn new<'py>(
//         _py: Python<'py>,
//         it: &PyAny,
//         max_norder: usize,
//         thereshold: f64,
//         batch_size: usize,
//     ) -> PyResult<Self> {
//         if batch_size % 4 != 0 {
//             return Err(pyo3::exceptions::PyValueError::new_err(
//                 "batch_size must be a multiple of 4",
//             ));
//         }
//
//         let it_states = it
//             .iter()?
//             .map(|batch| {
//                 let py_array = batch.unwrap().downcast::<PyArray1<f64>>().unwrap();
//                 let py_ro = py_array.readonly();
//                 py_ro.to_owned_array()
//             })
//             .flatten()
//             .map(|x| -> Result<_, Infallible> { Ok(MinMaxMeanState::from(x)) });
//
//         let state_builder = StateBuilder::new(
//             MinMaxMeanStateMerger::new(),
//             MinMaxMeanStateValidator::new(thereshold),
//         );
//
//         let tree_config = TreeConfig::new(12usize, 4usize, max_norder);
//         let state_builder = TreeBuilder {
//             state_builder,
//             config: tree_config,
//         };
//
//         let inner: Box<
//             TreeBuildIterator<
//                 MinMaxMeanStateMerger<f64>,
//                 MinMaxMeanStateValidator<f64>,
//                 MinMaxMeanState<f64>,
//                 dyn Iterator<Item = Result<MinMaxMeanState<f64>, Infallible>> + Send + Sync,
//             >,
//         > = Box::new(TreeBuildIterator::new(it_states, state_builder, batch_size));
//
//         Ok(Self { inner })
//     }
// }

#[pymodule]
fn mom_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_mom_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_mom_from_batch_it, m)?)?;
    m.add_class::<MomBuilderIterator>()?;
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
