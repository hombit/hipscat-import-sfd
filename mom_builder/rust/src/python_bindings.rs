use crate::build_tree::build_tree;
use crate::state::{
    MinMaxMeanState, MinMaxMeanStateMerger, MinMaxMeanStateValidator, StateBuilder,
};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::RwLock;

use crate::tree::Tree;
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
    threshold: T,
) -> Vec<(NdArray<usize>, NdArray<T>)>
where
    T: NdFloat,
    MinMaxMeanState<T>: Into<T>,
{
    let it = values.iter().map(|&x| -> Result<T, Infallible> { Ok(x) });

    mom_from_it(it, max_norder, threshold).expect("Should not fail with infallible error")
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
    threshold: T,
) -> Result<Vec<(NdArray<usize>, NdArray<T>)>, E>
where
    T: NdFloat,
    MinMaxMeanState<T>: Into<T>,
{
    let it_states = it
        .enumerate()
        .map(|(index, x)| x.map(|value| (index, MinMaxMeanState::from(value))));

    let state_builder = StateBuilder::new(
        MinMaxMeanStateMerger::new(),
        MinMaxMeanStateValidator::new(threshold),
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

#[pyclass(name = "MOMBuilder")]
struct MomBuilder {
    #[pyo3(get)]
    intermediate_norder: usize,
    #[pyo3(get)]
    max_norder: usize,
    intermediate_states: RwLock<BTreeMap<usize, Option<MinMaxMeanState<f64>>>>,
    intermediate_tree_config: TreeConfig,
    top_tree_config: TreeConfig,
    state_builder: StateBuilder<MinMaxMeanStateMerger<f64>, MinMaxMeanStateValidator<f64>>,
}

impl MomBuilder {
    fn tree_to_python<'py>(
        &self,
        py: Python<'py>,
        tree: Tree<MinMaxMeanState<f64>>,
        norder_offset: usize,
    ) -> Vec<(usize, &'py PyArray1<usize>, &'py PyArray1<f64>)> {
        tree.into_iter()
            .enumerate()
            .filter(|(_norder, tiles)| tiles.len() != 0)
            .map(|(norder, tiles)| {
                let (indexes, values) = tiles.into_tuple();
                (
                    norder + norder_offset,
                    NdArray::from_vec(indexes).into_pyarray(py),
                    values
                        .into_iter()
                        .map(|x| x.into())
                        .collect::<NdArray<_>>()
                        .into_pyarray(py),
                )
            })
            .collect()
    }

    fn max_norder_index_offset(&self, intermediate_index: usize) -> usize {
        intermediate_index * self.intermediate_tree_config.max_norder_ntiles()
    }
}

#[pymethods]
impl MomBuilder {
    #[new]
    fn __new__(max_norder: usize, intermediate_norder: usize, threshold: f64) -> PyResult<Self> {
        if intermediate_norder > max_norder {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "intermediate_norder must be less than or equal to max_norder",
            ));
        }

        let state_builder = StateBuilder::new(
            MinMaxMeanStateMerger::new(),
            MinMaxMeanStateValidator::new(threshold),
        );

        Ok(Self {
            intermediate_states: RwLock::new(BTreeMap::new()),
            intermediate_norder,
            max_norder,
            intermediate_tree_config: TreeConfig::new(
                1usize,
                4usize,
                max_norder - intermediate_norder,
            ),
            top_tree_config: TreeConfig::new(12usize, 4usize, intermediate_norder),
            state_builder,
        })
    }

    #[getter]
    fn intermediate_ntiles(&self) -> usize {
        self.top_tree_config.max_norder_ntiles()
    }

    fn subtree_maxnorder_indexes<'py>(
        &self,
        py: Python<'py>,
        intermediate_index: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        if intermediate_index >= self.top_tree_config.max_norder_ntiles() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "intermediate_index is out of range",
            ));
        }
        let offset = self.max_norder_index_offset(intermediate_index);

        let output = PyArray1::from_vec(
            py,
            (offset..offset + self.intermediate_tree_config.max_norder_ntiles())
                .map(|i| i)
                .collect(),
        );
        Ok(output)
    }

    fn build_subtree<'py>(
        &self,
        py: Python<'py>,
        intermediate_index: usize,
        a: PyReadonlyArray1<f64>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyArray1<f64>)>> {
        if self
            .intermediate_states
            .read()
            .expect("Cannot lock states storage for read")
            .contains_key(&intermediate_index)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "State with this index already exists",
            ));
        }

        // We need to build a subtree with a single root node, we will accumulate all the nodes later
        // Current subtree has an offset in indexes on the deepest level (maximum norder)
        let index_offset = self.max_norder_index_offset(intermediate_index);

        let array = a.as_array();

        let it_states =
            array
                .iter()
                .enumerate()
                .map(|(relative_index, &x)| -> Result<_, Infallible> {
                    Ok((index_offset + relative_index, MinMaxMeanState::new(x)))
                });
        let mut tree = build_tree(
            self.state_builder,
            self.intermediate_tree_config.clone(),
            it_states,
        )?;

        // Extract root node from the tree, it should have at most one state
        let root_tiles = tree.remove(0);
        let state = match root_tiles.len() {
            0 => None,
            1 => {
                let (indexes, states) = root_tiles.into_tuple();
                assert_eq!(indexes[0], intermediate_index);
                Some(states[0])
            }
            _ => panic!("Root node should have at most one state"),
        };
        self.intermediate_states
            .write()
            .expect("Cannot lock states storage for write")
            .insert(intermediate_index, state);

        // Return the rest of the subtree
        Ok(self.tree_to_python(py, tree, self.intermediate_norder + 1))
    }

    fn build_top_tree<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyArray1<f64>)>> {
        if self
            .intermediate_states
            .read()
            .expect("Cannot lock states storage for read")
            .keys()
            .enumerate()
            .any(|(i, index)| *index != i)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Intermediate tiles are not contiguous",
            ));
        }

        let states = std::mem::take(
            &mut *self
                .intermediate_states
                .write()
                .expect("Cannot lock states storage for write"),
        );

        let it_states = states.into_iter().filter_map(|(index, state)| {
            state.map(|state| -> Result<_, Infallible> { Ok((index, state)) })
        });

        let tree = build_tree(self.state_builder, self.top_tree_config.clone(), it_states)?;

        Ok(self.tree_to_python(py, tree, 0))
    }
}

#[pymodule]
fn mom_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_mom_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_mom_from_batch_it, m)?)?;
    m.add_class::<MomBuilder>()?;
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