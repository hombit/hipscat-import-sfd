use crate::build_tree::build_tree;
use crate::state::{
    MinMaxMeanState, MinMaxMeanStateMerger, MinMaxMeanStateValidator, StateBuilder,
};
use crate::tree::Tree;
use crate::tree_config::TreeConfig;
use itertools::Itertools;
use numpy::ndarray::{Array1 as NdArray, ArrayView1 as NdArrayView, NdFloat};
use numpy::IntoPyArray;
use numpy::{dtype, PyArray1, PyUntypedArray};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::iter::once;
use std::sync::RwLock;

/// Builds multi-order healpix maps from an array of leaf values
///
/// Parameters
/// ----------
/// a : numpy.ndarray of float32 or float64
///     Input array of leaf values at the maximum healpix norder. It must
/// max_norder : int
///     Maximum depth of the healpix tree.
/// threshold : float
///     When merging leaf states to their parent nodes, the relative difference
///     between minimum and maximum values is checked against this threshold.
///     Must be non-negative.
///
/// Returns
/// -------
/// list of (np.ndarray of uint64, np.ndarray of float32 or float64)
///     List of (indexes, values) pairs, a pair per norder.
///
#[pyfunction(name = "mom_from_array")]
fn py_mom_from_array<'py>(
    py: Python<'py>,
    a: &'py PyUntypedArray,
    max_norder: usize,
    threshold: f64,
) -> PyResult<Vec<(&'py PyArray1<usize>, &'py PyUntypedArray)>> {
    if a.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input array must be 1-dimensional",
        ));
    }

    let element_type = a.dtype();

    if element_type.is_equiv_to(dtype::<f32>(py)) {
        let a = a.downcast::<PyArray1<f32>>()?;
        Ok(mom_from_array(py, a, max_norder, threshold as f32))
    } else if element_type.is_equiv_to(dtype::<f64>(py)) {
        let a = a.downcast::<PyArray1<f64>>()?;
        Ok(mom_from_array(py, a, max_norder, threshold))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input array's dtype must be f32 or f64",
        ))
    }
}

fn mom_from_array<'py, T>(
    py: Python<'py>,
    py_array: &'py PyArray1<T>,
    max_norder: usize,
    threshold: T,
) -> Vec<(&'py PyArray1<usize>, &'py PyUntypedArray)>
where
    T: NdFloat + numpy::Element,
    MinMaxMeanState<T>: Into<T>,
{
    let readonly = py_array.readonly();
    let array = readonly.as_array();
    let it = array.iter().map(|&x| -> Result<T, Infallible> { Ok(x) });

    mom_from_it(py, it, max_norder, threshold).expect("Should not fail with infallible error")
}

/// Builds multi-order healpix maps from an iterator of leaf values.
///
/// It is a variant of `mom_from_array` which accepts an iterator
/// of leaf value batches instead of an array. It is useful when
/// the input array is too large to fit into memory.
///
/// Parameters
/// ----------
/// it : iterator of numpy.ndarray of float32 or float64
///     Iterator of batches of leaf values at the maximum healpix norder.
///     The batch size is not limited, but it is recommended to use batches
///     large enough to make the tree building efficient.
/// max_norder : int
///     Maximum depth of the healpix tree.
/// threshold : float
///     When merging leaf states to their parent nodes, the relative difference
///     between minimum and maximum values is checked against this threshold.
///     Must be non-negative.
///
/// Returns
/// -------
/// list of (np.ndarray of uint64, np.ndarray of float32 or float64)
///     List of (indexes, values) pairs, a pair per norder.
#[pyfunction(name = "mom_from_batch_it")]
fn py_mom_from_batch_it<'py>(
    py: Python<'py>,
    it: &'py PyAny,
    max_norder: usize,
    threshold: f64,
) -> PyResult<Vec<(&'py PyArray1<usize>, &'py PyUntypedArray)>> {
    let mut py_iter = it.iter()?;
    let first_element = py_iter
        .next()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Input iterator is empty"))??;

    if let Ok(array) = first_element.downcast::<PyArray1<f32>>() {
        mom_from_first_and_it(py, array, &py_iter, max_norder, threshold as f32)
    } else if let Ok(array) = first_element.downcast::<PyArray1<f64>>() {
        mom_from_first_and_it(py, array, &py_iter, max_norder, threshold)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Iterator items must be 1-D numpy arrays having dtype f32 or f64",
        ))
    }
}

fn mom_from_first_and_it<'py, T>(
    py: Python<'py>,
    first: &'py PyArray1<T>,
    py_iter: &'py PyIterator,
    max_norder: usize,
    threshold: T,
) -> PyResult<Vec<(&'py PyArray1<usize>, &'py PyUntypedArray)>>
where
    T: NdFloat + numpy::Element,
    MinMaxMeanState<T>: Into<T>,
{
    let it = once(first)
        .map(|array| Ok(array))
        .chain(py_iter.map(|batch| Ok(batch?.downcast::<PyArray1<T>>()?)))
        .map_ok(|py_array| {
            let py_ro = py_array.readonly();
            py_ro.to_owned_array()
        })
        .flatten_ok();

    mom_from_it(py, it, max_norder, threshold)
}

fn mom_from_it<'py, T, E>(
    py: Python<'py>,
    it: impl Iterator<Item = Result<T, E>>,
    max_norder: usize,
    threshold: T,
) -> Result<Vec<(&'py PyArray1<usize>, &'py PyUntypedArray)>, E>
where
    T: NdFloat + numpy::Element,
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
                indexes.into_iter().collect::<NdArray<_>>().into_pyarray(py),
                values
                    .into_iter()
                    .map(|x| x.into())
                    .collect::<NdArray<_>>()
                    .into_pyarray(py)
                    .as_untyped(),
            )
        })
        .collect::<Vec<_>>();
    Ok(output)
}

#[derive(Serialize, Deserialize)]
struct GenericMomBuilder<T> {
    split_norder: usize,
    max_norder: usize,
    thread_safe: bool,
    subtree_states: RwLock<BTreeMap<usize, Option<MinMaxMeanState<T>>>>,
    subtree_config: TreeConfig,
    top_tree_config: TreeConfig,
    state_builder: StateBuilder<MinMaxMeanStateMerger<T>, MinMaxMeanStateValidator<T>>,
}

impl<T> GenericMomBuilder<T>
where
    T: NdFloat + numpy::Element,
    MinMaxMeanState<T>: Into<T>,
{
    fn new(
        max_norder: usize,
        split_norder: usize,
        threshold: T,
        thread_safe: bool,
    ) -> PyResult<Self> {
        if split_norder > max_norder {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "split_norder must be less than or equal to max_norder",
            ));
        }

        let state_builder = StateBuilder::new(
            MinMaxMeanStateMerger::new(),
            MinMaxMeanStateValidator::new(threshold),
        );

        Ok(Self {
            subtree_states: RwLock::new(BTreeMap::new()),
            split_norder,
            max_norder,
            thread_safe,
            subtree_config: TreeConfig::new(1usize, 4usize, max_norder - split_norder),
            top_tree_config: TreeConfig::new(12usize, 4usize, split_norder),
            state_builder,
        })
    }

    fn subtree_states_is_empty(&self) -> bool {
        self.subtree_states
            .read()
            .expect("Cannot lock states storage for read")
            .is_empty()
    }

    fn tree_to_python<'py>(
        &self,
        py: Python<'py>,
        tree: Tree<MinMaxMeanState<T>>,
        norder_offset: usize,
    ) -> Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)> {
        tree.into_iter()
            .enumerate()
            .filter(|(_norder, tiles)| tiles.len() != 0)
            .map(|(norder, tiles)| {
                let (indexes, values) = tiles.into_tuple();
                (
                    norder + norder_offset,
                    PyArray1::from_vec(py, indexes),
                    PyArray1::from_iter(py, values.into_iter().map(|x| x.into())).as_untyped(),
                )
            })
            .collect()
    }

    fn max_norder_index_offset(&self, subtree_index: usize) -> usize {
        subtree_index * self.subtree_config.max_norder_nleaves()
    }

    fn build_subtree<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
        a: &'py PyArray1<T>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        py.allow_threads(|| {
            if self
                .subtree_states
                .read()
                .expect("Cannot lock states storage for read")
                .contains_key(&subtree_index)
            {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "State with this index already exists",
                ))
            } else {
                Ok(())
            }
        })?;

        let py_ro = a.readonly();

        // We have to introduce an overhead here to avoid double locking of the states storage
        let tree = if self.thread_safe {
            let owned_array = py_ro.to_owned_array();
            py.allow_threads(|| self.build_subtree_impl(owned_array.view(), subtree_index))
        } else {
            let array_view = py_ro.as_array();
            self.build_subtree_impl(array_view, subtree_index)
        }?;

        // Return the rest of the subtree
        Ok(self.tree_to_python(py, tree, self.split_norder + 1))
    }

    fn build_subtree_impl(
        &self,
        array: NdArrayView<T>,
        subtree_index: usize,
    ) -> PyResult<Tree<MinMaxMeanState<T>>> {
        // We need to build a subtree with a single root node, we will accumulate all the nodes later
        // Current subtree has an offset in indexes on the deepest level (maximum norder)
        let index_offset = self.max_norder_index_offset(subtree_index);

        let it_states =
            array
                .iter()
                .enumerate()
                .map(|(relative_index, &x)| -> Result<_, Infallible> {
                    Ok((index_offset + relative_index, MinMaxMeanState::new(x)))
                });
        let mut tree = build_tree(self.state_builder, self.subtree_config.clone(), it_states)?;

        // Extract root node from the tree, it should have at most one state
        let root_tiles = tree.remove(0);
        let state = match root_tiles.len() {
            0 => None,
            1 => {
                let (indexes, states) = root_tiles.into_tuple();
                assert_eq!(indexes[0], subtree_index);
                Some(states[0])
            }
            _ => panic!("Root node should have at most one state"),
        };
        self.subtree_states
            .write()
            .expect("Cannot lock states storage for write")
            .insert(subtree_index, state);

        Ok(tree)
    }

    fn build_top_tree<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        let tree = py.allow_threads(|| {
            {
                let states = self
                    .subtree_states
                    .read()
                    .expect("Cannot lock states storage for read");

                if states.len() != self.top_tree_config.max_norder_nleaves() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Not all subtrees are built",
                    ));
                }
                if states.keys().enumerate().any(|(i, index)| *index != i) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Subtree tiles are not contiguous",
                    ));
                }
            }

            // Large memory allocation, we need to optimize it
            let states = std::mem::take(
                &mut *self
                    .subtree_states
                    .write()
                    .expect("Cannot lock states storage for write"),
            );

            let it_states = states.into_iter().filter_map(|(index, state)| {
                state.map(|state| -> Result<_, Infallible> { Ok((index, state)) })
            });

            Ok(build_tree(
                self.state_builder,
                self.top_tree_config.clone(),
                it_states,
            )?)
        })?;

        Ok(self.tree_to_python(py, tree, 0))
    }

    fn extend(&self, other: &Self) -> PyResult<()> {
        if other.subtree_states_is_empty() {
            // Nothing to merge with
            return Ok(());
        }

        let mut states = self
            .subtree_states
            .write()
            .expect("Cannot lock states storage for write");

        // We are cleaning the other's states storage here
        let other_states = std::mem::take(
            &mut *other
                .subtree_states
                .write()
                .expect("Cannot lock states storage for read"),
        );

        for (index, state) in other_states.into_iter() {
            if let Some(_) = states.insert(index, state) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "State with index {index} already exists",
                )));
            }
        }

        Ok(())
    }
}

/// A low-level two-step builder of multi-order healpix maps.
///
/// The builder is supposed to be used in the following way:
/// 1. Create a builder with `MOMBuilder` constructor.
/// 2. Build all subtrees with `build_subtree()` method.
///    Use `subtree_maxnorder_indexes()` method to get indexes of
///    the subtree leaves on the maximum depth.
/// 3. Build the top tree with `build_top_tree()` method.
///
///
/// Note, that dtype of the input arrays must be the same for all subtrees,
/// and all subtrees must be built before building the top tree.
/// Consider to user `gen_mom_from_fn()` function, which wraps this class
/// and provides a more convenient interface.
///
///   t   t                     0
///   o   r      _______________|_________________
///   p   e     /          |           |          \
///       e    0           1           2          3
///          / | | \    / | | \    / | | \   /  |  | \
///  sub-   0 1  2 3   4  5 6  7  8 9 10 11 12 13 14 15
///  trees |||||||||  |||||||||  |||||||||  |||||||||||
///  ...   xxxxxxxxx  xxxxxxxxx  xxxxxxxxx  xxxxxxxxxxx
///
/// (1/12 of a healpix tree. split_norder=1, max_norder>2. We first build four
///  subtrees, and then, if all four subtree roots are not empty and mergeable,
///  we build the top tree.)
///
/// Parameters
/// ----------
/// max_norder : int
///     Maximum depth of the healpix tree. It is the maximum norder of the
///     whole tree, not of the subtrees.
/// split_norder : int
///     Maximum depth of the top tree. It is level of the tree where it is
///     split into subtrees.
/// threshold : float
///     When merging leaf states to their parent nodes, the relative difference
///     between minimum and maximum values is checked against this threshold.
///     Must be non-negative.
/// thread_safe : bool
///     If True, the builder will use a thread-safe implementation. It forces
///     copying of the input arrays to release GIL, so it could be a bit
///     slower. If False, the builder will use a non-thread-safe
///     implementation, which could make itself into a deadlock if used in
///     a multithreaded environment.
///
/// Attributes
/// ----------
/// max_norder : int
/// split_norder : int
/// num_subtrees : int
///     Number of subtrees, it is equal to the number of leaves in the top
///     tree on its maximum depth.
///
/// Methods
/// -------
/// subtree_maxnorder_indexes(subtree_index)
///     Returns indexes of the subtree leaves on the maximum depth.
/// build_subtree(subtree_index, a)
///     Builds a subtree from an array of leaf values. The array must
///     correspond to indexes given by `subtree_maxnorder_indexes()`.
///     Reurns a list of (norder, indexes, values) tuples.
/// build_top_tree()
///     Builds the top tree from subtrees. Returns a list of
///     (norder, indexes, values) tuples.
///
#[derive(Serialize, Deserialize)]
#[pyclass(name = "MOMBuilder", module = "mom_builder.mom_builder")]
struct MomBuilder {
    inner_f32: GenericMomBuilder<f32>,
    inner_f64: GenericMomBuilder<f64>,
}

#[derive(Clone, Copy)]
enum MomBuilderInnerType {
    F32,
    F64,
    Unknown,
    Invalid,
}

impl MomBuilderInnerType {
    fn mergeable_with_other(&self, other: &MomBuilderInnerType) -> Option<bool> {
        match (self, other) {
            (Self::F32, Self::F32) => Some(true),
            (Self::F64, Self::F64) => Some(true),
            (Self::F32, Self::F64) => Some(false),
            (Self::F64, Self::F32) => Some(false),
            (Self::F32 | Self::F64 | Self::Unknown, Self::Unknown) => Some(true),
            (Self::Unknown, Self::F32 | Self::F64) => Some(true),
            (Self::Invalid, _) => None,
            (_, Self::Invalid) => None,
        }
    }
}

impl MomBuilder {
    fn inner_type(&self) -> MomBuilderInnerType {
        match (
            !self.inner_f32.subtree_states_is_empty(),
            !self.inner_f64.subtree_states_is_empty(),
        ) {
            (true, false) => MomBuilderInnerType::F32,
            (false, true) => MomBuilderInnerType::F64,
            (false, false) => MomBuilderInnerType::Unknown,
            (true, true) => MomBuilderInnerType::Invalid,
        }
    }
}

#[pymethods]
impl MomBuilder {
    #[new]
    #[pyo3(signature = (max_norder, *, split_norder, threshold, thread_safe=true))]
    fn __new__(
        max_norder: usize,
        split_norder: usize,
        threshold: f64,
        thread_safe: bool,
    ) -> PyResult<Self> {
        let inner_f32 =
            GenericMomBuilder::<f32>::new(max_norder, split_norder, threshold as f32, thread_safe)?;
        let inner_f64 =
            GenericMomBuilder::<f64>::new(max_norder, split_norder, threshold, thread_safe)?;
        Ok(Self {
            inner_f32,
            inner_f64,
        })
    }

    /// Maximum depth of the healpix tree.
    #[getter]
    fn max_norder(&self) -> usize {
        self.inner_f32.max_norder
    }

    /// Depth of the top tree.
    #[getter]
    fn split_norder(&self) -> usize {
        self.inner_f32.split_norder
    }

    /// Number of subtrees
    ///
    /// It is equal to the number of leaves in the top tree on its maximum
    /// depth.
    #[getter]
    fn num_subtrees(&self) -> usize {
        self.inner_f32.top_tree_config.max_norder_nleaves()
    }

    /// Returns healpix indexes of the subtree leaves on the maximum depth.
    ///
    /// It is supposed to be used to create an input array for
    /// `build_subtree()`.
    ///
    /// Parameters
    /// ----------
    /// subtree_index : int
    ///     Index of the subtree, in [0, num_subtrees).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of uint64
    ///     Array of healpix indexes of the subtree leaves on the maximum
    ///     norder.
    ///
    fn subtree_maxnorder_indexes<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        if subtree_index >= self.inner_f32.top_tree_config.max_norder_nleaves() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "subtree_index is out of range",
            ));
        }
        let offset = self.inner_f32.max_norder_index_offset(subtree_index);

        let output = PyArray1::from_vec(
            py,
            (offset..offset + self.inner_f32.subtree_config.max_norder_nleaves())
                .map(|i| i)
                .collect(),
        );
        Ok(output)
    }

    /// Builds a subtree from an array of leaf values.
    ///
    /// This method builds a subtree, stores the root node state if exists
    /// and returns the rest of the subtree otherwise.
    ///
    /// It must be called once per subtree, and the subtrees must be built
    /// with `a` of the same dtype.
    ///
    /// Parameters
    /// ----------
    /// subtree_index : int
    ///     Healpix index of the subtree root, in [0, num_subtrees).
    /// a : numpy.ndarray of float32 or float64
    ///     Leaf values at the maximum healpix norder. It must correspond
    ///     to indexes given by `subtree_maxnorder_indexes()`.
    ///
    /// Returns
    /// -------
    /// list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
    ///     List of (norder, indexes, values) tuples, a tuple per non-empty
    ///     norder. The norder is absolute (not relative to the subtree).
    ///
    fn build_subtree<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
        a: &'py PyUntypedArray,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        if a.ndim() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input array must be 1-dimensional",
            ));
        }

        let a_element_type = a.dtype();
        let slf_element_type = py.allow_threads(|| self.inner_type());

        if a_element_type.is_equiv_to(dtype::<f32>(py)) {
            match slf_element_type {
                MomBuilderInnerType::F32 | MomBuilderInnerType::Unknown => {},
                MomBuilderInnerType::F64 => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Got f32 array, but previously f64 array was processed",
                )),
                MomBuilderInnerType::Invalid => return Err(pyo3::exceptions::PyValueError::new_err(
                    "MOMBuilder was used with both f32 and f64 arrays, please rebuild with the same dtype",
                )),
            };
            let a = a.downcast::<PyArray1<f32>>()?;
            self.inner_f32.build_subtree(py, subtree_index, a)
        } else if a_element_type.is_equiv_to(dtype::<f64>(py)) {
            match slf_element_type {
                MomBuilderInnerType::F64 | MomBuilderInnerType::Unknown => {},
                MomBuilderInnerType::F32 => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Got f64 array, but previously f32 array was processed",
                )),
                MomBuilderInnerType::Invalid => return Err(pyo3::exceptions::PyValueError::new_err(
                    "MOMBuilder was used with both f32 and f64 arrays, please rebuild with the same dtype",
                )),
            };
            let a = a.downcast::<PyArray1<f64>>()?;
            self.inner_f64.build_subtree(py, subtree_index, a)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Input array's dtype must be f32 or f64",
            ))
        }
    }

    /// Builds the top tree from subtrees.
    ///
    /// It must be called after all subtrees are built with `build_subtree()`.
    ///
    /// Returns
    /// -------
    /// list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
    ///     List of (norder, indexes, values) tuples, a tuple per non-empty
    ///     norder.
    fn build_top_tree<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        let slf_element_type = py.allow_threads(|| self.inner_type());

        match slf_element_type {
            MomBuilderInnerType::F32 => self.inner_f32.build_top_tree(py),
            MomBuilderInnerType::F64 => self.inner_f64.build_top_tree(py),
            MomBuilderInnerType::Unknown => Err(pyo3::exceptions::PyValueError::new_err(
                "No subtrees were built, please build at least one subtree",
            )),
            MomBuilderInnerType::Invalid => Err(pyo3::exceptions::PyValueError::new_err(
                "MOMBuilder was used with both f32 and f64 arrays, please rebuild with the same dtype",
            )),
        }
    }

    /// Extends from other MOMBuilder, clearing the other MOMBuilder.
    ///
    /// Parameters
    /// ----------
    /// other : MOMBuilder
    ///     MOMBuilder to extend from. It must have the same max_norder, split_norder, threshold,
    ///     and be built from the data of the same dtype.
    ///
    /// Returns
    /// -------
    /// None
    fn extend(&self, py: Python, other: &Self) -> PyResult<()> {
        if self.max_norder() != other.max_norder() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_norder must be the same",
            ));
        }
        if self.split_norder() != other.split_norder() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "split_norder must be the same",
            ));
        }

        let slf_element_type = py.allow_threads(|| self.inner_type());
        match slf_element_type.mergeable_with_other(&py.allow_threads(|| other.inner_type())) {
            Some(true) => {}
            Some(false) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "dtype must be the same",
                ));
            }
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "One of the MOMBuilder is invalid",
                ));
            }
        }

        py.allow_threads(|| -> PyResult<()> {
            // We merge both inner states for convinience, we have actually checked that
            // they are mergeable.
            self.inner_f32.extend(&other.inner_f32)?;
            self.inner_f64.extend(&other.inner_f64)?;
            Ok(())
        })?;

        Ok(())
    }

    // pickle stuff
    fn __getnewargs_ex__<'py>(
        &self,
        py: Python<'py>,
    ) -> ((usize,), HashMap<&'static str, PyObject>) {
        let args = (self.inner_f64.max_norder,);
        let kwargs = [
            ("split_norder", self.inner_f64.split_norder.into_py(py)),
            (
                "threshold",
                self.inner_f64
                    .state_builder
                    .validator
                    .threshold()
                    .into_py(py),
            ),
            ("thread_safe", self.inner_f64.thread_safe.into_py(py)),
        ];
        (args, kwargs.into_iter().collect())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let vec_bytes =
            serde_pickle::to_vec(&self, serde_pickle::SerOptions::new()).map_err(|err| {
                pyo3::exceptions::PyException::new_err(format!(
                    "Cannot pickle MOMBuilder: {}",
                    err.to_string()
                ))
            })?;
        Ok(PyBytes::new(py, &vec_bytes))
    }

    fn __setstate__<'py>(&mut self, state: &'py PyBytes) -> PyResult<()> {
        *self = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot unpickle MOMBuilder: {}",
                    err.to_string()
                ))
            })?;
        Ok(())
    }
}

#[pymodule]
fn mom_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_mom_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_mom_from_batch_it, m)?)?;
    m.add_class::<MomBuilder>()?;
    Ok(())
}
