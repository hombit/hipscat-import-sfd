//! Simple Rust crate and Python extension module to build multi-order healpix maps.
//!
//! The Rust part of the crate provides a set of tools for bottom-top tree building.
//! For a given balanced tree with fixed number of children per node,
//! the crate provides tools to merge leaves to their parent nodes recursively, see
//! [crate::build_tree] module. The merge rules are defined by the user, see [state] module
//! for the details.
//!
//! Trees could be merged in a streaming fashion, i.e. the user can provide an iterator over the
//! leaves values.
//!
//! Another notable feature is that currently we support "multi-root trees", i.e. a forest of
//! independent trees with the same structure and maximum depth (which we called `max_norder`
//! adopting [healpix](https://healpix.jpl.nasa.gov) terminology). See [tree_config] for the
//! tree specification.
//!
//! Currently, the only type of the leaf state is implemented, [state::MinMaxMeanState],
//! which is accompained by a couple of merge rules, [state::MinMaxMeanStateMerger] and
//! [state::MinMaxMeanStateValidator].
//!
//! The Python part of the crate provides a Python extension module, see `python_bindings` for the
//! details. The module implements building of healpix maps (twelve trees with four children per
//! node). The Python module is supposed to be built with [maturin](https://maturin.rs) and has
//! a small, but important part written in Python.
//!
//! TODO:
//! 1. Add tests
//! 2. --Add docs--
//! 3. --Reimplement without RWLock-- (done)
//! 4. --Run in parallel-- (implemented with MOMBuilder(thread_safe=true))
//! 5. --mom_from_func to build more lazily-- (added mom_from_it instead)
//! 6. --Support f32 arrays-- (done)
//! 7. --Make builder to be Rust iterator / Python generator--
//!     (hard to do, MomBuilder and gen_mom_from_fn instead)
//!
pub mod build_tree;
pub mod error;
pub mod exclusive_option;
pub mod norder_leaves;
mod python_bindings;
pub mod state;
pub mod tree;
pub mod tree_config;
