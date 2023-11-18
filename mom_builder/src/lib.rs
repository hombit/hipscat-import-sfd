/// Simple Python module to build Multi-order healpix maps.
///
/// TODO:
/// 1. Add more tests
/// 2. Add docs
/// 3. Reimplement without RWLock
/// 4. Run in parallel
/// 5. mom_from_func to build more lazily
/// 6. Support f32 arrays
mod build_tree;
mod error;
mod norder_tiles;
mod python;
mod state;
mod tree;
mod tree_config;
