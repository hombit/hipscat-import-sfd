/// Simple Python module to build Multi-order healpix maps.
///
/// TODO:
/// 1. Add more tests
/// 2. Add docs
/// 3. --Reimplement without RWLock-- (done)
/// 4. Run in parallel
/// 5. --mom_from_func to build more lazily-- (added mom_from_it instead)
/// 6. Support f32 arrays
/// 7. Make builder to be Rust iterator / Python generator
mod build_tree;
mod error;
mod exclusive_option;
mod norder_tiles;
mod python_bindings;
mod state;
mod tree;
mod tree_config;
