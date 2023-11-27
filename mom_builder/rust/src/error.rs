//! Error types for the mom_builder crate.

/// Error enum for the mom_builder crate.
#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("Index is invalid")]
    IndexError,
}
