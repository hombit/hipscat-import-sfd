#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("Index is invalid")]
    IndexError,
}
