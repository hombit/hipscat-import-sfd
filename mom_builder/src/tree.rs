use crate::norder_tiles::NorderTiles;
use std::sync::RwLock;

pub(crate) type Tree<S> = Vec<RwLock<NorderTiles<S>>>;
