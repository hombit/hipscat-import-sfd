use crate::norder_tiles::NorderTiles;

pub(crate) type Tree<S> = Vec<NorderTiles<S>>;
pub(crate) type TreeMutRef<'a, S> = &'a mut [NorderTiles<S>];
