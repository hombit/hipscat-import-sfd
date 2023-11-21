use crate::norder_tiles::NorderTiles;

pub(crate) type Tree<S> = Vec<NorderTiles<S>>;
pub(crate) type TreeRef<'a, S> = &'a [NorderTiles<S>];
pub(crate) type TreeMutRef<'a, S> = &'a mut [NorderTiles<S>];

pub(crate) fn len_over_threshold<S>(tree: TreeRef<S>, threshold: usize) -> Option<usize> {
    tree.iter()
        .enumerate()
        .filter_map(|(norder, norder_tiles)| {
            if norder_tiles.len() >= threshold {
                Some(norder)
            } else {
                None
            }
        })
        .next()
}
