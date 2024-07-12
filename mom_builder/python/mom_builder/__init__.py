"""Module for building multi-order healpix maps (MOMs).

The module provides multiple interfaces to build healpix MOMs merging
healpix tiles of some maximum order (healpix tile tree depth) into
tiles (leaves) of some lower order (top tree depth). The merging is
done by computing the minimum, maximum and mean values of the tiles.
A group of four children tiles are merged into a parent tile if
the relative difffenre between the minimum and maximum values of the
children tiles is below some user-defined threshold:
    `(max - min) / norm <= threshold`,
where `norm` is the maximum absolute value of the minimum and maximum.
"""

from .mom_builder import *


def gen_mom_from_fn(fn, *, max_norder, split_norder=None, merger, dtype=None):
    """Generator that builds a tree using the given function of healpix tile

    The generator goes over all subtrees and calls the given function with
    the max_norder and the indexes of the subtree, and yields the result of
    this subtree merge. Then it yields the result of the top tree merge.
    This generator is not thread-safe, and should not be sent across threads.

    See more details about the building process in the documentation of
    `MOMBuilder`.

    Parameters
    ----------
    fn : callable
        A function that takes the max_norder and the indexes of a subtree,
        and returns a MinMaxMeanState.
        The signature is:
        `fn(order:int, indexes:np.ndarray[uint64]) -> np.ndarray`
        Indexes are the healpix indexes of of the order `order`
        (always `max_norder`). The returned array is the value of the
        function for each of the indexes, it must has the same length as
        `indexes` and always be the same float dtype (so it should not return
        float32 for some indexes and int64 for others).
    merger : MOMMerger or float
        Merging algorithm to use. If a float is given, min-max-mean states
        are merged if the relative difference between the minimum and maximum
        values is below this threshold. It is the same as setting merger to
        MOMMerger("min-max-mean", "rtol", threshold=threshold, dtype=dtype),
        where dtype is derived from the return of `fn`.
    max_norder : int
        Maximum depth of the healpix tree.
    split_norder : int, optional
        The depth of the top tree. If not given, it is `max_norder // 2`,
        which should lead to a consistent memory usage. However, it can
        be suboptimal for performance of `fn`, consider set it to lower
        values to have a trade-off between memory and performance.

    Yields
    ------
    list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
        List of (norder, indexes, values) tuples. The generator has two
        phases: first it yields the subtrees, then it yields the top tree.
        During the first phase, it yeilds the subtrees in the order of
        increasing index and, in an inner loop, in the order of increasing
        norder. That means that the same `norder` can be yielded multiple
        times, but always with increasing `indexes`.
    """
    if not isinstance(merger, MOMMerger):
        import numpy as np

        if dtype is None:
            dtype = fn(max_norder, np.arange(16, dtype=np.uint64)).dtype
        merger = MOMMerger("min-max-mean", "rtol", threshold=merger, dtype=dtype)

    if split_norder is None:
        split_norder = max_norder // 2
    # We are not going to build the tree in parallel, so we can use the non-thread-safe builder
    builder = MOMBuilder(merger, max_norder=max_norder, split_norder=split_norder, thread_safe=False)

    for subtree_index in range(builder.num_subtrees):
        indexes = builder.subtree_maxnorder_indexes(subtree_index)
        values = fn(max_norder, indexes)
        yield from builder.build_subtree(subtree_index, values)

    yield from builder.build_top_tree()
