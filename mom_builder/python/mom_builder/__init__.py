from .mom_builder import *


def gen_mom_from_fn(fn, max_norder, *, subtree_norder=None, threshold):
    if subtree_norder is None:
        subtree_norder = max_norder // 2
    builder = MOMBuilder(max_norder, subtree_norder, threshold)

    for subtree_index in range(builder.subtree_ntiles):
        indexes = builder.subtree_maxnorder_indexes(subtree_index)
        values = fn(max_norder, indexes)
        yield from builder.build_subtree(subtree_index, values)

    yield from builder.build_top_tree()
