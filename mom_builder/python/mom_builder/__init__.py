from .mom_builder import *


def gen_mom_from_fn(fn, max_norder, *, intermediate_norder=None, threshold):
    if intermediate_norder is None:
        intermediate_norder = max_norder // 2
    builder = MOMBuilder(max_norder, intermediate_norder, threshold)

    for intermediate_index in range(builder.intermediate_ntiles):
        indexes = builder.subtree_maxnorder_indexes(intermediate_index)
        values = fn(max_norder, indexes)
        yield from builder.build_subtree(intermediate_index, values)

    yield from builder.build_top_tree()
