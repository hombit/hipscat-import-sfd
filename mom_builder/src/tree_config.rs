pub(crate) struct TreeConfig {
    #[allow(dead_code)]
    n_root: usize,
    n_children: usize,
    max_norder: usize,
    max_norder_n_tile: usize,
    penult_norder_n_tile: usize,
}

impl TreeConfig {
    pub(crate) fn new(
        n_root: impl Into<usize>,
        n_children: impl Into<usize>,
        max_norder: impl Into<usize>,
    ) -> Self {
        let n_root = n_root.into();
        let n_children = n_children.into();
        let max_norder = max_norder.into();

        let max_norder_n_tile = n_root * n_children.pow(max_norder as u32);
        let penult_norder_n_tile = max_norder_n_tile / n_children;

        Self {
            n_root,
            n_children,
            max_norder,
            max_norder_n_tile,
            penult_norder_n_tile: penult_norder_n_tile,
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        self.n_children
    }

    pub(crate) fn max_norder(&self) -> usize {
        self.max_norder
    }

    pub(crate) fn max_norder_n_tile(&self) -> usize {
        self.max_norder_n_tile
    }

    pub(crate) fn penult_norder_n_tile(&self) -> usize {
        self.penult_norder_n_tile
    }
}
