#[derive(Clone)]
pub(crate) struct TreeConfig {
    #[allow(dead_code)]
    n_root: usize,
    n_children: usize,
    max_norder: usize,
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

        Self {
            n_root,
            n_children,
            max_norder,
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        self.n_children
    }

    pub(crate) fn max_norder(&self) -> usize {
        self.max_norder
    }
}
