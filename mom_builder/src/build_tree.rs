use crate::norder_tiles::NorderTiles;
use crate::state::{MergeStates, StateBuilder, StateIsValid};
use crate::tree::{len_over_threshold, Tree, TreeMutRef};
use crate::tree_config::TreeConfig;

#[derive(Clone)]
pub(crate) struct TreeBuilder<Merger, Validator> {
    pub(crate) state_builder: StateBuilder<Merger, Validator>,
    pub(crate) config: TreeConfig,
}

impl<S, Merger, Validator> TreeBuilder<Merger, Validator>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
{
    fn build<E>(
        &self,
        max_norder_states: impl IntoIterator<Item = Result<S, E>>,
    ) -> Result<Tree<S>, E> {
        let mut tree: Tree<S> = (0..=self.config.max_norder())
            .map(|_| NorderTiles::new())
            .collect();
        let (max_norder_tiles, subtree) = tree.split_last_mut().expect("tree should not be empty");

        let mut state_it = max_norder_states.into_iter();

        let parent_order_n_tiles = self.config.max_norder_n_tile() / self.config.n_children();
        for parent_index in 0..parent_order_n_tiles {
            let states = (&mut state_it)
                .take(self.config.n_children())
                .collect::<Result<Vec<_>, E>>()?;
            assert_eq!(states.len(), self.config.n_children());

            // Run recursive merge
            let merged = self.merge_states(subtree, &states, parent_index);

            // If not merged then insert the states into the current norder, max_norder
            if !merged {
                for (index, state) in (self.config.n_children() * parent_index
                    ..self.config.n_children() * (parent_index + 1))
                    .zip(states.into_iter())
                {
                    max_norder_tiles
                        .insert(index, state)
                        .expect("Tree insertion failed");
                }
            }
        }

        Ok(tree)
    }

    fn merge_states(&self, subtree: TreeMutRef<S>, states: &[S], parent_index: usize) -> bool {
        assert_eq!(states.len(), self.config.n_children());
        assert_ne!(subtree.len(), 0);

        let merged_state = self.state_builder.merger.merge(states);

        if self.state_builder.validator.state_is_valid(&merged_state) {
            subtree
                .last_mut()
                .expect("tree should not be empty")
                .insert(parent_index, merged_state)
                .expect("Tree insertion failed");

            if parent_index % self.config.n_children() == self.config.n_children() - 1 {
                self.process_group_by_top_title_index(subtree, parent_index);
            }
            true
        } else {
            false
        }
    }

    fn process_group_by_top_title_index(&self, tree: TreeMutRef<S>, top_index: usize) {
        // If we at norder=0 (root nodes), we are done
        if tree.len() == 1 {
            return;
        }

        let offset_from_bottom = self.config.n_children() - 1;
        assert_eq!(top_index % self.config.n_children(), offset_from_bottom);

        let (norder_tiles, subtree) = tree.split_last_mut().expect("tree should not be empty");

        // Collect pre-merged states
        // If any of the states is missing we cannot merge further, so return
        let states_result = norder_tiles
            .get_last_checked(top_index - offset_from_bottom..top_index + 1)
            .map(|sl| sl.to_vec());
        let states = match states_result {
            Ok(states) => states,
            Err(_) => return,
        };

        let parent_index = top_index / self.config.n_children();

        // Merge tiles if possible
        let merged = self.merge_states(subtree, &states, parent_index);
        // Delete pre-merged states if they were merged
        if merged {
            for _ in 0..self.config.n_children() {
                norder_tiles.pop();
            }
        }
    }
}

pub(crate) struct TreeBuildStage<Merger, Validator, S> {
    builder: TreeBuilder<Merger, Validator>,
    tree: Tree<S>,
    penutl_index: usize,
    batch_size: usize,
}

impl<Merger, Validator, S> TreeBuildStage<Merger, Validator, S> {
    pub(crate) fn new(builder: TreeBuilder<Merger, Validator>, batch_size: usize) -> Self {
        assert_eq!(batch_size % builder.config.n_children(), 0);
        Self {
            tree: (0..=builder.config.max_norder())
                .map(|_| NorderTiles::new())
                .collect(),
            penutl_index: 0,
            batch_size,
            builder,
        }
    }

    pub(crate) fn next_with_iter<E>(
        &mut self,
        max_norder_tiles: &mut impl Iterator<Item = Result<S, E>>,
    ) -> Option<Result<(usize, NorderTiles<S>), E>>
    where
        Merger: MergeStates<State = S>,
        Validator: StateIsValid<State = S>,
        S: Copy + std::fmt::Debug,
    {
        let n_children = self.builder.config.n_children();

        // Run tlle adding until we reach the end or we have batch_size element on any norder
        loop {
            // If we have reached the end, return first non-empty norder tiles storage
            if self.penutl_index == self.builder.config.penult_norder_n_tile() {
                break self
                    .tree
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(norder, norder_tiles)| {
                        if norder_tiles.len() == 0 {
                            None
                        } else {
                            let norder_tiles = std::mem::replace(norder_tiles, NorderTiles::new());
                            Some(Ok((norder, norder_tiles)))
                        }
                    })
                    .next();
            }

            assert!(
                self.tree.len() > 1,
                "norder should be at least 1, so tree should have at least two norder_tiles"
            );

            if let Some(norder) = len_over_threshold(&mut self.tree, self.batch_size) {
                let norder_tiles = std::mem::replace(&mut self.tree[norder], NorderTiles::new());
                break Some(Ok((norder, norder_tiles)));
            }

            let result_states = max_norder_tiles
                .take(n_children)
                .collect::<Result<Vec<_>, E>>();
            let states = match result_states {
                Ok(states) => states,
                Err(e) => break Some(Err(e)),
            };
            assert_eq!(states.len(), n_children);

            let (max_norder_tiles, subtree) = self
                .tree
                .split_last_mut()
                .expect("tree should not be empty");

            // Run recursive merge
            let merged = self
                .builder
                .merge_states(subtree, &states, self.penutl_index);

            // If not merged then insert the states into the current norder, max_norder
            if !merged {
                for (index, state) in (n_children * self.penutl_index
                    ..n_children * (self.penutl_index + 1))
                    .zip(states.into_iter())
                {
                    max_norder_tiles
                        .insert(index, state)
                        .expect("Tree insertion failed");
                }
            }

            self.penutl_index += 1;
        }
    }
}

pub(crate) struct TreeBuildIterator<Merger, Validator, S, Input>
where
    Input: ?Sized,
{
    stage: TreeBuildStage<Merger, Validator, S>,
    max_norder_tiles: Input,
}

impl<Merger, Validator, S, Input, E> TreeBuildIterator<Merger, Validator, S, Input>
where
    Input: Iterator<Item = Result<S, E>>,
{
    pub(crate) fn new(
        max_norder_tiles: impl IntoIterator<Item = Result<S, E>, IntoIter = Input>,
        builder: TreeBuilder<Merger, Validator>,
        batch_size: usize,
    ) -> Self {
        assert_eq!(batch_size % builder.config.n_children(), 0);
        Self {
            stage: TreeBuildStage::new(builder, batch_size),
            max_norder_tiles: max_norder_tiles.into_iter(),
        }
    }
}

impl<Merger, Validator, S, Input, E> Iterator for TreeBuildIterator<Merger, Validator, S, Input>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
    Input: Iterator<Item = Result<S, E>>,
{
    type Item = Result<(usize, NorderTiles<S>), E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.stage.next_with_iter(&mut self.max_norder_tiles)
    }
}

pub(crate) fn build_tree<S, Merger, Validator, E>(
    state_builder: StateBuilder<Merger, Validator>,
    tree_config: TreeConfig,
    max_norder_states: impl IntoIterator<Item = Result<S, E>>,
) -> Result<Tree<S>, E>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
{
    TreeBuilder {
        state_builder,
        config: tree_config,
    }
    .build(max_norder_states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{MinMaxMeanState, MinMaxMeanStateMerger, MinMaxMeanStateValidator};
    use std::convert::Infallible;

    #[test]
    fn test_builder_vs_iterator() {
        let state_builder = StateBuilder::new(
            MinMaxMeanStateMerger::new(),
            MinMaxMeanStateValidator::new(0.5),
        );
        let tree_config = TreeConfig::new(12usize, 4usize, 1usize);
        let tree_builder = TreeBuilder {
            state_builder: state_builder,
            config: tree_config,
        };

        let max_norder_states: Vec<_> = (0..48)
            .map(|x| -> Result<_, Infallible> { Ok(MinMaxMeanState::new(x as f64)) })
            .collect();

        let tree_from_builder = tree_builder.build(max_norder_states.clone()).unwrap();

        let tree_from_iterator_large_batch: Tree<_> =
            TreeBuildIterator::new(max_norder_states.clone(), tree_builder.clone(), 48)
                .map(|r| r.expect("Infallible error should never occur"))
                // check that norder is in ascending order
                .enumerate()
                .map(|(i, (norder, norder_tiles))| {
                    assert_eq!(i, norder);
                    norder_tiles
                })
                .collect();
        assert_eq!(tree_from_builder, tree_from_iterator_large_batch);

        let tree_from_iterator_small_batches = {
            let mut tree = vec![NorderTiles::new(), NorderTiles::new()];
            TreeBuildIterator::new(max_norder_states, tree_builder, 4)
                .map(|r| r.expect("Infallible error should never occur"))
                .for_each(|(norder, mut norder_tiles)| {
                    tree[norder]
                        .append(&mut norder_tiles)
                        .expect("Tree insertion failed");
                });
            tree
        };
        assert_eq!(tree_from_builder, tree_from_iterator_small_batches);
    }
}
