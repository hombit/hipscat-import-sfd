use crate::norder_tiles::NorderTiles;
use crate::state::{MergeStates, StateBuilder, StateIsValid};
use crate::tree::{Tree, TreeMutRef};
use crate::tree_config::TreeConfig;

struct TreeBuilder<'a, Merger, Validator> {
    builder: &'a StateBuilder<Merger, Validator>,
    config: &'a TreeConfig,
}

impl<'a, S, Merger, Validator> TreeBuilder<'a, Merger, Validator>
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

        let merged_state = self.builder.merger.merge(states);

        if self.builder.validator.state_is_valid(&merged_state) {
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

pub(crate) fn build_tree<S, Merger, Validator, E>(
    state_builder: &StateBuilder<Merger, Validator>,
    tree_config: &TreeConfig,
    max_norder_states: impl IntoIterator<Item = Result<S, E>>,
) -> Result<Tree<S>, E>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
{
    TreeBuilder {
        builder: state_builder,
        config: tree_config,
    }
    .build(max_norder_states)
}
