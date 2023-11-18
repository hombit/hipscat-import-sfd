use crate::norder_tiles::NorderTiles;
use crate::state::{MergeStates, StateBuilder, StateIsValid};
use crate::tree::Tree;
use crate::tree_config::TreeConfig;
use std::sync::RwLock;

struct TreeBuilder<'a, S, Merger, Validator> {
    builder: &'a StateBuilder<Merger, Validator>,
    config: &'a TreeConfig,
    tree: Tree<S>,
}

impl<'a, S, Merger, Validator> TreeBuilder<'a, S, Merger, Validator>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
{
    fn build(&mut self, max_norder_states: impl IntoIterator<Item = S>) {
        let mut state_it = max_norder_states.into_iter();

        let parent_order_n_tiles = self.config.max_norder_n_tile() / self.config.n_children();
        for parent_index in 0..parent_order_n_tiles {
            let states: Vec<_> = (&mut state_it).take(self.config.n_children()).collect();
            assert_eq!(states.len(), self.config.n_children());

            // Run recursive merge
            let merged = self.merge_states(&states, parent_index, self.config.max_norder() - 1);

            // If not merged then insert the states into the current norder, max_norder
            if !merged {
                let mut max_norder_tiles = self.tree[self.config.max_norder()]
                    .write()
                    .expect("Cannot lock tree max_norder-storage to write");
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
    }

    fn merge_states(&mut self, states: &[S], parent_index: usize, parent_norder: usize) -> bool {
        assert_eq!(states.len(), self.config.n_children());

        let merged_state = self.builder.merger.merge(states);

        if self.builder.validator.state_is_valid(&merged_state) {
            self.tree[parent_norder]
                .write()
                .expect("Cannot lock tree norder-storage for write")
                .insert(parent_index, merged_state)
                .expect("Tree insertion failed");

            if parent_index % self.config.n_children() == self.config.n_children() - 1 {
                self.process_group_by_top_title_index(parent_index, parent_norder);
            }
            true
        } else {
            false
        }
    }

    fn process_group_by_top_title_index(&mut self, top_index: usize, norder: usize) {
        // If norder is 0, then we are done.
        if norder == 0 {
            return;
        }

        let offset_from_bottom = self.config.n_children() - 1;
        assert_eq!(top_index % self.config.n_children(), offset_from_bottom);

        // Collect pre-merged states
        // If any of the states is missing we cannot merge further, so return
        let states_result = self.tree[norder]
            .read()
            .expect("Cannot lock tree norder-storage for read")
            .get_last_checked(top_index - offset_from_bottom..top_index + 1)
            .map(|sl| sl.to_vec());
        let states = match states_result {
            Ok(states) => states,
            Err(_) => return,
        };

        let parent_index = top_index / self.config.n_children();
        let parent_norder = norder - 1;

        // Merge tiles if possible
        let merged = self.merge_states(&states, parent_index, parent_norder);
        // Delete pre-merged states if they were merged
        if merged {
            let mut norder_tiles = self.tree[norder]
                .write()
                .expect("Cannot lock tree norder-storage for write");
            for _ in 0..self.config.n_children() {
                norder_tiles.pop();
            }
        }
    }
}

pub(crate) fn build_tree<S, Merger, Validator>(
    state_builder: &StateBuilder<Merger, Validator>,
    tree_config: &TreeConfig,
    max_norder_states: impl IntoIterator<Item = S>,
) -> Tree<S>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
    S: Copy + std::fmt::Debug,
{
    let mut tree_builder = TreeBuilder {
        builder: state_builder,
        config: tree_config,
        tree: (0..=tree_config.max_norder())
            .map(|_| RwLock::new(NorderTiles::new()))
            .collect(),
    };
    tree_builder.build(max_norder_states);
    tree_builder.tree
}
