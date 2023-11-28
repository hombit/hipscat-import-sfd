from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import TypeVar, Collection, Optional


STATE_T = TypeVar('STATE_T')


class TileStates:
    def __init__(self):
        self.indexes: list[int] = []
        self.values: list[STATE_T] = []

    def __getitem__(self, index: int) -> STATE_T:
        idx = bisect_right(self.indexes, index) - 1
        if idx < 0:
            raise IndexError(f"index is out of range: {self.indexes[0] - self.indexes[-1]}")
        if self.indexes[idx] != index:
            raise IndexError(f"index not found")
        return self.values[idx]

    def __setitem__(self, index: int, value: STATE_T) -> None:
        if len(self.indexes) > 0 and index <= self.indexes[-1]:
            raise IndexError(f"Indexes must be inserted in ascending order: {index} <= {self.indexes[-1]}")
        self.indexes.append(index)
        self.values.append(value)

    def __contains__(self, index: int) -> bool:
        idx = bisect_right(self.indexes, index) - 1
        return self.indexes[idx] == index

    def __len__(self) -> int:
        return len(self.indexes)

    def pop(self):
        self.indexes.pop()
        self.values.pop()


class AbstractMultiorderMapBuilder(ABC):
    # Quad-tree by default
    n_children = 4
    """Binary logarithm of children nodes number"""

    # Healpix has 12 norder=1 tiles
    n_root = 12
    """Number of root nodes"""

    def __init__(self, max_norder: int):
        self.max_norder = max_norder
        self.tiles = {norder: TileStates() for norder in range(self.max_norder + 1)}

    @property
    def max_norder_n_tile(self):
        """Number of tiles of the maximum norder"""
        return self.n_root * self.n_children**self.max_norder

    @abstractmethod
    def calculate_state(self, index_max_norder: int) -> STATE_T:
        raise NotImplementedError

    @abstractmethod
    def merge_states(self, states: Collection[STATE_T]) -> Optional[STATE_T]:
        raise NotImplementedError

    def process_states(self, states: Collection[STATE_T], parent_index: int, parent_norder: int) -> bool:
        """Process states of a group of tiles and merge them if possible

        If states are merged, `process_top_tile` which could call this function again

        Parameters
        ----------
        states : collection of states, (n_children,)
            States to investigate and merge
        parent_index : int
            Index of the common parent of considered tiles
        parent_norder : int
            And its norder

        Returns
        -------
        merged : bool
            True if the tiles were merged, False otherwise.
        """
        assert len(states) == self.n_children, str(states)

        merged_state = self.merge_states(states)
        merged = merged_state is not None

        if merged:
            self.tiles[parent_norder][parent_index] = merged_state
            # Check if parent tile is the last one in its group
            # If so, process the group.
            if parent_index % self.n_children == self.n_children - 1:
                self.process_group_by_top_tile(parent_index, parent_norder)

        return merged

    def process_group_by_top_tile(self, top_index: int, norder: int) -> None:
        """Process the group of tiles selected by largest element

        Parameters
        ----------
        top_index : int
            Largest index of the group, it is assumed that all (n_children-1)
            indexes have already been processed previoisly
        norder : int
            Its norder
        """
        offset_from_bottom = self.n_children - 1
        assert top_index % self.n_children == offset_from_bottom

        # If norder is 0, we are done
        if norder == 0:
            return

        norder_tiles = self.tiles[norder]

        # Collect pre-merged states
        # If any of the states is missing we cannot merge further, so return
        states: list[STATE_T] = []
        indexes = range(top_index - offset_from_bottom, top_index + 1)
        for index in indexes:
            try:
                state = norder_tiles[index]
            except IndexError:
                return
            states.append(state)

        parent_index = top_index // self.n_children
        parent_norder = norder - 1

        # Merge tiles if possible
        merged = self.process_states(states, parent_index, parent_norder)

        # Delete pre-merged states if they were merged
        if merged:
            for _index in reversed(indexes):
                norder_tiles.pop()

    def build(self) -> dict[TileStates]:
        max_norder_tiles = self.tiles[self.max_norder]

        for parent_index in range(0, self.max_norder_n_tile // self.n_children):
            indexes = range(self.n_children * parent_index, self.n_children * (parent_index + 1))

            states = [self.calculate_state(index) for index in indexes]

            merged = self.process_states(states, parent_index, self.max_norder - 1)
            if not merged:
                for index, state in zip(indexes, states):
                    max_norder_tiles[index] = state

        return self.tiles
