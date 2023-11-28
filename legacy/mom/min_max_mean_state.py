from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Collection, Optional


@dataclass
class MinMaxMeanState:
    min: Real
    max: Real
    mean: Real


class MinMaxMeanStateMerger(ABC):
    @abstractmethod
    def validate_state(self, state: MinMaxMeanState) -> bool:
        raise NotImplementedError

    def merge_states(self, states: Collection[MinMaxMeanState]) -> Optional[MinMaxMeanState]:
        state =  MinMaxMeanState(
            min=min(state.min for state in states),
            max=max(state.max for state in states),
            mean=sum(state.mean for state in states) / len(states),
        )
        if self.validate_state(state):
            return state
        return None
