use numpy::ndarray::NdFloat;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub(crate) trait MergeStates {
    type State: Sized;
    fn merge(&self, states: &[Self::State]) -> Self::State;
}

pub(crate) trait StateIsValid {
    type State;
    fn state_is_valid(&self, state: &Self::State) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct MinMaxMeanState<T> {
    pub(crate) min: T,
    pub(crate) max: T,
    pub(crate) mean: T,
}

impl<T> MinMaxMeanState<T>
where
    T: Copy,
{
    pub(crate) fn new(value: T) -> Self {
        Self {
            min: value,
            max: value,
            mean: value,
        }
    }
}

impl From<MinMaxMeanState<f32>> for f32 {
    fn from(val: MinMaxMeanState<f32>) -> Self {
        val.mean
    }
}

impl From<MinMaxMeanState<f64>> for f64 {
    fn from(val: MinMaxMeanState<f64>) -> Self {
        val.mean
    }
}

impl<T> From<T> for MinMaxMeanState<T>
where
    T: Copy,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct MinMaxMeanStateMerger<T> {
    phantom: PhantomData<T>,
}

impl<T> MinMaxMeanStateMerger<T> {
    pub(crate) fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> MergeStates for MinMaxMeanStateMerger<T>
where
    T: NdFloat,
{
    type State = MinMaxMeanState<T>;

    fn merge(&self, states: &[Self::State]) -> Self::State {
        assert!(!states.is_empty());

        let mut min = states[0].min;
        let mut max = states[0].max;
        let mut sum = states[0].mean;
        for state in states.iter().skip(1) {
            if state.min < min {
                min = state.min;
            }
            if state.max > max {
                max = state.max;
            }
            sum += state.mean;
        }
        Self::State {
            min,
            max,
            mean: sum / T::from(states.len()).expect("N cannot be casted to float"),
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct MinMaxMeanStateValidator<T> {
    threshold: T,
}

impl<T> MinMaxMeanStateValidator<T>
where
    T: NdFloat,
{
    pub(crate) fn new(threshold: T) -> Self {
        assert!(threshold >= T::zero());
        Self { threshold }
    }

    pub(crate) fn threshold(&self) -> T {
        self.threshold
    }
}

impl<T> StateIsValid for MinMaxMeanStateValidator<T>
where
    T: NdFloat,
{
    type State = MinMaxMeanState<T>;

    fn state_is_valid(&self, state: &Self::State) -> bool {
        let denominator = T::max(T::abs(state.min), T::abs(state.max));
        if denominator.is_zero() {
            return true;
        }
        let ratio = (state.max - state.min) / denominator;
        ratio <= self.threshold
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct StateBuilder<Merger, Validator> {
    pub(crate) merger: Merger,
    pub(crate) validator: Validator,
}

impl<S, Merger, Validator> StateBuilder<Merger, Validator>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
{
    pub(crate) fn new(merger: Merger, validator: Validator) -> Self {
        Self { merger, validator }
    }
}
