//! Leaf state and merge rules.
//!
//! Merge logic is represented by two traits, [MergeStates] and [StateIsValid].
//! The first one is responsible for merging leaf states to their parent nodes, and the second one
//! is responsible for checking if the merged state is valid. Two objects implementing these traits
//! are combined into [StateBuilder] object.
//!
//! Currently, the only type of the leaf state is implemented, [MinMaxMeanState], with a pair of
//! merge rules, [MinMaxMeanStateMerger] and [MinMaxMeanStateValidator]. The state is represented
//! by three values: minimum, maximum and mean. [MinMaxMeanStateMerger] merges states by taking
//! minimum and maximum of the states and calculating the mean value. [MinMaxMeanStateValidator]
//! checks if the relative difference between minimum and maximum is less than a given threshold.

use numpy::ndarray::NdFloat;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Trait for merging leaf states to their parent nodes.
pub trait MergeStates {
    /// Type of the leaf state.
    type State: Sized;

    /// Merges the given states to a single state.
    fn merge(&self, states: &[Self::State]) -> Self::State;
}

/// Trait for checking if the merged state is valid.
pub trait StateIsValid {
    /// Type of the leaf state.
    type State;

    /// Checks if the given state is valid, i.e. if it should be stored in the tree.
    fn state_is_valid(&self, state: &Self::State) -> bool;
}

/// Leaf state with minimum, maximum and mean values.
///
/// It implements [Into] trait for [f32] and [f64], so it can be converted to its mean value.
/// [From] trait for `T` which just assigns the given value to all three fields.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinMaxMeanState<T> {
    /// Minimum value.
    pub min: T,
    /// Maximum value.
    pub max: T,
    /// Mean value.
    pub mean: T,
}

impl<T> MinMaxMeanState<T>
where
    T: Copy,
{
    /// Creates a new [MinMaxMeanState] with the given value, which is used as minimum, maximum and
    /// mean.
    pub fn new(value: T) -> Self {
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

/// Merges leaf states by taking minimum and maximum of the states and calculating the mean value.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct MinMaxMeanStateMerger<T> {
    phantom: PhantomData<T>,
}

impl<T> MinMaxMeanStateMerger<T> {
    pub fn new() -> Self {
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

    /// Merges the given states by taking minimum and maximum of the states and calculating the mean
    /// value.
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

/// Checks if the relative difference between minimum and maximum is less than a given threshold.
///
/// Basically it checks if `abs(max - min) / max <= threshold`, but with some edge-case handling.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct MinMaxMeanStateValidator<T> {
    threshold: T,
}

impl<T> MinMaxMeanStateValidator<T>
where
    T: NdFloat,
{
    /// Creates a new [MinMaxMeanStateValidator] with the given threshold.
    ///
    /// The threshold must be non-negative, otherwise the method panics.
    pub fn new(threshold: T) -> Self {
        assert!(threshold >= T::zero());
        Self { threshold }
    }

    /// Returns the threshold.
    pub fn threshold(&self) -> T {
        self.threshold
    }
}

impl<T> StateIsValid for MinMaxMeanStateValidator<T>
where
    T: NdFloat,
{
    type State = MinMaxMeanState<T>;

    /// Checks if the relative difference between minimum and maximum is less than the threshold.
    ///
    /// Basically it checks if `abs(max - min) / norm <= threshold`, where `norm` is the maximum of
    /// the absolute values of `min` and `max`. If both are zero, the method returns `true`.  
    fn state_is_valid(&self, state: &Self::State) -> bool {
        let denominator = T::max(T::abs(state.min), T::abs(state.max));
        if denominator.is_zero() {
            return true;
        }
        let ratio = (state.max - state.min) / denominator;
        ratio <= self.threshold
    }
}

/// State merge rules.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct StateBuilder<Merger, Validator> {
    /// State merger, implements [MergeStates] trait.
    pub merger: Merger,
    /// State validator, implements [StateIsValid] trait.
    pub validator: Validator,
}

impl<S, Merger, Validator> StateBuilder<Merger, Validator>
where
    Merger: MergeStates<State = S>,
    Validator: StateIsValid<State = S>,
{
    /// Creates a new [StateBuilder] with the given merger and validator.
    pub fn new(merger: Merger, validator: Validator) -> Self {
        Self { merger, validator }
    }
}
