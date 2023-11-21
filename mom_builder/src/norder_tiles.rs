use crate::error::Error;
use std::ops::Range;

#[derive(PartialEq, Debug)]
pub(crate) struct NorderTiles<T> {
    indexes: Vec<usize>,
    values: Vec<T>,
}

impl<T> NorderTiles<T> {
    pub(crate) fn new() -> Self {
        Self {
            indexes: Vec::new(),
            values: Vec::new(),
        }
    }

    pub(crate) fn insert(&mut self, index: usize, value: T) -> Result<(), Error> {
        if !self.indexes.is_empty() && index <= self.indexes[0] {
            return Err(Error::IndexError);
        }
        self.indexes.push(index);
        self.values.push(value);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn append(&mut self, other: &mut Self) -> Result<(), Error> {
        if !self.indexes.is_empty() && !other.indexes.is_empty() {
            if self.indexes[self.indexes.len() - 1] >= other.indexes[0] {
                return Err(Error::IndexError);
            }
        }
        self.indexes.append(&mut other.indexes);
        self.values.append(&mut other.values);
        Ok(())
    }

    pub(crate) fn pop(&mut self) {
        self.indexes.pop();
        self.values.pop();
    }

    pub(crate) fn len(&self) -> usize {
        self.indexes.len()
    }

    pub(crate) fn get_last_checked(&self, indexes: Range<usize>) -> Result<&[T], Error> {
        let last = self.len();
        if indexes.len() > last {
            return Err(Error::IndexError);
        }
        let first = last - indexes.len();
        if self.indexes[first..last]
            .iter()
            .zip(indexes.clone())
            .all(|(&a, b)| a == b)
        {
            let last = self.len();
            let first = last - indexes.len();
            Ok(&self.values[first..last])
        } else {
            Err(Error::IndexError)
        }
    }

    pub(crate) fn into_tuple(self) -> (Vec<usize>, Vec<T>) {
        (self.indexes, self.values)
    }
}
