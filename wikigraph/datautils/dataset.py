"""Base class of the datasets."""

import abc
from typing import Any, Iterator


class Dataset(abc.ABC):
    """Base class for all datasets.
  All sub-classes should define `_load_data()` where an iterator
  `self._data_iter` should be instantiated that iterates over the dataset.
  """

    def __init__(self):
        """Constructor."""
        self._data_iter = None  # An iterator produced by `self._load_data`.

    @abc.abstractmethod
    def _load_data(self) -> Iterator[Any]:
        """Prepare data for another pass through the dataset.
    This method should return a generator in a child class.
    """

    def __next__(self):
        return next(self._data_iter)

    def __iter__(self):
        self._data_iter = self._load_data()
        return self
