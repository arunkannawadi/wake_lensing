from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np

__all__ = ("Region",)

class ShapeProfile(ABC):
    """Abstract base class for different shapes."""

    @abstractmethod
    def is_in_shape(self, point: np.ndarray) -> bool:
        """Determines if a point is within the shape."""
        pass


class Region(ABC):
    """An abstract base class to describe a region of space.

    It defines a `contains` interface that can be used to determine if a given
    point is within the region.
    """

    def __init__(self, *, center: np.ndarray = np.zeros(3)):
        self.center = center

    def __repr__(self) -> str:
        args = ", ".join(f"{key}={val}" for key, val in self.__dict__.items())
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__:
            return False

        return self.__dict__ == other.__dict__

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Determines if a point is within the region.

        Parameters
        ----------
        point : np.ndarray
            The point to check if it is within the region.

        Returns
        -------
        in_region : bool
            True if the point is within the region, False otherwise.
        """
        raise NotImplementedError("Concrete subclasses must implement this method.")
