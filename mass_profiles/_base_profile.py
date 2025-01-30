from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

__all__ = ["MassProfile"]


class MassProfile(ABC):
    """Abstract base class for mass profiles."""

    def __init__(self, *, r_s: float, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, r: float) -> float:
        """Call signature for mass profiles."""
        raise NotImplementedError

    @property
    def r_s(self):
        """Characteristic radius. See the class docstring for details."""
        return self._r_s

    @r_s.setter
    def r_s(self, value: float):
        if not value >= 0:
            raise ValueError("Scale radius has to be a non-negative value.")
        self._r_s = value
