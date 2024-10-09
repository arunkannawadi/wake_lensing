from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from typing import Self


__all__ = (
    "MassProfile",
)


class MassProfile(ABC):
    """Abstract base class for mass profiles."""

    @abstractmethod
    def __call__(self: Self, r: float) -> float:
        """Call signature for mass profiles."""
        raise NotImplementedError
