# shape_profiles/base_shape.py
from abc import ABC, abstractmethod
import numpy as np

class ShapeProfile(ABC):
    """Abstract base class for different shapes."""

    @abstractmethod
    def is_in_shape(self, point: np.ndarray) -> bool:
        """Determines if a point is within the shape."""
        pass
