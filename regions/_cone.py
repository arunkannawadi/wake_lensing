import numpy as np

from ._region import Region

__all__ = ("Cone",)


class Cone(Region):
    """Class representing a cone shape."""

    def __init__(self, base: np.ndarray, aperture_angle: float, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.aperture_cos = np.cos(aperture_angle / 2)

    @property
    def apex(self):
        return self.center

    def is_in_shape(self, point: np.ndarray) -> bool:
        # Vector from apex to point
        apex_to_point = point - self.apex
        # Vector from apex to base
        axis_vector = self.base - self.apex

        # Check if point is in the infinite cone
        dot_product = np.dot(apex_to_point, axis_vector)
        is_in = (
            dot_product / (np.linalg.norm(apex_to_point) * np.linalg.norm(axis_vector))
            > self.aperture_cos
        )

        if not is_in:
            return False

        # Check if point is below the cone's round cap
        projection_length = dot_product / np.linalg.norm(axis_vector)
        return projection_length < np.linalg.norm(axis_vector)
