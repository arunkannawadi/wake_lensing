import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .probability_profiles import ShapeProfile

__all__ = ("ProbabilityCalculator",)


class ProbabilityCalculator:
    @staticmethod
    def calculate(file_path, shape_left, shape_right):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return None, None, None

        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        particle_ids = data[:, 0].astype(int)
        unique_particles = np.unique(particle_ids)
        total_timesteps = len(np.unique(data[:, 1]))

        left_time_counts = []
        right_time_counts = []
        all_positions = []

        for particle_id in unique_particles:
            particle_data = data[particle_ids == particle_id]
            positions = particle_data[:, 2:5]  # x, y, z columns
            all_positions.append(positions)

            # Check if positions are inside the left shape
            left_in_shape = np.array([shape_left.is_in_shape(pos) for pos in positions])
            left_time_counts.append(np.sum(left_in_shape))

            # Check if positions are inside the right shape
            right_in_shape = np.array(
                [shape_right.is_in_shape(pos) for pos in positions]
            )
            right_time_counts.append(np.sum(right_in_shape))

        left_probabilities = np.array(left_time_counts) / total_timesteps
        right_probabilities = np.array(right_time_counts) / total_timesteps
        all_positions = np.vstack(all_positions)

        return left_probabilities, right_probabilities, all_positions


__all__ = ["ProbabilityCalculator"]
