"""Helper functions to generate different shapes."""
from typing import Callable
import numpy as np


def windowed_sinc(lobe_count: int) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a cosine-windowed sinc function."""
    def shape_func(t: np.ndarray) -> np.ndarray:
        t = 2*t - 1
        window = np.cos(t * np.pi) * 0.5 + 0.5
        return np.sinc(t * (lobe_count + 1)) * window

    return shape_func


def constant(value: float) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a constant function."""
    def shape_func(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, value)

    return shape_func
