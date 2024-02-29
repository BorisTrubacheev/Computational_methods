import numpy as np


def get_spectral_conditionality_number(a: np.ndarray, n=2) -> float:
    return np.linalg.norm(a, n) * np.linalg.norm(np.linalg.inv(a), n)


def get_volume_conditionality_number(a: np.ndarray):
    return np.prod(np.sqrt(np.sum(a ** 2, axis=1))) / np.abs(np.linalg.det(a))


def get_angle_conditionality_number(a: np.ndarray):
    c = np.linalg.inv(a)
    return np.max(np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=0))
