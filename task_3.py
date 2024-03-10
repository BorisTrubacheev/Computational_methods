import numpy as np


def get_rotate_matrix(n, sin, cos, i, j) -> np.array:
    matrix = np.eye(n)
    matrix[i, i] = cos
    matrix[i, j] = -sin
    matrix[j, i] = sin
    matrix[j, j] = cos
    return matrix


def get_sin_cos_for_rotate_matrix(a, b):
    if a == 0 and b == 0:
        return 0, 1
    return b / np.sqrt(a ** 2 + b ** 2), -a / np.sqrt(a ** 2 + b ** 2)


def qr_decompose(matrix: np.array):
    n = matrix.shape[0]
    Q = np.eye(n)
    R = matrix.copy()
    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            sin, cos = get_sin_cos_for_rotate_matrix(a, b)
            rotational_matrix = get_rotate_matrix(n, sin, cos, i, j)
            R = rotational_matrix.T @ R
            Q = Q @ rotational_matrix
    return Q, R
