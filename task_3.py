import numpy as np
from commons import get_hilbert_matrix


def get_single_vector(length: int, position: int):
    e = np.zeros(length)
    e[position - 1] += 1
    return e


def get_reflection_matrix(vect: np.array):
    e = get_single_vector(len(vect), 1)
    x = (vect - np.linalg.norm(vect, 2) * e)
    y = np.linalg.norm(vect - np.linalg.norm(vect, 2) * e, 2)
    x = x / y
    I = np.eye(len(vect))
    y = x[:, np.newaxis]
    print(x[np.newaxis, :].shape)
    z = np.matmul(y, x[np.newaxis, :])
    return I - 2 * z


def get_block_diagonal_matrix(m1: np.array, m2: np.array):
    res = np.zeros((len(m1) + len(m2), len(m1) + len(m2)))
    res[:len(m1), :len(m1)] = m1
    res[-len(m2):, -len(m2):] = m2
    return res


def get_qr_matrix_decomposition(matrix: np.array):
    Q = np.eye(len(matrix))
    for i in range(len(matrix)):
        U = get_reflection_matrix(matrix[i:, i])
        U1 = get_block_diagonal_matrix(np.eye(i), U)
        Q = np.matmul(U1, Q)
        matrix[i:, i:] = np.matmul(U, (matrix[i:, i:]))

    print(Q)
    print(matrix)
    return


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
