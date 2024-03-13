import numpy as np
from commons import MatrixEquation


def get_simple_iteration_solution(equation: MatrixEquation, eps=10 ** (-5)) -> np.array:
    A = equation.A.matrix
    b = equation.b
    n = len(A)
    precisions_list = []

    diagonal_predominance = True
    for i in range(n):
        if np.sum(np.abs(A[i])) >= 2 * np.abs(A[i][i]):
            diagonal_predominance = False

    positive_definite = (np.allclose(A, A.T) and min(np.linalg.eig(A)[0]) >= 0)

    if diagonal_predominance:
        B = np.zeros((n, n))
        c = b.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    B[i, j] = -(A[i, j] / A[i, i])
                else:
                    B[i, j] = 0
            c[i] = c[i] / A[i, i]
    elif positive_definite:
        eigen_value = np.linalg.eig(A)[0]
        min_eig = min(eigen_value)
        max_eig = max(eigen_value)
        alfa = 2 / (min_eig + max_eig)
        B = np.eye(n) - alfa * A
        c = alfa * b
    else:
        print('Метод не может быть применён, т.к. матрица должна быть положительно определённой или '
              'с диагональным преобладанием')
        return

    x = c.copy()
    diff = np.inf
    iteration_number = 0
    while abs(diff) >= eps:
        '''if iteration_number % 1000 == 0:
            print(iteration_number)'''
        precisions_list.append(np.linalg.norm(equation.x - x))
        previous_x = x
        x = B.dot(x) + c
        iteration_number += 1
        diff = np.linalg.norm(B) * np.linalg.norm(x - previous_x) / (1 - np.linalg.norm(B))
    return x, iteration_number, precisions_list


def L_D_R_decomposition(A: np.array):
    D = np.tril(np.triu(A))
    L = np.tril(A) - D
    R = np.triu(A) - D
    return L, D, R


def get_seidel_solution(equation: MatrixEquation, epsilon=10 ** (-5)) -> np.array:
    A = equation.A.matrix
    b = equation.b
    precisions_list = []

    L, D, R = L_D_R_decomposition(A)
    B = (-1 * np.linalg.inv(D + L)) @ R
    c = np.linalg.inv(D + L) @ b

    x = c.copy()
    diff = np.inf
    iteration_number = 0
    while abs(diff) >= epsilon:
        precisions_list.append(np.linalg.norm(equation.x - x))
        previous_x = x
        x = B.dot(x) + c
        iteration_number += 1
        diff = np.linalg.norm(B) * np.linalg.norm(x - previous_x) / (1 - np.linalg.norm(B))
    return x, iteration_number, precisions_list
