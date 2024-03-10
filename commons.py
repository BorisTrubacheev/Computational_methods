import numpy as np
from task_1 import get_spectral_conditionality_number, get_volume_conditionality_number, get_angle_conditionality_number
import pandas as pd


class Matrix:
    def __init__(self, name: str, matrix: np.array, cond_numbers=None):
        self.name = name
        self.matrix = matrix
        self.conditional_numbers = cond_numbers


class MatrixEquation:
    def __init__(self, A: Matrix, x_create_mode: float):
        self.A = A
        if x_create_mode > 0:
            self.x = np.full(len(A.matrix), fill_value=x_create_mode)
        else:
            self.x = np.random.random(len(A.matrix))

        self.b = A.matrix @ self.x

    def print_equation(self):
        np.set_printoptions(linewidth=150)
        print('Исходная матрица:', '\n')
        print(self.A.name)
        print(self.A.matrix, '\n')
        print('Вектор b: ', self.b, '\n')
        print('Точное решение: ', self.x, '\n')


def print_results(values: list, labels: list[str]):
    pd.options.display.expand_frame_repr = False
    data = pd.DataFrame(data=np.array(values).T, columns=labels)
    print(data)


def get_hilbert_matrix(n: int) -> np.ndarray:
    m = np.ndarray((n, n))
    div = 1
    for i in range(n):
        for j in range(n):
            m[i][j] = 1 / (div + j)
        div += 1
    return m


def get_test_cases():
    np.random.seed(2024)
    test_matrices = [Matrix('random (4,4)', np.random.random((4, 4)) + np.random.randint(0, 100, (4, 4))),
                  Matrix('random (8,8)', np.random.random((8, 8)) + np.random.randint(0, 100, (8, 8))),
                  Matrix('diag (3,3)', np.array([[192, 0, 0],
                                                 [0, 0.123, 0],
                                                 [0, 0, 82]])),
                  Matrix('hilbert (3,3)', get_hilbert_matrix(3)),
                  Matrix('hilbert (7,7)', get_hilbert_matrix(7))]
    test_cases = []
    for m in test_matrices:
        test_cases.append(MatrixEquation(m, 2))

    return test_cases


def get_cond_numbers_for_matrix(matrix: np.array) -> np.array:
    cond_numbers = [get_spectral_conditionality_number(matrix),
                    get_volume_conditionality_number(matrix),
                    get_angle_conditionality_number(matrix)]
    return cond_numbers


def read_float(message: str) -> float:
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Incorrect input!")


def proceed(question: str) -> bool:
    while True:
        f = input(question + " (Да/нет)? ").lower()
        if  f == "" or f == "yes":
            print()
            return True
        elif f == "no":
            return False
        else:
            print("Ожидаемый ответ 'да' или 'нет'")