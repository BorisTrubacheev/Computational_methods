import numpy as np
from commons import MatrixEquation, print_results, proceed, Matrix, get_hilbert_matrix
from task_1 import get_spectral_conditionality_number
from task_3 import qr_decompose


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


def print_qr_decomposition(equation: MatrixEquation, Q, R, x):
    equation.print_equation()
    print('Матрица Q:')
    print(Q, '\n')
    print('Матрица R:')
    print(R, '\n')
    print('Вычисленное решение: ', x)


def main():
    test_cases = get_test_cases()
    case_names = []
    computed_solution_list = []
    deviation_list = []
    q_matrix_list = []
    r_matrix_list = []
    matrix_cond_list = []
    q_cond_list = []
    r_cond_list = []
    for case in test_cases:
        case_names.append(case.A.name)
        matrix = case.A.matrix
        Q, R = qr_decompose(matrix)
        q_matrix_list.append(Q)
        r_matrix_list.append(R)
        solution = np.linalg.solve(R, Q.T @ case.b)
        computed_solution_list.append(solution)
        deviation_list.append(np.linalg.norm(case.x - solution))
        matrix_cond_list.append(get_spectral_conditionality_number(matrix))
        q_cond_list.append(get_spectral_conditionality_number(Q))
        r_cond_list.append(get_spectral_conditionality_number(R))

    for i in range(len(test_cases)):
        print_qr_decomposition(test_cases[i], q_matrix_list[i], r_matrix_list[i], computed_solution_list[i])

    proceed('Хотите посмотреть числа обусловленности матриц?')

    print_results([case_names, matrix_cond_list, q_cond_list, r_cond_list, deviation_list],
                  ['name', 'A_cond', 'Q_cond', 'R_cond', 'x_deviation'])


if __name__ == '__main__':
    main()
