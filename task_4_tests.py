import numpy as np
from matplotlib import pyplot as plt
from commons import Matrix, MatrixEquation, get_hilbert_matrix, proceed, print_results, read_non_negative_int
from task_1 import get_spectral_conditionality_number
from task_4 import get_simple_iteration_solution, get_seidel_solution


def get_test_cases():
    np.random.seed(2024)
    test_matrices = [Matrix('random (4,4)', np.random.random((4, 4)) + np.random.randint(0, 100, (4, 4)) +
                     np.eye(4)*400),
                     Matrix('random (8,8)', np.random.random((8, 8)) + np.random.randint(0, 100, (8, 8)) +
                     np.eye(8)*800),
                     Matrix('diag (3,3)', np.array([[192, 0, 0],
                                                    [0, 0.123, 0],
                                                    [0, 0, 82]])),
                     Matrix('hilbert (3,3)', get_hilbert_matrix(3)),
                     Matrix('hilbert (4,4)', get_hilbert_matrix(4)),
                     Matrix('sparse (225, 225)', np.eye(225)*128.7 + np.eye(225, k=3)*4 + np.eye(225, k=-115)*42)]
    test_cases = []
    for m in test_matrices:
        test_cases.append(MatrixEquation(m, 2))

    return test_cases


def print_solutions_by_2_methods(equation: MatrixEquation, simple_iteration_solution, seidel_solution, eps):
    equation.print_equation()
    print('Используемая точность: ', eps)
    print('Решение методом простой итерации: ', simple_iteration_solution)
    print('Решение методом Зейделя: ', seidel_solution)


def main():
    while proceed('Хотите начать заново?'):
        test_cases = get_test_cases()
        case_names = []
        simple_iteration_solution_list = []
        simple_iteration_iteration_list = []
        simple_iteration_deviation_list = []

        seidel_solution_list = []
        seidel_iteration_list = []
        seidel_deviation_list = []
        matrix_cond_list = []
        eps_list = [1e-3, 1e-6, 1e-10]
        for case in test_cases:
            for eps in eps_list:
                case_names.append(case.A.name)
                matrix = case.A.matrix

                simple_iteration_solution, simple_iteration_iteration, _ = get_simple_iteration_solution(case, eps)
                simple_iteration_solution_list.append(simple_iteration_solution)
                simple_iteration_iteration_list.append(simple_iteration_iteration)

                seidel_solution, seidel_iteration, _ = get_seidel_solution(case, eps)
                seidel_solution_list.append(seidel_solution)
                seidel_iteration_list.append(seidel_iteration)

                simple_iteration_deviation_list.append(np.linalg.norm(case.x - simple_iteration_solution))
                seidel_deviation_list.append(np.linalg.norm(case.x - seidel_solution))
                matrix_cond_list.append(get_spectral_conditionality_number(matrix))

        for i in range(len(test_cases)):
            for eps in eps_list:
                print_solutions_by_2_methods(test_cases[i], simple_iteration_solution_list[i], seidel_solution_list[i], eps)

        proceed('Хотите посмотреть сводную таблицу эксперимента?')
        eps_list_for_print = eps_list * len(test_cases)

        print_results([case_names, matrix_cond_list,  simple_iteration_deviation_list, simple_iteration_iteration_list,
                       seidel_deviation_list, seidel_iteration_list, eps_list_for_print],
                      ['name', 'A_cond', 'simp_i_dev', 'simp_i_steps', 'seidel_dev', 'seidel_steps', 'eps'])

        numb = read_non_negative_int('Введите номер эксперимента, на сходимость которого вы хотели бы посмотреть: ')
        _, simple_iteration_steps, simple_iteration_precisions = \
            get_simple_iteration_solution(test_cases[numb//len(eps_list)], eps_list_for_print[numb])
        plt.plot(np.arange(simple_iteration_steps), simple_iteration_precisions)

        _, seidel_steps, seidel_precisions = get_seidel_solution(test_cases[numb //len(eps_list)], eps_list_for_print[numb])
        plt.plot(np.arange(seidel_steps), seidel_precisions)
        plt.show()


if __name__ == '__main__':
    main()
