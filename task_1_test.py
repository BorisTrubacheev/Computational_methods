import numpy as np
import pandas as pd
from task_1 import get_spectral_conditionality_number, get_volume_conditionality_number, \
    get_angle_conditionality_number
from main import read_float, proceed


def get_hilbert_matrix(n: int) -> np.ndarray:
    m = np.ndarray((n, n))
    div = 1
    for i in range(n):
        for j in range(n):
            m[i][j] = 1 / (div + j)
        div += 1
    return m


def get_test_data():
    np.random.seed(2024)
    matrices = pd.Series({'random (4,4)': np.random.random((4, 4)) + np.random.randint(0, 100, (4, 4)),
                      'random (8,8)': np.random.random((8, 8)) + np.random.randint(0, 100, (8, 8)),
                      'diag (3,3)': np.array([[192, 0, 0],
                                              [0, 0.123, 0],
                                              [0, 0, 82]]),
                      'hilbert (3,3)': get_hilbert_matrix(3),
                      'hilbert (7,7)': get_hilbert_matrix(7)})
    x_values = get_x_values(matrices.values)
    b_values = get_b_values(matrices.values, x_values)
    test_data = pd.DataFrame(data={'matrix': matrices.values, 'x': x_values, 'b': b_values}, index=matrices.index)
    return test_data


def get_x_values(matrices: np.ndarray, vector_const=2) -> np.array:
    x_list = []
    for i in range(len(matrices)):
        m = matrices[i]
        x_list.append(np.full(len(m), fill_value=vector_const))
    return x_list


def get_b_values(matrices: np.ndarray, x_values: list) -> np.array:
    b_list = []
    for i in range(len(matrices)):
        m = matrices[i]
        x = x_values[i]
        b_list.append(np.matmul(m, x))
    return b_list


def main():
    np.set_printoptions(linewidth=150)
    pd.options.display.expand_frame_repr = False
    test_data = get_test_data()
    for name in test_data.index:
        print(name)
        print('Matrix:\n', test_data.loc[name, 'matrix'])
        print('x: ', test_data.loc[name, 'x'])
        print('b: ', test_data.loc[name, 'b'], end='\n'*3)

    test_df = pd.DataFrame({'test_name': [],
                            'spectral_cond': [],
                            'volume_cond': [],
                            'angle_cond': [],
                            'b_deviation': [],
                            'm_deviation': []})

    while proceed("Do you want enter deviations for matrix and b coefficient?"):
        b_deviation = read_float('Enter b coefficient deviation: ')
        m_deviation = read_float('Enter matrix deviation: ')
        for name in test_data.index:
            a = test_data.loc[name, 'matrix']
            test_df.loc[len(test_df)] = [name, get_spectral_conditionality_number(a),
                                         get_volume_conditionality_number(a), get_angle_conditionality_number(a),
                                         b_deviation, m_deviation]
        test_data['dev_matrix'] = test_data['matrix'] + m_deviation
        test_data['dev_b'] = test_data['b'] + b_deviation
        solutions = [np.linalg.solve(test_data.loc[name, 'dev_matrix'], test_data.loc[name, 'dev_b'])
                     for name in test_data.index]
        sol_deviations = [test_data['x'].values[i] - solutions[i] for i in range(len(solutions))]
        test_df['sol_deviation'] = [np.linalg.norm(dev) for dev in sol_deviations]
        print(test_df, end='\n'*4)

        if proceed('Do you want to see correlation matrix for this dataframe?'):
            print(test_df.corr('spearman'))


if __name__ == '__main__':
    main()
