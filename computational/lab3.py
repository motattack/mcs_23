import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Task:
    A: List[List[float]]
    b: List[float]


@dataclass
class Test(Task):
    x: List[float]


tests = [
    Test(
        [
            [81, -45, 45],
            [-45, 50, -15],
            [45, -15, 38]
        ],
        [531, -460, 193],
        [6, -5, -4]
    ),
    Test(
        [
            [6.25, -1, 0.5],
            [-1, 5, 2.12],
            [0.5, 2.12, 3.6]
        ],
        [7.5, -8.68, -0.24],
        [0.8, -2, 1]
    ),
    Test(
        [
            [1, 3, -2, 0, -2],
            [3, 4, -5, 1, -3],
            [-2, -5, 3, -2, 2],
            [0, 1, -2, 5, 3],
            [-2, -3, 2, 3, 4]
        ],
        [0.5, 5.4, 5.0, 7.5, 3.3],
        [-6.0978, -2.2016, -6.8011, -8.8996, 0.1998]
    ),
    Test(
        [
            [1, 2, 4],
            [2, 13, 23],
            [4, 23, 77],
        ],
        [10, 50, 150],
        [2.22, 0.55, 1.67]
    ),
]

tasks = [
    Task(
        [
            [5.8, 0.3, -0.2],
            [0.3, 4.0, -0.7],
            [-0.2, -0.7, 6.7]
        ],
        [3.1, -1.7, 1.1]
    ),
    Task(
        [
            [4.12, 0.42, 1.34, 0.88],
            [0.42, 3.95, 1.87, 0.43],
            [1.34, 1.87, 3.20, 0.31],
            [0.88, 0.43, 0.31, 5.17]
        ],
        [11.17, 0.115, 9.909, 9.349]
    )
]


def dim(matrix):
    if len(matrix[0]) == len(matrix):
        return len(matrix)
    else:
        raise Exception("Non-square matrix")


def cholesky(a):
    definite = True
    dim_a = dim(a)
    u = [[0] * dim_a for _ in range(dim_a)]

    try:
        u[0][0] = math.sqrt(a[0][0])

        for i in range(dim_a):
            for j in range(dim_a):
                if i == 0:
                    u[i][j] = a[i][j] / u[i][i]
                if i == j:
                    summa = 0
                    for k in range(i):
                        summa += u[k][i] ** 2

                    u[i][i] = math.sqrt(a[i][i] - summa)
                if i < j:
                    summa = 0
                    for k in range(i):
                        summa += u[k][i] * u[k][j]

                    u[i][j] = a[i][j] - summa
                    u[i][j] /= u[i][i]
    except ValueError as err:
        definite = False
        print(f'[{err}] это НЕ симметричные положительно определенная матрица')
    return u, definite


def transpose(matrix) -> List[list]:
    dim_a = len(matrix)
    new_matrix = [[matrix[j][i] for j in range(dim_a)] for i in range(dim_a)]
    return new_matrix


def main():
    print('-' * 16 + 'Тесты' + 16 * '-')
    for number, test in enumerate(tests):
        print(f'Тест №{number + 1}')

        u, definite = cholesky(test.A)
        if definite:
            ut = transpose(u)
            y = np.linalg.solve(ut, test.b)
            x = np.linalg.solve(u, y)

            print(f'Мой: {x.tolist()}')
            print(f'Реальный: {test.x}')

    print('-' * 16 + 'Ответы' + 16 * '-')
    for number, task in enumerate(tasks):
        u, definite = cholesky(task.A)
        if definite:
            ut = transpose(u)
            y = np.linalg.solve(ut, task.b)
            x = np.linalg.solve(u, y)

            print(f'СЛАУ №{number + 1}: {x.tolist()}')


if __name__ == "__main__":
    main()
