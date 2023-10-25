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
            [2.1, -4.5, -2.0],
            [3.0, 2.5, 4.3],
            [-6.0, 3.5, 2.5]
        ],
        [19.07, 3.21, -18.25],
        [1.34025, -4.75798, 2.5771]
    ),
    Test(
        [
            [5, -1, 5],
            [-3, 6, 2],
            [10, -7, 0]
        ],
        [3.2, 5.4, -1.2],
        [0.7297, 1.2138, 0.1531]
    ),
    Test(
        [
            [5, 2, 3],
            [1, 6, 1],
            [3, -4, -2]
        ],
        [3, 5, 8],
        [2, 1, -3]
    ),
    Test(
        [
            [1, 2, 1, 4],
            [2, 0, 4, 3],
            [4, 2, 2, 1],
            [-3, 1, 3, 2]
        ],
        [13, 28, 20, 6],
        [3, -1, 4, 2]
    ),
    Test(
        [
            [2, 1, 3],
            [11, 7, 5],
            [9, 8, 4]
        ],
        [1, -6, -5],
        [-1, 0, 1]
    )
]

tasks = [
    Task(
        [
            [13.14, -2.12, 1.17],
            [-2.12, 6.3, -2.45],
            [1.17, -2.45, 4.6]
        ],
        [1.27, 2.13, 3.14]
    ),
    Task(
        [
            [4.31, 0.26, 0.61, 0.27],
            [0.26, 2.32, 0.18, 0.34],
            [0.61, 0.18, 3.20, 0.31],
            [0.27, 0.34, 0.31, 5.17]
        ],
        [1.02, 1.00, 1.34, 1.27]
    )
]


def dim(matrix):
    if len(matrix[0]) == len(matrix):
        return len(matrix)
    else:
        raise Exception("Non-square matrix")


def LU(matrix):
    dim_A = dim(matrix)
    u = [[0] * dim_A for _ in range(dim_A)]
    l = [[0] * dim_A for _ in range(dim_A)]

    for i in range(dim_A):
        for j in range(dim_A):
            l[i][i] = 1

            if i == 0:
                u[0][j] = matrix[0][j]
            if j == 0:
                l[i][0] = matrix[i][0] / u[0][0]

            if i <= j:
                summa = 0
                for k in range(0, i):
                    summa += l[i][k] * u[k][j]
                u[i][j] = matrix[i][j] - summa
            if i > j:
                summa = 0
                for k in range(0, j):
                    summa += l[i][k] * u[k][j]
                l[i][j] = 1 / u[j][j] * (matrix[i][j] - summa)
    return [l, u]


def main():
    print('-' * 16 + 'Тесты' + 16 * '-')
    for number, test in enumerate(tests):
        print(f'Тест №{number + 1}')

        l, u = LU(test.A)
        y = np.linalg.solve(l, test.b)
        x = np.linalg.solve(u, y)

        print(f'Мой: {x.tolist()}')
        print(f'Реальный: {test.x}')

    print('-' * 16 + 'Ответы' + 16 * '-')
    for number, task in enumerate(tasks):
        l, u = LU(task.A)
        y = np.linalg.solve(l, task.b)
        x = np.linalg.solve(u, y)

        print(f'СЛАУ №{number + 1}: {x.tolist()}')


if __name__ == "__main__":
    main()
