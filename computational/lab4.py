# Householder
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
            [1, 2, 3],
            [4, 6, 7],
            [8, 9, 0]
        ],
        [6, 12, 24],
        [-11.538, 12.923, -2.769]
    ),
    Test(
        [
            [6.03, 13, -17],
            [13, 29.03, -38],
            [-17, -38, 50.03]
        ],
        [2.0909, 4.1509, -5.1191],
        [1.03, 1.03, 1.03]
    ),
]

tasks = [
    Task(
        [
            [2, 0, 1],
            [0, 1, -1],
            [1, 1, 1]
        ],
        [3, 0, 3]
    )
]


def get_H(matrix, k):
    n, _ = matrix.shape

    v = np.zeros(n)
    a = matrix[:, k]

    v[k] = a[k] + np.sign(a[k]) * np.linalg.norm(a[k:])
    for i in range(k + 1, n):
        v[i] = a[i]

    v = v[:, np.newaxis]
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)

    return H


def householder(matrix):
    n, m = matrix.shape
    Q = np.eye(n)
    R = np.copy(matrix)

    for k in range(m):
        H = get_H(R, k)
        Q = Q @ H
        R = H @ R

    return Q.T, R


def main():
    print('-' * 16 + 'Тесты' + 16 * '-')
    for number, test in enumerate(tests):
        print(f'Тест №{number + 1}')
        A = np.matrix(test.A)

        Q, R = householder(A)

        y = np.linalg.solve(Q.T, test.b)
        x = np.linalg.solve(R, y)

        print(f'Мой: {x}')
        print(f'Реальный: {test.x}')

    print('-' * 16 + 'Ответы' + 16 * '-')
    for number, task in enumerate(tasks):
        A = np.matrix(task.A)

        Q, R = householder(A)

        y = np.linalg.solve(Q.T, task.b)
        x = np.linalg.solve(R, y)

        print(f'СЛАУ №{number + 1}: {x}')


if __name__ == "__main__":
    main()
