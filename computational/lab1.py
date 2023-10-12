# v7

import os
import random
import pickle
from dataclasses import dataclass
import numpy as np

n = 25


@dataclass
class Matrix:
    id: int
    elements: []


def save_matrices(matrices, filename):
    with open(filename, 'wb') as file:
        pickle.dump(matrices, file)


def load_matrices(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def create_matrix(matrix_id, rows=n, columns=n):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(random.uniform(0, 1) / 10000)
        matrix.append(row)
    return Matrix(matrix_id, matrix)


def make_or_load_matrices():
    matrices = []

    if not os.path.exists('matrices.pkl'):
        for k in range(1, 5 + 1):
            matrix = create_matrix(k)
            matrices.append(matrix)

        save_matrices(matrices, 'matrices.pkl')
    else:
        matrices = load_matrices('matrices.pkl')

    return matrices


def norma(matrix):
    maximum_sum = 0
    for i in range(1, n + 1):
        cur_sum = 0
        for j in range(1, n + 1):
            cur_sum += matrix.elements[i - 1][j - 1]
        maximum_sum = max(cur_sum, maximum_sum)
    return maximum_sum


def find_inverse(matrix):
    try:
        matrix_array = np.array(matrix.elements)

        inverse_matrix_array = np.linalg.inv(matrix_array)

        inverse_elements = inverse_matrix_array.tolist()

        return Matrix(matrix.id, inverse_elements)
    except np.linalg.LinAlgError:
        print(f"Матрица {matrix.id} не имеет обратной величины.")
        return None


def cond(matrix, inverse):
    return norma(matrix) * norma(inverse)


def wandermonize(vector, rows=n + 1, columns=n + 1):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(0)
        matrix.append(row)

    for i in range(rows):
        matrix[i][0] = 1

    for j, x in enumerate(vector):
        for i in range(2, rows + 1):
            matrix[j][i - 1] = x ** (i - 1)

    return Matrix(9, matrix)


def norma_sum(vector):
    summa = 0
    for x_i in vector:
        summa += abs(x_i)
    return summa


def cond_sum(matrix, inverse):
    return norma_sum(matrix) * norma_sum(inverse)


def main():
    matrices = make_or_load_matrices()

    print('k\t|\tnorma\t\t\t\t\t|\tcond\t')
    for matrix in matrices:
        inverse_matrix = find_inverse(matrix)
        print(matrix.id, norma(matrix), cond(matrix, inverse_matrix), sep='\t|\t')

    x = [0.1 + (1 - 0.1) / (n) * (i) for i in range(n + 1)]
    wandermont = wandermonize(x)

    y = [sum(line) for line in wandermont.elements]

    x_ans = np.linalg.solve(np.array(wandermont.elements), np.array(y))
    print("Моё решение:")
    print(x_ans.tolist())

    x_real = [1] * (n + 1)
    print("Точное решение:")
    print(x_real)

    print("Число обусловленности матрицы Вандермонда")
    print(cond(wandermont, find_inverse(wandermont)))
    print("// число обусловленности большое => все плохо :d")

    vector1 = np.array(x_ans)
    vector2 = np.array(x_real)
    raz = (vector1 - vector2).tolist()

    print("Норма разницы двух решений:")
    print(norma_sum(raz))


if __name__ == "__main__":
    main()
