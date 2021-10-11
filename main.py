import math
import numpy as np


def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca.
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca
    """
    return 2 * math.pi * r * (r + h)


np.arrange(0, 8, 0.5)
np.linspace(4, 5, 10)


def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    fibo = np.array(n * [])
    fibo[0] = 0
    fibo[1] = 1
    for i in range(0, n):
        if i > 1:
            fibo[i] = fibo[i - 1] + fibo[i - 2]
    return fibo


def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej
    na podstawie parametru a.
    Szczegółowy opis w zadaniu 4.

    Parameters:
    a (float): wartość liczbowa

    Returns:
    touple: krotka zawierająca wyniki obliczeń
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    mat = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])

    return np.inv(mat), np.transpose(mat), np.linalg.det(mat)


matrix = np.array([[3, 1, -2, 4], [0, 1, 1, 5], [-2, 1, 1, 6], [4, 3, 0, 1]])
print(matrix[0, 0], matrix[2, 2], matrix[2, 1])
w1 = matrix[:, 2]
w2 = matrix[1, :]


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie
    z opisem zadania 7.

    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    p = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            if i > j:
                p[i, j] = i
            else:
                p[i, j] = j

    return p


print(custom_matrix(4, 4))

v1 = np.array([[1], [3], [13]])
v2 = np.array([[85, -2]])

print(np.multiply(4, v1))
print(np.multiply(-1, v2) + np.multiply(2, np.ones((3, 1))))
print(np.dot(v1, v2))
print(np.multiply(v1, v2))

m1 = np.array([[1, -7, 3], [-12, 3, 4], [5, 13, -3]])
print(np.multiply(3, m1))
print(np.multiply(3, m1)+np.ones((3, 3)))
print(np.transpose(m1))
print(np.dot(m1, v1))
print(np.dot(np.transpose(v2), m1))