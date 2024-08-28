import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error

ITERATION = 1000
MU = 0.001
N = 7

def reg_lin():
    x = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04])
    y = np.array([0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.10, 0.14, 0.53, 0.71, 1.53])
    A = np.zeros((1, N + 1))

    cost_vector_point = np.zeros(ITERATION)
    predict = np.zeros(len(x))

    for j in range(ITERATION):

        L = 0
        dLda = np.zeros(A.shape)

        for i in range(len(x)):
            x_vector = [x[i] ** j for j in range(N + 1)]
            yhat = A @ np.transpose(x_vector)
            L += (yhat - y[i]) ** 2
            dLda += 2 * (yhat - y[i]) * x_vector

        A = A - MU * dLda
        cost_vector_point[j] = L

    T = 100
    interval = np.linspace(-1.25, 1.25, T)

    ytest = np.zeros(T)
    for i in range(T):
        # ytest[i] = interval[i] * A[0][1] + A[0][0]
        for j in range(len(A[0])):
            ytest[i] += interval[i] ** j * A[0][j]

    plt.figure(1)
    plt.scatter(x, y)
    plt.plot(interval, ytest)
    plt.show()


def inv_matrix_4():
    A = np.array([[2, 1, 1, 2], [1, 2, 3, 2], [2, 1, 1, 2], [3, 1, 4, 1]])
    B = np.zeros((4, 4), dtype=np.float32)
    I = np.identity(4, dtype=np.float32)

    cost_vector = np.zeros((ITERATION,), dtype=np.float32)

    for i in range(ITERATION):
        L = np.sum((B @ A - I) ** 2)
        dLdB = 2 * (B @ A - I) @ np.transpose(A)
        B = B - MU * dLdB
        cost_vector[i] = L

    plt.figure(1)
    plt.plot(cost_vector)
    plt.grid()
    plt.show()

    print(B @ A)

def inv_matrix_6():
    A = np.array([[3, 4, 1, 2, 1, 5], [5, 2, 3, 2, 2, 1], [6, 2, 2, 6, 4, 5], [1, 2, 1, 3, 1, 2], [1, 5, 2, 3, 3, 3], [1, 2, 2, 4, 2, 1]])
    B = np.zeros((6, 6), dtype=np.float32)
    I = np.identity(6, dtype=np.float32)

    cost_vector = np.zeros((ITERATION,), dtype=np.float32)

    for i in range(ITERATION):
        L = np.sum((B @ A - I) ** 2)
        dLdB = 2 * (B @ A - I) @ np.transpose(A)
        B = B - MU * dLdB
        cost_vector[i] = L

    plt.figure(1)
    plt.plot(cost_vector)
    plt.grid()
    plt.show()

    print(B @ A)

def inv_matrix_3():
    A =  np.array([[3, 4, 1], [5, 2, 3], [6, 2, 2]])
    B = np.zeros((3,3), dtype=np.float32)
    I = np.identity(3, dtype=np.float32)

    cost_vector = np.zeros((ITERATION,), dtype=np.float32)

    for i in range(ITERATION):
        L = np.sum((B@A - I) ** 2)
        dLdB = 2 * (B@A - I) @ np.transpose(A)
        B = B - MU * dLdB
        cost_vector[i] = L

    plt.figure(1)
    plt.plot(cost_vector)
    plt.grid()
    plt.show()

    print(B@A)


if __name__ == '__main__':
    reg_lin()
