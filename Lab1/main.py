import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

N = 10
lb = -np.pi
rb = np.pi
lb_graph = -3 * np.pi
rb_graph = 3 * np.pi
l = (rb - lb) / 2
x = np.linspace(lb, rb, 1000)


def f(x):
    return 9 * np.sin(9 * np.pi * x)


def integral_expression(x, k):
    return f(x) * np.sin(np.pi * k * x / l)


def plot_function():
    fig, ax = plt.subplots()
    plt.plot(x, f(x))
    ax.set_title('Графік функції 9*sin(9*x*pi) в межах ({}, {})'.format(lb_graph.__round__(2), rb_graph.__round__(2)))
    ax.grid()
    plt.show()


# Обчислення коефіцієнтів ряду Фур'є
def fourier_coefficients(N):
    a_0 = 0
    a_k = np.zeros(N)
    b_k = np.zeros(N)

    for k in range(1, N + 1):
        b_k[k - 1] = 2 / l * quad(integral_expression, 0, l, args=(k,))[0]

    print("Обчислення значень a_k та b_k :")
    print("a_k:", a_k)
    print("b_k:", b_k)
    return a_0, a_k, b_k


# Обчислення наближення функції f(x) рядом Фур'є з точністю до порядку N
def fourier_series(N, b_k, x_i=x):
    sum = 0
    for k in range(1, N + 1):
        sum += b_k[k - 1] * np.sin(np.pi * k * x_i / l)
    return sum


def plot_approximation(N, b_k):
    fig, ax = plt.subplots()
    ax.set_title('Графік апроксимації в межах ({}, {})'.format(lb_graph.__round__(2), rb_graph.__round__(2)))

    for n in range(1, N + 1):
        y = fourier_series(n, b_k)
        plt.plot(x, y, label=f"N={n}")

    plt.legend(loc='upper right', fontsize='xx-small')
    ax.grid()
    plt.show()


def plot_harmonics(a_k, b_k):
    fig, axs = plt.subplots(2, figsize=(8, 8))
    axs[0].stem([0] + a_k)
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('a(k)')
    axs[0].set_title('Графік гармонік для a_k')
    axs[1].set_xlabel('k')
    axs[1].stem(range(1, N + 1), b_k)
    axs[1].set_ylabel('b(k)')
    axs[1].set_title('Графік гармонік для b_k')
    axs[1].set_xticks(range(1, N + 1))
    axs[1].set_xticklabels([f'{i}' for i in range(1, N + 1)])
    plt.tight_layout()
    plt.show()


def approximation_relative_error(N, b_k):
    # Обчислення відносної похибки в кожній точці
    y_approx = [fourier_series(N, b_k, x_i) for x_i in x]
    y_exact = [f(x_i) for x_i in x]
    relative_error = []
    for i in range(0, len(x)):
        relative_error.append(np.abs((y_approx[i] - y_exact[i]) / (y_exact[i] + 1e-10)))

    # Побудова графіка відносних похибок
    fig, ax = plt.subplots()
    ax.plot(x, relative_error)
    ax.set_title('Похибка наближення')
    ax.grid()
    plt.show()
    return relative_error


plot_function()
a_0, a_k, b_k = fourier_coefficients(N)
plot_approximation(N, b_k)
plot_harmonics(a_k, b_k)
relative_error = approximation_relative_error(N, b_k)

# Збереження у файл
with open('fourier_series_results.txt', 'w') as file:
    file.write('N = {}\n'.format(N))
    file.write('a_0 = {}\n'.format(a_0))
    file.write('a_k = {}\n'.format(a_k))
    file.write('b_k = {}\n'.format(b_k))
    file.write('Похибка наближення = {}\n'.format(relative_error))
