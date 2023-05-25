import numpy as np
import matplotlib.pyplot as plt

# Задані параметри
A = 1.0  # Амплітуда синусоїди
n = 9  # Параметр n
phi = np.pi / 4  # Зсув по фазі
N = n * 100  # Кількість значень у послідовності


# Функція для генерації тестової послідовності з випадковими спотвореннями
def generate_sequence():
    x = np.linspace(0, 2, N)  # Інтервал x для вимірювання значень
    y_exact = A * np.sin(n * x + phi)  # Точне значення без спотворень
    deviation = np.random.uniform(-0.05 * A, 0.05 * A, N)  # Випадкове спотворення
    y = y_exact + deviation  # Загальна послідовність зі спотвореннями
    return x, y, y_exact


# Функції для обчислення середніх значень
def arithmetic_mean(sequence):
    return np.mean(sequence)


def harmonic_mean(sequence):
    return len(sequence) / np.sum(1 / sequence)


def geometric_mean(sequence):
    sequence = np.where(sequence <= 0, np.nan, sequence)  # Заміна недопустимих значень на NaN
    return np.nanprod(sequence) ** (1 / np.sum(~np.isnan(sequence)))


# Функція для виводу результату на екран у вигляді графіку
def plot_results(x, y, y_exact):
    plt.plot(x, y, label='Зі спотвореннями', color='darkorange')
    plt.plot(x, y_exact, label='Точне значення', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# Функція для обчислення точного значення
def exact_value(x):
    return A * np.sin(n * x + phi)


# Функція для порівняння наближеного значення з точним
def compare_values(approximate, exact):
    absolute_error = np.abs(approximate - exact)
    relative_error = absolute_error / np.abs(exact)
    return absolute_error, relative_error


# Головна частина програми
x, y, y_exact = generate_sequence()

# Обчислення середніх значень
arithmetic = arithmetic_mean(y)
harmonic = harmonic_mean(y)
geometric = geometric_mean(y)

# Вивід результатів
print('\nАрифметичне середнє:', arithmetic)
print('Гармонійне середнє:', harmonic)
print('Геометричне середнє:', geometric)

# Виведення графіку
plot_results(x, y, y_exact)

# Обчислення точного значення та порівняння з наближеним
exact = exact_value(x)
absolute_error, relative_error = compare_values(y, exact)

# Порівняння максимумів і мінімумів похибок
max_absolute_error = np.max(absolute_error)
min_absolute_error = np.min(absolute_error)
max_relative_error = np.max(relative_error)
min_relative_error = np.min(relative_error)

print('\nМаксимальна абсолютна похибка:', max_absolute_error)
print('Мінімальна абсолютна похибка:', min_absolute_error)
print('\nМаксимальна відносна похибка:', max_relative_error)
print('Мінімальна відносна похибка:', min_relative_error)
