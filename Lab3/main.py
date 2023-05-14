import numpy as np
import timeit
import matplotlib.pyplot as plt


# Швидке перетворення Фур'є за допомогою алгоритму Кулі-Тьюкі
def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi / n)
        return np.concatenate([even + factor * odd, even - factor * odd])


# Генерація випадкової послідовності
N = 19
x = np.random.rand(N)

# Доповнюємо вхідний сигнал нулями до степеня 2
M = 2 ** int(np.ceil(np.log2(N)))
x = np.concatenate([x, np.zeros(M - N)])

# Обчислення ШПФ та оцінка часу обчислення
start_time = timeit.default_timer()
X = fft(x)
elapsed_time = timeit.default_timer() - start_time

# Оцінка кількості операцій
add_operations = M * np.log2(N)
mult_operations = 3 * M * np.log2(N)

print(f"Час обчислення: {elapsed_time:.10f}с")
print(f"Кількість операцій додавання: {add_operations:.0f}")
print(f"Кількість операцій множення: {mult_operations:.0f}\n")
for i, c in enumerate(X):
    print(f'C_{i} = {c}')

# Обчислення спектра амплітуд та фаз та побудова графіків
amp_spectrum = np.abs(X)
phase_spectrum = np.angle(X)
freq_axis = np.arange(M)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(freq_axis, amp_spectrum, 'black')
plt.title('Амплітудний спектр')
plt.xlabel('Частота')
plt.ylabel('Амплітуда')

plt.subplot(1, 2, 2)
plt.stem(freq_axis, phase_spectrum, 'black')
plt.title('Фазовий спектр')
plt.xlabel('Частота')
plt.ylabel('Фаза')
plt.show()
