import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Задана функція
def f(x):
    return np.cos(x) + x

# Локальна область
bounds = [(-np.pi, np.pi)]

# Знаходження локального мінімуму
result_min = minimize(f, 0.0, bounds=bounds)
local_minimum_x, local_minimum_value = result_min.x[0], result_min.fun

# Знаходження локального максимуму
result_max = minimize(lambda x: -f(x), 0.0, bounds=bounds)
local_maximum_x, local_maximum_value = result_max.x[0], -result_max.fun

# Виведення результатів
print(f"Локальний мінімум: x = {local_minimum_x}, f(x) = {local_minimum_value}")
print(f"Локальний максимум: x = {local_maximum_x}, f(x) = {local_maximum_value}")

# Побудова графіку
x_values = np.linspace(-np.pi, np.pi, 1000)
y_values = f(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="$f(x) = \cos(x) + x$")
plt.scatter([local_minimum_x, local_maximum_x], [local_minimum_value, local_maximum_value], color='red', label='Локальний мінімум та максимум')
plt.title("Графік функції з локальним мінімумом та максимумом")
plt.xlabel("x")
plt.ylabel("$f(x)$")
plt.legend()
plt.grid(True)
plt.show()
