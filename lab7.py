import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def scalar_function_2(x):
    return x[0]**2 + x[1]**2


def scalar_function(x):
    return x**2 + 2*x + 2*np.cos(x)

def gradient_descent(initial_x, learning_rate, num_iterations, search_range):
    x = initial_x

    for _ in range(num_iterations):
        gradient = 2*x + 2 - 2*np.sin(x)  # Градієнт функції

        # Оновлення значення x за допомогою градієнтного спуску
        x = x - learning_rate * gradient

        # Обмеження області пошуку
        x = max(min(x, search_range[1]), search_range[0])

    return x, scalar_function(x)

# Початкова точка
initial_point = [0, 0]

# Область пошуку
bounds = [(-2, 2), (-2, 2)]

# Знаходження локалізованого мінімуму
result = minimize(scalar_function_2, initial_point, bounds=bounds)

# Виведення результату
print("Локалізований мінімум функції двох змінних знаходиться у точці:", result.x)
print("Значення функції у локалізованому мінімумі:", result.fun)


# Початкові значення
initial_x = 0.0
learning_rate = 0.1
num_iterations = 1000
search_range = [-4, 4]  

minimum_x, minimum_value = gradient_descent(initial_x, learning_rate, num_iterations, search_range)

print(f"Локалізований мінімум функції однієї змінної знаходиться при x = {minimum_x}, значення функції: {minimum_value}")


# Знаходження мінімуму для функції з однією змінною
result_1d = minimize(scalar_function, 0.0)
minimum_x_1d, minimum_value_1d = result_1d.x[0], result_1d.fun

# Побудова графіка для функції з однією змінною
x_values = np.linspace(-4, 4, 100)
y_values = scalar_function(x_values)

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Функція")
plt.scatter(minimum_x_1d, minimum_value_1d, color='green', label="Мінімум")
plt.title("Графік функції з однією змінною")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()



# Знаходження мінімуму для функції з двома змінними
result_2d = minimize(scalar_function_2, [0, 0])
minimum_x_2d, minimum_value_2d = result_2d.x, result_2d.fun

# Побудова тривимірного графіка для функції з двома змінними
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x_range, y_range)
z = scalar_function_2([x, y])

ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
ax.scatter(minimum_x_2d[0], minimum_x_2d[1], minimum_value_2d, color='green', s=100, label="Мінімум")
ax.set_title("Тривимірний графік функції з двома змінними")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.legend()
plt.show()

# Карта ліній рівня для функції з двома змінними
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x_range, y_range)
z = scalar_function_2([x, y])

plt.figure(figsize=(8, 6))
contour = plt.contour(x, y, z, levels=20, cmap='viridis')
plt.scatter(minimum_x_2d[0], minimum_x_2d[1], color='green', label="Мінімум")
plt.title("Карта ліній рівня функції з двома змінними")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(contour)
plt.legend()
plt.show()

