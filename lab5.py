import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(num_days, Pr, Pt):
    np.random.seed(42)
    robot_orders = np.random.poisson(Pr, num_days)
    traditional_orders = np.random.poisson(Pt, num_days)
    
    data = pd.DataFrame({'Day': range(1, num_days + 1),
                         'Robot_Orders': robot_orders,
                         'Traditional_Orders': traditional_orders})
    
    data['Total_Orders'] = data['Robot_Orders'] + data['Traditional_Orders']
    
    return data

def plot_comparison(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Day'], data['Robot_Orders'], label='Робот')
    plt.plot(data['Day'], data['Traditional_Orders'], label='Традиційні методи')
    plt.plot(data['Day'], data['Total_Orders'], label='Всього', linestyle='--', color='black')
    plt.title('Порівняння кількості порцій кави за дні')
    plt.xlabel('День')
    plt.ylabel('Кількість порцій')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_difference_histogram(data):
    plt.figure(figsize=(8, 5))
    plt.hist(data['Robot_Orders'] - data['Traditional_Orders'], bins=15, color='skyblue', edgecolor='black')
    plt.title('Гістограма різниці кількості порцій між роботом та традиційними методами')
    plt.xlabel('Різниця кількості порцій')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()

def calculate_statistics(data):
    mean_difference = np.mean(data['Robot_Orders'] - data['Traditional_Orders'])
    total_orders_robot = data['Robot_Orders'].sum()
    total_orders_traditional = data['Traditional_Orders'].sum()
    
    return mean_difference, total_orders_robot, total_orders_traditional

def perform_checks(data):
    # Перевірка, чи кількість порцій у робота більша за традиційні методи хоча б у 50% випадків
    success_rate = (data['Robot_Orders'] > data['Traditional_Orders']).mean()
    return success_rate

# Параметри для ефективності робота
Pr = 3  # продуктивність робота-асистента в порціях кави за годину
Pt = 2  # продуктивність традиційних методів виготовлення кави в порціях за годину

# Симуляція даних
num_days = 30
data = generate_data(num_days, Pr, Pt)

# Відображення таблички з результатами
print("Табличка з результатами:")
print(data)

# Графіки
plot_comparison(data)
plot_difference_histogram(data)

# Розрахунок статистики
mean_diff, total_robot, total_traditional = calculate_statistics(data)
print(f'Середня різниця: {mean_diff:.2f}')
print(f'Загальна кількість порцій робота: {total_robot}')
print(f'Загальна кількість порцій традиційні методи: {total_traditional}')

# Перевірки
success_rate = perform_checks(data)
print(f'Відсоток випадків, коли робот більш продуктивний: {success_rate * 100:.2f}%')
