import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Загрузка данных из Excel
file_path = 'меню.xlsx'
data = pd.read_excel(file_path)

prices = data['цена'].values
calories = data['ккал'].values
proteins = data['белки'].values
fats = data['жиры'].values
carbs = data['углеводы'].values

# Интервалы
calorie_tolerance = 20
protein_tolerance = 5
fat_tolerance = 5
carb_tolerance = 5

# Ввод данных для выбранных ограничений
desired_calories = float(input("Введите желаемое количество калорий: "))
desired_proteins = float(input("Введите желаемое количество белков: "))
desired_fats = float(input("Введите желаемое количество жиров: "))
desired_carbs = float(input("Введите желаемое количество углеводов: "))

# Создаем списки для A и b
A_list = []
b_list = []

# Добавляем выбранные ограничения в A и b
A_list.extend([calories, calories])
b_list.extend([desired_calories + calorie_tolerance, desired_calories - calorie_tolerance])
A_list.extend([proteins, proteins])
b_list.extend([desired_proteins + protein_tolerance, desired_proteins - protein_tolerance])
A_list.extend([fats, fats])
b_list.extend([desired_fats + fat_tolerance, desired_fats - fat_tolerance])
A_list.extend([carbs, carbs])
b_list.extend([desired_carbs + carb_tolerance, desired_carbs - carb_tolerance])

# Преобразуем списки в матрицы
A = np.array(A_list)
b = np.array(b_list)

# Расширение целевой функции z
num_constraints = len(b)
z_extended = np.concatenate((prices, np.zeros(num_constraints)))

# Строим диагональную матрицу для добавления
diag_matrix = np.array([1, -1] * (num_constraints // 2))
identity_with_alternate_signs = np.diag(diag_matrix)

# Объединяем матрицы
A_extended = np.hstack((A, identity_with_alternate_signs))

# Решение задачи с помощью linprog
result = linprog(c=z_extended, A_eq=A_extended, b_eq=b, method='highs')

# Проверка и вывод результата
if result.success:
    print("\nОптимальное решение найдено:")
    print(f"Минимальная стоимость: {result.fun:.2f}")
    print("\nКоличество каждого блюда:")

    count = 0
    res = ''
    for i, qty in enumerate(result.x[:len(prices)]):
        if qty > 0:
            print(f"  Блюдо {i+1}: {qty:.4f} ед.")
            count += 1
            res += f", {qty:.2f}"
    print("Общий результат: " + str(result.fun) + res)
    # Проверка итоговых КБЖУ
    total_calories = np.dot(calories, result.x[:len(prices)])
    total_proteins = np.dot(proteins, result.x[:len(prices)])
    total_fats = np.dot(fats, result.x[:len(prices)])
    total_carbs = np.dot(carbs, result.x[:len(prices)])

    print("\nИтоговые значения:")
    print(f"  Калории: {total_calories:.2f}")
    print(f"  Белки: {total_proteins:.2f}")
    print(f"  Жиры: {total_fats:.2f}")
    print(f"  Углеводы: {total_carbs:.2f}")
else:
    print("Решение невозможно:", result.message)
