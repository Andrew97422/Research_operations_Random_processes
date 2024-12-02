import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


def SimplexMethod(simplex_table, extremum_sign, it=0):
    while True:
        raw_names = simplex_table.index
        inputs = []

        # условие оптимальности
        z_raw = simplex_table.iloc[0, :-1]
        a = None
        count = 0
        for i in range(len(z_raw)):
            if extremum_sign * z_raw[i] > 0:
                count += 1
                if a is None or extremum_sign * z_raw[i] > extremum_sign * a[1]:
                    a = [i, z_raw[i]]
        if count == 0:
            if any('R' in x for x in raw_names):
                print('\n' + 'Нет решения')
                return
            print('\n' + 'Решение')
            res = ''
            for j in simplex_table.index:
                if len(j) == 3 and int(j[1:3]) >= len(prices):
                    continue
                else:
                    print(f"{j} = {simplex_table.loc[j].loc['values']}")
                    #res.append(float(simplex_table.loc[j].loc['values']))
                    res += str(float(simplex_table.loc[j].loc['values'])) + ', '

            print(res)


            print(f'Число итераций: {it}')
            return simplex_table

        input_idx = a[0]

        solve_col = simplex_table.iloc[1:, -1]
        leader_col = simplex_table.iloc[:, input_idx]
        #print(input_idx, leader_col, solve_col)
        pos_values = []
        for i in range(len(leader_col[1:])):
            if leader_col[1:][i] != 0:
                if solve_col[i] / leader_col[1:][i] >= 0:
                    pos_values.append((i, solve_col[i] / leader_col[1:][i]))
        #print(pos_values)
        if len(pos_values) != 0:
            excluded_idx = min(pos_values, key=lambda x: x[1])[0]
        else:
            print('\n' + 'Нет решения')
            for j in simplex_table.index:
                print(f"{j} = {simplex_table.loc[j].loc['values']}")
            print(f'Число итераций: {it}')
            return simplex_table
        excluded_idx += 1
        inputs.append((input_idx, excluded_idx))

        leader_raw = simplex_table.iloc[excluded_idx] / simplex_table.iloc[excluded_idx, input_idx]
        simplex_table.iloc[excluded_idx] = leader_raw
        for i in range(len(simplex_table)):
            if i != excluded_idx:
                simplex_table.iloc[i] -= leader_col[i] * leader_raw

        for i in inputs:
            print(f'Включаемая переменная: {simplex_table.columns[i[0]]}\t исключаемая: {raw_names[excluded_idx]}')

        simplex_table = simplex_table.rename(index={raw_names[excluded_idx]: simplex_table.columns[input_idx]})
        #print(raw_names[excluded_idx], simplex_table.columns[input_idx])
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 4,
                               'display.width', 1000,
                               ):
            print(simplex_table)
        it += 1


def M_method(z, A, b, M, extremum_sign):
    z = -1 * z
    print(30 * '_' + 'Минимизация' + 30 * '_')

    max_cols = 0
    for row in A:
        max_cols = max(max_cols, len(row))
    A = np.pad(A, [(0, 0), (0, max_cols - len(row))], mode='constant')

    m = len(A)
    n = len(A[0])

    count = 0
    for i in A:
        if len(i) == n:
            count += 1

    R = np.full(count, -extremum_sign * M, dtype=float)
    z = np.concatenate([z, R])
    #print(z.shape, R.shape)

    for i in range(len(R)):
        A = np.column_stack((A, np.eye(1, m, i).T))
    simplex_matrix = np.column_stack((np.r_[[z], A], np.insert(b, 0, 0)))

    # согласование z-строки
    for i in range(1, m + 1):
        simplex_matrix[0] += extremum_sign * simplex_matrix[i] * M

    # названия столбцов
    vars = ['x' + str(i + 1) if z[i] != -extremum_sign * M else 'R' + str(i + 1) for i in range(len(z))] + ['values']

    idx = 0
    while vars[idx][0] != 'R':
        idx += 1
    idx = idx + 1

    raw_names = ['z']
    for i in range(m):
        raw_names.append('R' + str(idx + i))

    df = pd.DataFrame(simplex_matrix, columns=vars, index=raw_names)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 4,
                           'display.width', 1000,
                           ):
        print(df)

    return SimplexMethod(df, extremum_sign)


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

M = int(input("Введите M:"))

# Объединяем матрицы
A_extended = np.hstack((A, identity_with_alternate_signs))
print(z_extended.shape, A_extended.shape, b.shape)
M_method(z_extended, A_extended, b, M=M, extremum_sign=1)