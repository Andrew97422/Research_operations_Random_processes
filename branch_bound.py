import re
import warnings
from math import ceil, floor
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def print_simplex_table(simplex_table):
    # Функция для печати симплекс-таблицы
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 4,
                           ):
        print(simplex_table)


def SimplexMethod(simplex_table, extremum_sign, it=0):
    global solution
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
                print('\n' + 'Решение')
                res = ''
                z = 0
                for j in simplex_table.index:
                    if len(j) == 3 and int(j[1:3]) >= len(prices):
                        continue
                    else:
                        if j == 'z':
                            continue
                        print(f"{j} = {simplex_table.loc[j].loc['values']}")
                        # res.append(float(simplex_table.loc[j].loc['values']))

                        if len(j) == 3:
                            if int(j[1:3]) - 1 >= len(prices):
                                continue
                            z += float(simplex_table.loc[j].loc['values']) * prices[int(j[1:3])-1]
                            #print('j =', j, 'price =', prices[int(j[1:3])])
                        if len(j) == 2:
                            if int(j[1]) - 1 >= len(prices):
                                continue
                            z += float(simplex_table.loc[j].loc['values']) * prices[int(j[1])-1]
                            #print('j =', j, 'price =', prices[int(j[1])])
                        #print('z =', z)
                        res += str(float(simplex_table.loc[j].loc['values'])) + ', '
                        solution[j] = simplex_table.loc[j].loc['values']
                res = str(z) + ', ' + res
                print("z =", str(z))
                print(res)
                return
            print('\n' + 'Решение')
            res = ''
            for j in simplex_table.index:
                if len(j) == 3 and int(j[1:3]) >= len(prices):
                    continue
                else:
                    solution[j] = simplex_table.loc[j].loc['values']
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
            print('Нечего исключать')
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
    #A = np.concatenate((A, b[:-1]), axis=1)
    #b_for_A = np.insert(b, 0, 0)
    #A = np.concatenate((A, b_for_A), axis=1)

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



def DualSimplex(simplex_table):
    while True:
        # Условие допустимости
        neg_values = [(i + 1, j) for i, j in enumerate(simplex_table['values'][1:]) if j < 0]
        if not neg_values:
            return simplex_table
        else:
            excluded_idx = min([i for i in neg_values if i[1] < 0], key=lambda x: x[1])[0]

        # Условие оптимальности
        z_row = simplex_table.loc['z']
        excluded_row = simplex_table.iloc[excluded_idx]
        ratios_raw = [(i, abs(z_row[i] / excluded_row[i])) for i in range(simplex_table.shape[1] - 1) if
                      excluded_row[i] < 0]
        if ratios_raw:
            input_idx = min(ratios_raw, key=lambda x: x[1])[0]
        else:
            print('Нет решения в двойственной')
            return -1

        # МГЖ
        leader_col = simplex_table.iloc[:, input_idx]
        leader_raw = simplex_table.iloc[excluded_idx] / simplex_table.iloc[excluded_idx, input_idx]
        simplex_table.iloc[excluded_idx] = leader_raw
        for i in range(simplex_table.shape[0]):
            if i != excluded_idx:
                simplex_table.iloc[i] = simplex_table.iloc[i] - leader_col[i] * leader_raw

        simplex_table.rename(index={simplex_table.index[excluded_idx]: simplex_table.columns[input_idx]}, inplace=True)

def is_int(x, eps):
    return abs(int(x) - x) <= eps

def is_record(val, record, extremum_sign):
    if extremum_sign == 1:
        return val < record
    elif extremum_sign == -1:
        return val > record

def make_round(x, eps):
    if abs(ceil(x) - x) <= eps:
        return ceil(x)
    if x - floor(x) <= eps:
        return floor(x)
    s = re.findall(r'\d+[.]\d+[0]{5}', str(x))
    if s:
        return round(x, s[0].find('0' * 5))
    s = re.findall(r'\d+[.]\d+[9]{5}', str(x))
    if s:
        return round(x, s[0].find('9' * 5))
    else:
        return x

def BranchAndBoundsMethod(simplex_table, int_vars, extremum_sign, record, supremum, eps, iterations_limit, iteration=0, precision=None):
    if precision is not None: # если установлено precision, то значения в таблице симплекса округляются до указанной точности
        simplex_table = simplex_table.round(precision)

    simplex_table['values'] = simplex_table['values'].apply(make_round, eps=1e-6) # округляем таблицу симплекса до eps

    # Ищем переменные, которые ещё должны стать целыми
    new_record = False
    constraint_vars = [i for i in simplex_table.index[1:] if i in int_vars and not is_int(simplex_table['values'].loc[i], eps)]

    if constraint_vars:
        if iteration < iterations_limit:
            for i in constraint_vars:
                # Добавление ограничений для переменной i
                left_right_constraint = (floor(simplex_table['values'].loc[i]), ceil(simplex_table['values'].loc[i]))
                signs = [-1, 1]

                for j in range(2):
                    simplex_table_copy = simplex_table.copy()
                    new_str = []
                    for k in simplex_table_copy.loc[i][:-1]:
                        if k == 0 or k == 1:
                            new_str.append(0)
                        else:
                            new_str.append(signs[j] * k)
                    new_str.append(signs[j] * (simplex_table_copy['values'].loc[i] - left_right_constraint[j]))
                    new_str = [make_round(elem, eps) for elem in new_str]

                    # Добавление строки
                    new_var = "s"+ f"{len([var for var in simplex_table_copy.columns if var[0] == 's']) + 1}"
                    simplex_table_copy.loc[new_var] = new_str

                    # Добавление столбца
                    new_col = [0 if var!=new_var else 1 for var in simplex_table_copy.index]
                    simplex_table_copy.insert(simplex_table_copy.shape[1]-1, new_var, new_col, True)

                    # Решаем Двойственным Симплекс-методом
                    simplex_table_copy = DualSimplex(simplex_table_copy)
                    if type(simplex_table_copy) == pd.core.frame.DataFrame:
                        # Рекурсивный вызов МВГ для проверки дальнейших вариантов
                        simplex_table_copy = BranchAndBoundsMethod(simplex_table_copy, int_vars, extremum_sign, record, supremum, eps, iterations_limit, iteration+1, precision)
                        if type(simplex_table_copy) == pd.core.frame.DataFrame:

                            # Если новый результат - рекорд (лучший)
                            if is_record(simplex_table_copy['values'][0], record, extremum_sign):
                                # Проверяется соответствие найденного рекорда критериям оптимальности в соответствии с знаком экстремума
                                if extremum_sign == -1 and simplex_table_copy['values'][0] >= floor(supremum):
                                    return simplex_table_copy
                                elif extremum_sign == 1 and simplex_table_copy['values'][0] <= ceil(supremum):
                                    return simplex_table_copy
                                else:
                                    record = simplex_table_copy['values'][0]
                                    print(f'Лучший результат: {record}, итерация: {iteration}')
                                    record_table = simplex_table_copy
                                    new_record = True
                            # Если количество итераций достигло предельного значения, возвращается текущая симплекс-таблица
                            elif iteration >= iterations_limit:
                                return simplex_table_copy

            # Если найдено новое лучшее значение, возвращается таблица этого результата
            if new_record:
                return record_table
            # Если не найдено целочисленное решение и это была первая итерация, выводится сообщение "Нет целочисленного решения"
            else:
                if iteration == 0:
                    print('Нет целочисленного решения')
                    return None
                else:
                    return None
        else:
            return None
    else:
        return simplex_table


def print_res_vars(result):
    vars = ['x' + str(i) for i in range(1, len(prices) + 1)]
    for i in result.index:
        if i in vars:
            print(f'{i} = {result.loc[i][-1]}')
    print(f'z = {result.iloc[0][-1]}')


file_path = 'меню.xlsx'
data = pd.read_excel(file_path)

prices = data['цена'].values
calories = data['ккал'].values
proteins = data['белки'].values
fats = data['жиры'].values
carbs = data['углеводы'].values

'''
# Пример простых данных
prices = np.array([10, 20, 30])  # Цены на блюда
calories = np.array([100, 200, 300])  # Калории
proteins = np.array([10, 20, 30])  # Белки
fats = np.array([5, 10, 15])  # Жиры
carbs = np.array([30, 40, 50])  # Углеводы

# Пример ограничений
desired_calories = 500
desired_proteins = 50
desired_fats = 30
desired_carbs = 100
delta = 0.1  # 10%
'''

# Ввод данных для выбранных ограничений

desired_calories = float(input("Введите желаемое количество калорий: "))
desired_proteins = float(input("Введите желаемое количество белков: "))
desired_fats = float(input("Введите желаемое количество жиров: "))
desired_carbs = float(input("Введите желаемое количество углеводов: "))

# Ввод данных для процентных ограничений
delta = float(input("Введите процентное отклонение (дельту): ")) / 100

# Пересчет допусков в абсолютные значения
calorie_tolerance = desired_calories * delta
protein_tolerance = desired_proteins * delta
fat_tolerance = desired_fats * delta
carb_tolerance = desired_carbs * delta

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
diag_matrix = np.array([-1, 1] * (num_constraints // 2))
identity_with_alternate_signs = np.diag(diag_matrix)

M = 1000
solution = {}

# Объединяем матрицы
A_extended = np.hstack((A, identity_with_alternate_signs))
#simplex_table = M_method(z_extended, A_extended, b, M, extremum_sign=1, sign_list=['<=', '>='] * 4) #, sign_list=['<=', '>='] * 4
simplex_table = M_method(z_extended, A_extended, b, M, 1)
theM = simplex_table

if simplex_table is not None:
    supremum = simplex_table['values'][0]
    record = 10**10
    eps = 1e-6

    result = BranchAndBoundsMethod(simplex_table, ['x' + str(i) for i in range(1, len(prices) + 1)],
                                   1, record, supremum, eps, 10)

    #print_res_vars(result)
    if result is not None:
        print_simplex_table(simplex_table)
        print_res_vars(result)
    else:
        print('Нет целочисленного решения')
else:
    print('Нет решения')
