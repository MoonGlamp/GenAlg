from pyeasyga import pyeasyga
import random


# ----------------------------------------------------------------
# считывает данные из файла построчно, разделяя
# ----------------------------------------------------------------
def ReadFile(filename):
    f = open(filename, "r")
    table = [line.split() for line in f]
    f.close()
    return table


# ----------------------------------------------------------------
# возвращает грузоподъемность,10*вместимость
# (из первой строки считанного файла)
# ----------------------------------------------------------------
# объем умножает на 10,  чтобы работать с целыми числами
# ----------------------------------------------------------------
def GetMax(table):
    return int(table[0][0]), 10 * int(table[0][1])


# ----------------------------------------------------------------
# возвращает список кортежей - предметы (вес, объем, ценность)
# (начиная со второй строки считанного файла)
# ----------------------------------------------------------------
# объем умножает на 10,  чтобы работать с целыми числами
# ----------------------------------------------------------------
def GetData(table):
    n_obj = len(table) - 1
    data = [(int(table[i][0]), int(float(table[i][1]) * 10), int(table[i][2])) for i in range(1, n_obj + 1)]
    return data


# ----------------------------------------------------------------
# возвращает характеристики особи (рюкзака) - суммарные вес, объем, ценность
# ----------------------------------------------------------------
def GetAll(individual, data):
    weight, volume, price = 0, 0, 0
    for (selected, item) in zip(individual, data):
        if selected:
            weight += item[0]
            volume += item[1]
            price += item[2]
    return weight, volume, price


# ----------------------------------------------------------------
# fitness-функция (функция приспособленности особи)
# возвращает ценность рюкзака
# 0 для пустого рюкзака или переполненного
# ----------------------------------------------------------------
def Fitness(individual, data):
    weight, volume, price = GetAll(individual, data)
    if weight > g_w or volume > g_v:
        price = 0
    return price


# ----------------------------------------------------------------
# функция проверки входных данных
# >0 : полностью заполненный рюкзак является решением (возвращается ценность)
#  0 : нет решений задачи
# -1 : данные корректны
# ----------------------------------------------------------------
def CheckData(data):
    all_1 = [1] * len(data)
    ret = Fitness(all_1, data)
    if ret == 0:
        for item in data:
            if item[0] <= g_w and item[1] <= g_v:
                ret = -1
                break
    return ret


# ----------------------------------------------------------------
# 1.  Начальная популяция – кол-во особей всегда = 200 (POP_SIZE):
# 1.1 случайная генерация
# ----------------------------------------------------------------
# выбирает особей со случайным набором бит и ненулевой ценностью
def InitPop(data):
    n_obj = len(data)

    pop_out = [None] * POP_SIZE
    for i in range(POP_SIZE):
        while 1:
            individual = [random.randint(0, 1) for i in range(n_obj)]
            price = Fitness(individual, data)
            if price > 0:
                pop_out[i] = individual
                break
    return pop_out


# ----------------------------------------------------------------
# 2.  Отбор особей для скрещивания:
# 2.1 выбор каждой особи пропорционально приспособленности (рулетка)
# ----------------------------------------------------------------
def Roulette(pop, data):
    # отбор с ненулевой фитнесс-функцией - отбираем только решения
    potential = [individual for individual in pop if Fitness(individual, data) != 0]
    pot_fitness = [Fitness(individual, data) for individual in potential]
    n_potential = len(pot_fitness)

    min_fitness = min(pot_fitness)
    right_bounds = [0] * n_potential
    right_bounds[0] = pot_fitness[0] / min_fitness
    # делит рулетку (отрезок) на сектора (отрезки длиной, пропорциональной фитнесс-функции)
    for i in range(1, n_potential):
        right_bounds[i] = right_bounds[i - 1] + pot_fitness[i] / min_fitness

    choosen = [None] * PARENT_SIZE
    for j in range(PARENT_SIZE):
        rndm_v = random.random() * right_bounds[n_potential - 1]
        for i in range(n_potential):
            if (rndm_v <= right_bounds[i]):
                choosen[j] = potential[i]
                break

    return choosen


# ----------------------------------------------------------------
# � еализация многоточечного кроссинговера с 3мя точками для 2 особей
# ----------------------------------------------------------------
def Crossingover(individual_1, individual_2):
    n_obj = len(individual_1)
    # выбираем 3 средние точки
    points = random.sample([i + 1 for i in range(n_obj - 2)], 3)
    points.sort()

    individual_1[0:points[0]], individual_2[0:points[0]] = individual_2[0:points[0]], individual_1[0:points[0]]
    individual_1[points[1]:points[2]], individual_2[points[1]:points[2]] = individual_2[
                                                                           points[1]:points[2]], individual_1[
                                                                                                 points[1]:points[2]]

    return individual_1, individual_2


# ----------------------------------------------------------------
# 3. Скрещивание (кроссинговер) между выбранными особями. Каждая особь
#    скрещивается 1 раз за 1 поколение, 1 пара дает 2 потомка:
# 3.1 многоточечный с 3мя точками
# ----------------------------------------------------------------
def GetChilds(parents):
    n_parents = len(parents)
    childs = [None] * n_parents
    parent_list = [i for i in range(n_parents)]
    for i in range(0, n_parents, 2):
        two = random.sample(parent_list, 2)
        p1 = parents[two[0]][:]  # поверхностная копия!!! - иначе Crossingover меняет текущую популяцию
        p2 = parents[two[1]][:]
        childs[i], childs[i + 1] = Crossingover(p1, p2)
        parent_list.remove(two[0])
        parent_list.remove(two[1])

    return childs


# ----------------------------------------------------------------
# 4. Мутация:
# 4.3 добавление 1 случайной вещи 10% особей
# ----------------------------------------------------------------
def Mute(pop):
    n_pop = len(pop)
    n_obj = len(pop[0])

    # выбираем 10% индексов
    chousen_10p = random.sample([i for i in range(n_pop)], int(n_pop / 10))
    all_1 = [1] * n_pop
    for i in range(n_pop):
        if (i in chousen_10p) and (pop[i] != all_1):
            # если индекс в группе избранных и возможно добавление вещи
            # выбираем случайную вещь из отсутствующих в рюкзаке, добавляем ее
            not_1 = [j for j in range(n_obj) if pop[i][j] == 0]  # список отсутствующих вещей
            chousen_1 = random.sample(not_1, 1)
            pop[i][chousen_1[0]] = 1

    return pop


# ----------------------------------------------------------------
# 5. Формирование новой популяции (кол-во особей - константа)
# 5.1 замена не более 30% худших особей на потомков
# ----------------------------------------------------------------
def GetNewGen(old, new, data):
    n_old = len(old)
    n_obj = len(old[0])
    n_30p = int(n_old * 0.3)  # 30%

    new = sorted(new, key=lambda ind: -Fitness(ind, data))
    choosen = new[0:n_30p]  # отбираем лучших потомков в количестве 30% от популяции поколения

    old = old + choosen  # добавляем выбранных потомков к текущему поколению
    old = sorted(old, key=lambda ind: -Fitness(ind, data))

    old = old[:-(n_30p)]  # убираем худших (количество убранных = количеству добавленных)

    return old


# ----------------------------------------------------------------
# 6. Оценка результата
# Наступила сходимость (функция приспособленности лучшей особи в популяциях
# отличается не более, чем на 10%)
# или прошло 100 поколений - проверка в main
# ----------------------------------------------------------------
def Result(max_prev, max_current):
    return (1.1 * max_prev < max_current)


# ----------------------------------------------------------------
# ----------------------------------------------------------------
POP_SIZE = 200  # Начальная популяция – кол-во особей всегда = 200
GEN_N = 100  # Максимум 100 поколений
PARENT_SIZE = POP_SIZE  # Число особей для скрещивания

# ----------------------------------------------------------------
# ----------------------------------------------------------------
def main():
    global g_w
    global g_v  # вместимость*10 (чтобы работать с целыми объемами)

    table_data = ReadFile("12.txt")
    g_w, g_v = GetMax(table_data)
    print("Грузоподъемность: ", g_w, "Вместимость: ", g_v / 10)

    data = GetData(table_data)
    n_obj = len(data)

    # ----------------------------------------------------------------
    # 1. GeneticAlgorithm from pyeasyga
    # ----------------------------------------------------------------
    ga = pyeasyga.GeneticAlgorithm(data)  # initialise the GA with data
    ga.population_size = 200  # increase population size to 200 (default value is 50)

    ga.fitness_function = Fitness  # set the GA's fitness function
    ga.run()  # run the GA

    solution = ga.best_individual()[1]
    param = GetAll(solution, data)
    set_objects = [i + 1 for i in range(n_obj) if solution[i] == 1]
    weight = param[0]
    volume = param[1] / 10
    price = param[2]

    print("� ешение, полученное применением генетического алгоритма из pyeasyga")
    print("Ценность: ", price)
    print("Набор предметов: ", set_objects)
    print("Вес набора: ", weight, "Объем набора: ", volume)

    # ----------------------------------------------------------------
    # 2. Собственная реализация генетического алгоритма
    # ----------------------------------------------------------------

    random.seed()  # initialization of random

    my_price = CheckData(data)

    if my_price == 0:
        print("данные некоректны, задача не имеет решения")
        my_solution = [0] * n_obj
        my_set_objects = []
        my_weight = 0
        my_volume = 0

    else:
        if my_price > 0:
            print("полностью заполненный рюкзак является решением")
            my_solution = [1] * n_obj
            my_param = GetAll(my_solution, data)
            my_set_objects = [i + 1 for i in range(n_obj)]
            my_weight = my_param[0]
            my_volume = my_param[1] / 10
        else:
            print("ищем решение")
            pop = InitPop(data)
            pop_fitness = [Fitness(individual, data) for individual in pop]
            max_pf = max(pop_fitness)
            # -------------------------------
            for n_gen in range(GEN_N):

                parents = Roulette(pop[:], data)
                childs = GetChilds(parents[:])
                muted = Mute(childs)
                pop = GetNewGen(pop[:], muted, data)

                pop_fitness = [Fitness(individual, data) for individual in pop]
                new_max_pf = max(pop_fitness)

                if Result(max_pf, new_max_pf):
                    break

                max_pf = new_max_pf
            # -------------------------------

            pop = sorted(pop, key=lambda ind: -Fitness(ind, data))
            my_solution = pop[0]
            my_param = GetAll(my_solution, data)
            my_set_objects = [i + 1 for i in range(n_obj) if my_solution[i] == 1]
            my_weight = my_param[0]
            my_volume = my_param[1] / 10
            my_price = my_param[2]

        print("Решение, полученное применением cобственной реализации генетического алгоритма")
        print("Ценность: ", my_price)
        print("Набор предметов: ", my_set_objects)
        print("Вес набора: ", my_weight, "Объем набора: ", my_volume)

    return


if __name__ == "__main__":
    main()