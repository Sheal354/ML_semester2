import random
import matplotlib.pyplot as plt
from copy import deepcopy

# Определение данных о продуктах
products = [
    {"name": "Стейк из лосося, 300г", "protein": 44.7, "fat": 27.6, "carbs": 0, "calories": 543, "cost": 660},
    {"name": "Гречка, 80г", "protein": 10.4, "fat": 1.6, "carbs": 57.6, "calories": 288, "cost": 20},
    {"name": "Брокколи, 200г", "protein": 6, "fat": 0, "carbs": 10, "calories": 60, "cost": 80},
    {"name": "Грецкие орехи, 100г", "protein": 14.8, "fat": 64, "carbs": 13.7, "calories": 698, "cost": 200},
    {"name": "Кефир 4%, 250мл", "protein": 7.5, "fat": 10, "carbs": 10, "calories": 160, "cost": 60},
    {"name": "Чечевица, 50г", "protein": 13, "fat": 1, "carbs": 28.5, "calories": 175, "cost": 10},
    {"name": "Авокадо, 150г", "protein": 3, "fat": 22.5, "carbs": 13.5, "calories": 240, "cost": 150},
    {"name": "Филе куриной грудки, 200г", "protein": 26, "fat": 38, "carbs": 50, "calories": 506, "cost": 160},
    {"name": "Филе индейки, 200г", "protein": 48, "fat": 3, "carbs": 4, "calories": 220, "cost": 150},
    {"name": "Стейк из говядины, 300г", "protein": 48, "fat": 54, "carbs": 15, "calories": 660, "cost": 600},
    {"name": "Картофель, 200г", "protein": 4, "fat": 0.8, "carbs": 38, "calories": 231, "cost": 45},
    {"name": "Яйца куриные, 210г", "protein": 26.7, "fat": 24.15, "carbs": 1.5, "calories": 330, "cost": 45},
    {"name": "Творог, 220г", "protein": 35.2, "fat": 11, "carbs": 6.6, "calories": 222.2, "cost": 160},
    {"name": "Сок яблочный, 200мл", "protein": 0, "fat": 0, "carbs": 23, "calories": 92, "cost": 30},
    {"name": "Бананы, 300г", "protein": 4.5, "fat": 0.6, "carbs": 65.4, "calories": 285, "cost": 60},
    {"name": "Яблоко зеленое, 220г", "protein": 0.88, "fat": 0.44, "carbs": 42, "calories": 158.4, "cost": 35},
    {"name": "Рис, 66.6г", "protein": 4.7, "fat": 0.7, "carbs": 49.3, "calories": 222, "cost": 15},
    {"name": "Сосиски вареные с сыром, 120г", "protein": 15.6, "fat": 26.4, "carbs": 3.6, "calories": 312, "cost": 110},
    {"name": "Яйца перепелиные, 30г", "protein": 3.84, "fat": 3.93, "carbs": 1.5, "calories": 51, "cost": 15},
    {"name": "Миндаль, 100г", "protein": 21.2, "fat": 49.9, "carbs": 21.7, "calories": 575, "cost": 115},
    {"name": "Куриная печень, 150г", "protein": 27, "fat": 15, "carbs": 0, "calories": 240, "cost": 55},
    {"name": "Стейк из семги, 250г", "protein": 38.75, "fat": 19.25, "carbs": 0.25, "calories": 457.5, "cost": 625},
    {"name": "Морковь вареная, 150г", "protein": 1.2, "fat": 0.3, "carbs": 7.8, "calories": 52.5, "cost": 35},
    {"name": "Капуста цветная, 200г", "protein": 5, "fat": 0.6, "carbs": 10.8, "calories": 86.6, "cost": 60},
    {"name": "Шоколад молочный, 80г", "protein": 4.32, "fat": 22.4, "carbs": 49.6, "calories": 417.6, "cost": 50},
    {"name": "Молоко 3,5 - 4,5%, 250мл", "protein": 7.5, "fat": 11.25, "carbs": 11.75, "calories": 177.5, "cost": 40},
    {"name": "Сметана 20%, 100г", "protein": 2.5, "fat": 20, "carbs": 3.4, "calories": 204, "cost": 45},
    {"name": "Плюшка сдобная, 200г", "protein": 15.4, "fat": 16.8, "carbs": 107.8, "calories": 650, "cost": 55},
    {"name": "Паштет из мяса индейки, 40г", "protein": 4, "fat": 7.6, "carbs": 10, "calories": 110.8, "cost": 95},
    {"name": "Сыр плавленый, 35г", "protein": 3.85, "fat": 7.35, "carbs": 2.1, "calories": 91, "cost": 25},
    {"name": "Мороженое пломбир ваф.стаканчик, 70г", "protein": 2.8, "fat": 7.7, "carbs": 17.15, "calories": 147,
     "cost": 50},
]

# Эталонные значения
TARGET = {
    "protein": 76,
    "fat": 70,
    "carbs": 250,
    "calories": 2200
}

# Допуски отклонения
TOLERANCES = {
    "protein": 8,
    "fat": 7.5,
    "carbs": 25,
    "calories": 150
}


def fitness(chromosome):
    """Функция приспособленности с приоритетом на допуски и ограничением на количество продуктов"""
    totals = {"protein": 0, "fat": 0, "carbs": 0, "calories": 0, "cost": 0}

    # Подсчитываем количество выбранных продуктов
    num_products = sum(chromosome)

    for i, included in enumerate(chromosome):
        if included:
            p = products[i]
            totals["protein"] += p["protein"]
            totals["fat"] += p["fat"]
            totals["carbs"] += p["carbs"]
            totals["calories"] += p["calories"]
            totals["cost"] += p["cost"]

    # Штраф за выход за пределы допусков
    penalty = 0
    if abs(totals["protein"] - TARGET["protein"]) > TOLERANCES["protein"]:
        penalty += 100000
    if abs(totals["fat"] - TARGET["fat"]) > TOLERANCES["fat"]:
        penalty += 100000
    if abs(totals["carbs"] - TARGET["carbs"]) > TOLERANCES["carbs"]:
        penalty += 100000
    if abs(totals["calories"] - TARGET["calories"]) > TOLERANCES["calories"]:
        penalty += 100000

    # Штраф за превышение количества продуктов
    if num_products > 6:
        # Формула штрафа: 1000 + 500 * (num_products - 6)
        # 7 продуктов: 1000, 8 продуктов: 1500, 9 продуктов: 2000 и т.д.
        product_penalty = 1000 + 500 * (num_products - 6)
        penalty += product_penalty

    # Основной расчет (только для решений в пределах допусков)
    deviation = (
            (totals["protein"] - TARGET["protein"]) ** 2 +
            (totals["fat"] - TARGET["fat"]) ** 2 +
            (totals["carbs"] - TARGET["carbs"]) ** 2 +
            (totals["calories"] - TARGET["calories"]) ** 2
    )

    # Если решение вне допусков или с избыточным количеством продуктов, возвращаем только штраф
    if penalty > 0:
        return penalty

    # Если решение в допусках, учитываем отклонение и стоимость
    return deviation * 10 + totals["cost"]


def generate_population(size, n_genes=len(products)):
    """Генерирует начальную популяцию"""
    return [random.choices([0, 1], k=n_genes) for _ in range(size)]


def tournament_selection(population, fitnesses, k=3):
    """Турнирный отбор"""
    selected_idx = random.sample(range(len(population)), k)
    selected_fitness = [fitnesses[i] for i in selected_idx]
    winner_idx = selected_idx[selected_fitness.index(min(selected_fitness))]
    return population[winner_idx]


# Операторы скрещивания
def crossover_one_point(parent1, parent2):
    """Одноточечное скрещивание"""
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def crossover_two_point(parent1, parent2):
    """Двухточечное скрещивание"""
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    return (
        parent1[:point1] + parent2[point1:point2] + parent1[point2:],
        parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    )


def crossover_uniform(parent1, parent2, p=0.5):
    """Равномерное скрещивание"""
    child1 = [parent1[i] if random.random() < p else parent2[i] for i in range(len(parent1))]
    child2 = [parent2[i] if random.random() < p else parent1[i] for i in range(len(parent1))]
    return child1, child2


# Операторы мутации
def mutation_force_include(chromosome):
    """Принудительное включение случайного продукта"""
    idx = random.randint(0, len(chromosome) - 1)
    chromosome[idx] = 1
    return chromosome


def mutation_flip(chromosome):
    """Изменение значения на противоположное"""
    idx = random.randint(0, len(chromosome) - 1)
    chromosome[idx] = 1 - chromosome[idx]
    return chromosome


def mutation_force_exclude(chromosome):
    """Принудительное исключение случайного продукта"""
    idx = random.randint(0, len(chromosome) - 1)
    chromosome[idx] = 0
    return chromosome


def run_ga(crossover_func, mutation_func,
           pop_size=100, max_generations=25,
           crossover_prob=0.85, mutation_prob=0.15,
           verbose=False):
    """Запуск генетического алгоритма"""
    population = generate_population(pop_size)
    best_fitness_history = []
    best_solution = None
    best_fitness = float('inf')

    for gen in range(max_generations):
        # Оценка приспособленности
        fitnesses = [fitness(chrom) for chrom in population]
        current_best_idx = fitnesses.index(min(fitnesses))
        current_best_fitness = fitnesses[current_best_idx]

        best_fitness_history.append(current_best_fitness)

        # Сохраняем лучшее решение
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx][:]

        if verbose and gen % 10 == 0:
            print(f"Поколение {gen}: Лучшая приспособленность = {current_best_fitness:.2f}")

        # Формирование нового поколения
        new_population = []

        # Формирование нового поколения
        while len(new_population) < pop_size:
            # Отбор родителей
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Скрещивание
            if random.random() < crossover_prob:
                child1, child2 = crossover_func(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Мутация
            for child in [child1, child2]:
                if random.random() < mutation_prob:
                    child = mutation_func(deepcopy(child))
                new_population.append(child)

        # Замена популяции
        population = new_population[:pop_size]

    return best_fitness_history, best_solution


def plot_results(crossover_results, mutation_results):
    """Визуализация результатов экспериментов"""
    plt.figure(figsize=(14, 6))

    # График для методов скрещивания
    plt.subplot(1, 2, 1)
    for name, (history, _) in crossover_results.items():
        plt.plot(history, label=name)
    plt.title('Сравнение методов скрещивания')
    plt.xlabel('Поколение')
    plt.ylabel('Отклонение')
    plt.legend()
    plt.grid(True)

    # График для методов мутации
    plt.subplot(1, 2, 2)
    for name, (history, _) in mutation_results.items():
        plt.plot(history, label=name)
    plt.title('Сравнение методов мутации')
    plt.xlabel('Поколение')
    plt.ylabel('Отклонение')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ga_results.png', dpi=300)
    plt.show()


def calculate_deviations(chromosome):
    """Рассчитывает отклонения от эталона для хромосомы"""
    totals = {"protein": 0, "fat": 0, "carbs": 0, "calories": 0}

    for i, included in enumerate(chromosome):
        if included:
            p = products[i]
            totals["protein"] += p["protein"]
            totals["fat"] += p["fat"]
            totals["carbs"] += p["carbs"]
            totals["calories"] += p["calories"]

    # Рассчитываем отклонения
    protein_dev = abs(totals["protein"] - TARGET["protein"])
    fat_dev = abs(totals["fat"] - TARGET["fat"])
    carbs_dev = abs(totals["carbs"] - TARGET["carbs"])
    calories_dev = abs(totals["calories"] - TARGET["calories"])

    return {
        "protein_dev": protein_dev,
        "fat_dev": fat_dev,
        "carbs_dev": carbs_dev,
        "calories_dev": calories_dev,
        "totals": totals
    }


def is_within_limits(chromosome):
    """Проверка, удовлетворяет ли хромосома заданным допускам"""
    deviations = calculate_deviations(chromosome)
    return (deviations["protein_dev"] <= TOLERANCES["protein"] and
            deviations["fat_dev"] <= TOLERANCES["fat"] and
            deviations["carbs_dev"] <= TOLERANCES["carbs"] and
            deviations["calories_dev"] <= TOLERANCES["calories"])


def main():
    # Параметры экспериментов
    POP_SIZE = 150
    MAX_GENERATIONS = 25

    print("=" * 60)
    print("ЭКСПЕРИМЕНТЫ С РАЗНЫМИ МЕТОДАМИ СКРЕЩИВАНИЯ")
    print("=" * 60)

    # Эксперименты со скрещиванием (фиксированная мутация - flip)
    crossover_results = {}
    crossover_funcs = {
        'Одноточечное': crossover_one_point,
        'Двухточечное': crossover_two_point,
        'Равномерное': crossover_uniform
    }

    print("Запуск экспериментов со скрещиванием...")
    for name, func in crossover_funcs.items():
        print(f"\nЗапуск эксперимента: {name} скрещивание")
        history, best_solution = run_ga(
            crossover_func=func,
            mutation_func=mutation_flip,
            pop_size=POP_SIZE,
            max_generations=MAX_GENERATIONS
        )

        # Рассчитываем отклонения для лучшего решения
        deviations = calculate_deviations(best_solution)

        # Подсчитываем количество продуктов в решении
        num_products = sum(best_solution)

        # Сохраняем результаты
        crossover_results[name] = (history, best_solution)

        # Выводим результаты
        print(f"\n{name} скрещивание - результаты:")
        print(f"Лучшая приспособленность: {min(history):.2f}")
        print(f"Количество продуктов в рационе: {num_products}")
        print(f"Отклонения от эталона:")
        print(f"  - Белки: {deviations['protein_dev']:.1f} г (допуск ±{TOLERANCES['protein']})")
        print(f"  - Жиры: {deviations['fat_dev']:.1f} г (допуск ±{TOLERANCES['fat']})")
        print(f"  - Углеводы: {deviations['carbs_dev']:.1f} г (допуск ±{TOLERANCES['carbs']})")
        print(f"  - Калории: {deviations['calories_dev']:.1f} ккал (допуск ±{TOLERANCES['calories']})")

        # Определяем, в пределах ли допусков
        within_limits = is_within_limits(best_solution)

        status = "В пределах допусков" if within_limits else "Вне допусков"
        print(f"Статус: {status}")

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТЫ С РАЗНЫМИ МЕТОДАМИ МУТАЦИИ")
    print("=" * 60)

    # Эксперименты с мутацией
    mutation_results = {}
    mutation_funcs = {
        'Принудительное включение': mutation_force_include,
        'Битовая инверсия': mutation_flip,
        'Принудительное исключение': mutation_force_exclude
    }

    print("\nЗапуск экспериментов с мутацией...")
    for name, func in mutation_funcs.items():
        print(f"\nЗапуск эксперимента: {name} мутация")
        history, best_solution = run_ga(
            crossover_func=crossover_one_point,
            mutation_func=func,
            pop_size=POP_SIZE,
            max_generations=MAX_GENERATIONS
        )

        # Рассчитываем отклонения для лучшего решения
        deviations = calculate_deviations(best_solution)

        # Подсчитываем количество продуктов в решении
        num_products = sum(best_solution)

        # Сохраняем результаты
        mutation_results[name] = (history, best_solution)

        # Выводим результаты
        print(f"\n{name} мутация - результаты:")
        print(f"Лучшая приспособленность: {min(history):.2f}")
        print(f"Количество продуктов в рационе: {num_products}")
        print(f"Отклонения от эталона:")
        print(f"  - Белки: {deviations['protein_dev']:.1f} г (допуск ±{TOLERANCES['protein']})")
        print(f"  - Жиры: {deviations['fat_dev']:.1f} г (допуск ±{TOLERANCES['fat']})")
        print(f"  - Углеводы: {deviations['carbs_dev']:.1f} г (допуск ±{TOLERANCES['carbs']})")
        print(f"  - Калории: {deviations['calories_dev']:.1f} ккал (допуск ±{TOLERANCES['calories']})")

        # Определяем, в пределах ли допусков
        within_limits = is_within_limits(best_solution)

        status = "В пределах допусков" if within_limits else "Вне допусков"
        print(f"Статус: {status}")

    # Визуализация результатов
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    plot_results(crossover_results, mutation_results)

    # Анализ лучших решений по всем экспериментам
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЛУЧШИХ РЕШЕНИЙ ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
    print("=" * 60)

    # Найти лучшее решение среди всех методов скрещивания
    best_crossover = min(crossover_results.items(), key=lambda x: min(x[1][0]))
    name, (history, solution) = best_crossover
    deviations = calculate_deviations(solution)
    num_products = sum(solution)

    print(f"\nЛучшее решение среди методов скрещивания: {name}")
    print(f"Количество продуктов: {num_products}")
    print(f"Лучшая приспособленность: {min(history):.2f}")
    print(f"Отклонения:")
    print(f"  - Белки: {deviations['protein_dev']:.1f} г")
    print(f"  - Жиры: {deviations['fat_dev']:.1f} г")
    print(f"  - Углеводы: {deviations['carbs_dev']:.1f} г")
    print(f"  - Калории: {deviations['calories_dev']:.1f} ккал")

    # Найти лучшее решение среди всех методов мутации
    best_mutation = min(mutation_results.items(), key=lambda x: min(x[1][0]))
    name, (history, solution) = best_mutation
    deviations = calculate_deviations(solution)
    num_products = sum(solution)

    print(f"\nЛучшее решение среди методов мутации: {name}")
    print(f"Количество продуктов: {num_products}")
    print(f"Лучшая приспособленность: {min(history):.2f}")
    print(f"Отклонения:")
    print(f"  - Белки: {deviations['protein_dev']:.1f} г")
    print(f"  - Жиры: {deviations['fat_dev']:.1f} г")
    print(f"  - Углеводы: {deviations['carbs_dev']:.1f} г")
    print(f"  - Калории: {deviations['calories_dev']:.1f} ккал")

    # Общее лучшее решение
    all_solutions = list(crossover_results.values()) + list(mutation_results.values())
    all_best = min(all_solutions, key=lambda x: min(x[0]))

    print("\n" + "=" * 60)
    print("ОБЩЕЕ ЛУЧШЕЕ РЕШЕНИЕ СРЕДИ ВСЕХ МЕТОДОВ")
    print("=" * 60)

    # Определяем, откуда взято решение
    source = ""
    if all_best in crossover_results.values():
        source = "метода скрещивания"
        name = [k for k, v in crossover_results.items() if v == all_best][0]
    else:
        source = "метода мутации"
        name = [k for k, v in mutation_results.items() if v == all_best][0]

    history, solution = all_best
    deviations = calculate_deviations(solution)
    num_products = sum(solution)

    print(f"Найдено с помощью: {name} ({source})")
    print(f"Количество продуктов: {num_products}")
    print(f"Лучшая приспособленность: {min(history):.2f}")
    print(f"Отклонения:")
    print(f"  - Белки: {deviations['protein_dev']:.1f} г")
    print(f"  - Жиры: {deviations['fat_dev']:.1f} г")
    print(f"  - Углеводы: {deviations['carbs_dev']:.1f} г")
    print(f"  - Калории: {deviations['calories_dev']:.1f} ккал")

    # Проверяем, в пределах ли допусков
    within_limits = is_within_limits(solution)

    status = "В пределах допусков" if within_limits else "Вне допусков"
    print(f"\nСтатус общего лучшего решения: {status}")

    # Выводим детали лучшего решения
    print("\nДетали лучшего решения:")
    total = {"protein": 0, "fat": 0, "carbs": 0, "calories": 0, "cost": 0}
    selected = []

    for i, include in enumerate(solution):
        if include:
            p = products[i]
            selected.append(p["name"])
            total["protein"] += p["protein"]
            total["fat"] += p["fat"]
            total["carbs"] += p["carbs"]
            total["calories"] += p["calories"]
            total["cost"] += p["cost"]

    print(f"\nВыбранные продукты ({len(selected)}):")
    for item in selected:
        print(f" - {item}")

    print("\nИтоговые характеристики:")
    print(f"Белки: {total['protein']:.1f} г (цель: {TARGET['protein']})")
    print(f"Жиры: {total['fat']:.1f} г (цель: {TARGET['fat']})")
    print(f"Углеводы: {total['carbs']:.1f} г (цель: {TARGET['carbs']})")
    print(f"Калории: {total['calories']:.1f} ккал (цель: {TARGET['calories']})")
    print(f"Стоимость: {total['cost']} руб")


if __name__ == "__main__":
    main()