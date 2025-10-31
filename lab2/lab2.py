import matplotlib.pyplot as plt
import numpy as np


def triangular_mu(x, a, b, c):
    """Вычисляет степень принадлежности x к треугольному нечёткому множеству."""
    if x < a or x > c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def plot_fuzzy_sets(fuzzy_sets, x):
    """
    Строит график всех треугольных функций принадлежности и отмечает точку x.
    """
    plt.figure(figsize=(10, 6))

    # Определяем общий диапазон по оси X
    all_params = [param for _, (a, b, c) in fuzzy_sets for param in (a, b, c)]
    x_min = min(all_params) - 2
    x_max = max(all_params) + 2
    x_vals = np.linspace(x_min, x_max, 1000)

    # Рисуем каждую функцию принадлежности
    for name, (a, b, c) in fuzzy_sets:
        y_vals = [triangular_mu(val, a, b, c) for val in x_vals]
        plt.plot(x_vals, y_vals, label=name, linewidth=2)

    # Отмечаем введённое значение x
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1.5, label=f'x = {x}')

    # Настройка графика
    plt.title("Нечёткие множества с треугольной функцией принадлежности", fontsize=14)
    plt.xlabel("Значение объекта (x)", fontsize=12)
    plt.ylabel("Степень принадлежности μ(x)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print("Задание нечётких множеств с треугольной функцией принадлежности")
    print("=" * 65)

    # Ввод количества множеств
    try:
        n = int(input("Сколько нечётких множеств вы хотите задать? "))
        if n <= 0:
            print("Ошибка: количество должно быть положительным целым числом.")
            return
    except ValueError:
        print("Ошибка: введите целое число.")
        return

    fuzzy_sets = []

    # Ввод параметров для каждого множества
    for i in range(1, n + 1):
        print(f"\n--- Множество {i} ---")
        name = input("Название множества: ").strip()
        try:
            a = float(input("Параметр a (левая граница): "))
            b = float(input("Параметр b (пик): "))
            c = float(input("Параметр c (правая граница): "))
        except ValueError:
            print("Ошибка: параметры должны быть числами.")
            return

        if not (a <= b <= c):
            print("Ошибка: должно выполняться условие a ≤ b ≤ c.")
            return

        fuzzy_sets.append((name, (a, b, c)))

    # Ввод объекта
    try:
        x = float(input("\nВведите значение объекта (x): "))
    except ValueError:
        print("Ошибка: значение объекта должно быть числом.")
        return

    # Вывод результатов в консоль
    print("\n" + "=" * 65)
    print(f"Объект: x = {x}")
    print("Степени принадлежности к заданным нечётким множествам:")
    print("-" * 65)

    for name, (a, b, c) in fuzzy_sets:
        mu = triangular_mu(x, a, b, c)
        print(f"{name:20} → μ = {mu:.4f}   [a={a}, b={b}, c={c}]")

    print("=" * 65)

    # Построение графика
    try:
        plot_fuzzy_sets(fuzzy_sets, x)
    except Exception as e:
        print(f"Не удалось построить график: {e}")


if __name__ == "__main__":
    main()