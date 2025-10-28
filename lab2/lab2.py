def triangular_mu(x, a, b, c):
    """Вычисляет степень принадлежности x к треугольному нечёткому множеству."""
    if x < a or x > c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


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

    # Вывод результатов
    print("\n" + "=" * 65)
    print(f"Объект: x = {x}")
    print("Степени принадлежности к заданным нечётким множествам:")
    print("-" * 65)

    for name, (a, b, c) in fuzzy_sets:
        mu = triangular_mu(x, a, b, c)
        print(f"{name:20} → μ = {mu:.4f}   [a={a}, b={b}, c={c}]")

    print("=" * 65)


if __name__ == "__main__":
    main()