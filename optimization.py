from simulation import simulateRevolutePress
from scipy.optimize import minimize
from targetFunction import targetFunction
import random
import math
import numpy as np
from copy import deepcopy

nth = 0


def objective_function(x):
    # Создаём параметры симуляции

    try:
        # Проводим симуляцию
        X, Y = simulateRevolutePress(x, 360)

        # Вычисляем целевую функцию
        obj = targetFunction(X, Y)
    except KeyboardInterrupt:
        raise
    except ValueError:
        return np.inf
    global nth
    if nth == 1000:
        nth = 0
        print(obj)
    nth += 1
    return obj[0]


def optimize_function(mechanism) -> tuple[str, np.ndarray]:
    bounds = mechanism.getBounds()
    for i in range(100000):
        # Начальная точка для оптимизации (выбирается рандомно)
        initial_guess = np.zeros(len(bounds))
        for j in range(len(bounds)):
            initial_guess[j] = random.uniform(*bounds[j])
        if not mechanism.isRevolutable(initial_guess):
            continue

        # Оптимизация
        result = minimize(
            objective_function,
            initial_guess,
            bounds=bounds,
            options={"maxiter": 100000, "ftol": 1e-9, "gtol": 1e-5, "disp": True},
        )

        # Проверка на успех
        if result.success:
            resultParams = result.x
            X, Y = mechanism.simulate(resultParams, 360)
            obj = targetFunction(X, Y)
            desc = repr(resultParams)
            desc += obj[1]
            if result.fun > 0.1024:
                continue
            print(result.x)
            return (desc, resultParams)
        else:
            print("Not success")
    raise Exception("Optimization failed 10000 times in a row")
