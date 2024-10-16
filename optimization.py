from simulation import simulateRevolutePress
from scipy.optimize import minimize
from targetFunction import targetFunction
import random
import math
import numpy as np
from copy import deepcopy


def objective_function(x):
    # Создаём параметры симуляции

    try:
        # Проводим симуляцию
        X, Y = simulateRevolutePress(x)

        # Вычисляем целевую функцию
        obj = targetFunction(Y[:, 5])
    except KeyboardInterrupt:
        raise
    except:  # noqa: E722
        return np.inf
    return obj[0]


def optimize_function(mechanism) -> tuple[str, np.array]:
    bounds = mechanism.getBounds()
    for i in range(100000):
        # Начальная точка для оптимизации (выбирается рандомно)
        initial_guess = np.zeros(len(bounds))
        for j in range(len(bounds)):
            initial_guess[j] = random.uniform(*bounds[j])
        if not mechanism.isRevolutable(initial_guess):
            continue

        # Оптимизация
        result = minimize(objective_function, initial_guess, bounds=bounds)

        # Проверка на успех
        if result.success:
            resultParams = result.x
            X, Y = mechanism.simulate(resultParams)
            obj = targetFunction(Y[:, 5])
            desc = repr(resultParams)
            desc += obj[1]
            if result.fun > 0.1024:
                continue
            print(result.x)
            return (desc, resultParams)
    raise Exception("Optimization failed 10000 times in a row")
