from simulation import SimulationParams, simulateRevolutePress
from scipy.optimize import minimize
from targetFunction import targetFunction
import random
import math
import numpy as np
from copy import deepcopy


def objective_function(x):
    # Создаём параметры симуляции
    params = SimulationParams()
    params.l_AB = x[0]
    params.l_BC = x[1]
    params.l_CD = x[2]
    params.l_CE = x[3]
    params.l_EF = x[4]
    params.delta_1_x = x[5]
    params.delta_1_y = x[6]
    params.delta_2_x = x[7]
    params.alpha_1_angle = x[8]

    try:
        # Проводим симуляцию
        X, Y = simulateRevolutePress(params)

        # Вычисляем целевую функцию
        obj = targetFunction(Y[:, 5])
    except KeyboardInterrupt:
        raise
    except:  # noqa: E722
        return np.inf
    return sum(obj[0])


def optimize_function() -> tuple[str, np.ndarray]:
    for i in range(100000):
        newBounds = deepcopy(bounds)
        # Начальная точка для оптимизации (выбирается рандомно)
        initial_guess = np.zeros(len(bounds))
        for j in range(len(newBounds)):
            initial_guess[j] = random.uniform(*newBounds[j])
        # Оптимизация
        result = minimize(objective_function, initial_guess, bounds=bounds)

        # Проверка на успех
        if result.success:
            res = result.x
            params = SimulationParams()
            X, Y = simulateRevolutePress(params)
            obj = targetFunction(Y[:, 5])
            desc = repr(res)
            desc += str(list(zip(*obj)))
            if result.fun > 0.1024:
                continue
            print(result.x)
            return (desc, res)
    raise Exception("Optimization failed 10000 times in a row")


bounds = [
    (30, 30),
    (30, 500),
    (30, 500),
    (30, 500),
    (30, 500),
    (-60, 60),
    (30, 500),
    (-300, 300),
    (-90, 90),
]
