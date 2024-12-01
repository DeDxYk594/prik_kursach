import numpy as np
import math

from numba import njit

deg = math.pi / 180.0


@njit
def get_intersections(
    x0: np.ndarray,
    y0: np.ndarray,
    r0: float,
    x1: np.ndarray,
    y1: np.ndarray,
    r1: float,
):
    # Вычисляем расстояние между центрами окружностей
    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # Проверяем условия для не пересекающихся окружностей
    if np.any(d > r0 + r1):
        raise ValueError("Окружности не пересекаются: слишком далеко друг от друга.")
    if np.any(d < np.abs(r0 - r1)):
        raise ValueError(
            "Окружности не пересекаются: одна окружность находится внутри другой."
        )
    if np.any((d == 0) & (r0 == r1)):
        raise ValueError("Окружности совпадают.")

    # Вычисляем параметры a и h для нахождения точек пересечения
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(r0**2 - a**2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d

    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d

    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    if (
        np.isnan(x3).any()
        or np.isnan(y3).any()
        or np.isnan(x4).any()
        or np.isnan(y4).any()
    ):
        raise ValueError("NaN somewhere in result")

    return (x3, y3), (x4, y4)


@njit
def simulateRevolutePress(
    params: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает симуляцию пресса с вращательной кинематической парой.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (steps, 6).
    Y (numpy.ndarray): Матрица координат Y размером (steps, 6).
    Порядок точек (центров вращательных кинематических пар) по столбцам: A, B, C, D, E, F
    """

    l_AB = params[0]
    l_BC = params[1]
    l_CD = params[2]
    l_CE = params[3]
    l_EF = params[4]
    delta_1_x = params[5]
    delta_1_y = params[6]
    delta_2_x = params[7]
    alpha_1_angle = params[8]

    A_x = delta_1_x
    A_y = delta_1_y
    AB_length = l_AB
    angles = np.deg2rad(
        np.arange(0, 360, 360 / steps)
    )  # Углы в радианах от 0 до 360 градусов

    x_A = np.full(steps, A_x)
    y_A = np.full(steps, A_y)

    x_B = A_x + AB_length * np.cos(angles)
    y_B = A_y + AB_length * np.sin(angles)

    x_D = np.zeros(steps)
    y_D = np.zeros(steps)

    x_C = np.zeros(steps)
    y_C = np.zeros(steps)

    (x3, y3), (x4, y4) = get_intersections(x_B, y_B, l_BC, x_D, y_D, l_CD)
    if x3[0] > x4[0]:
        x_C[0] = x3[0]
        y_C[0] = y3[0]
    else:
        x_C[0] = x4[0]
        y_C[0] = y4[0]

    for i in range(1, steps):
        prev_x = x_C[i - 1]
        prev_y = y_C[i - 1]
        d3 = math.sqrt((prev_x - x3[i]) ** 2 + (prev_y - y3[i]) ** 2)
        d4 = math.sqrt((prev_x - x4[i]) ** 2 + (prev_y - y4[i]) ** 2)
        if d3 > d4:
            x_C[i] = x4[i]
            y_C[i] = y4[i]
        else:
            x_C[i] = x4[i]
            y_C[i] = y4[i]

    x_E = np.zeros(steps)
    x_E = np.zeros(steps)

    x_BC_e = x_C - x_B
    y_BC_e = y_C - y_B
    x_BC_e /= l_BC
    y_BC_e /= l_BC

    x_CE_e = x_BC_e * math.cos(alpha_1_angle * deg) + y_BC_e * math.sin(
        alpha_1_angle * deg
    )

    y_CE_e = -x_BC_e * math.sin(alpha_1_angle * deg) + y_BC_e * math.cos(
        alpha_1_angle * deg
    )

    x_CE = x_CE_e * l_CE
    y_CE = y_CE_e * l_CE

    x_E = x_C + x_CE
    y_E = y_C + y_CE

    x_F = np.full(steps, x_E[0] - delta_2_x)

    y_F = y_E - np.sqrt(-((x_E - x_F) ** 2) + l_EF**2)

    # Создаем матрицы
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F))

    return X, Y


@njit
def simulatePrismaticPress(params: np.ndarray, steps: int):
    """
    Делает симуляцию пресса с поступательной кинематической парой.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (steps, 6).
    Y (numpy.ndarray): Матрица координат Y размером (steps, 6).
    Порядок точек (центров вращательных кинематических пар) по столбцам: A, B, C, D, E, F
    """
    l_AB = params[0]  # мм
    l_BC = params[1]  # мм
    l_CD = params[2]  # мм
    l_CE = params[3]  # мм
    delta_1_x = params[4]  # мм
    delta_1_y = params[5]  # мм
    alpha_1_angle = params[6]  # градусов
    alpha_2_angle = params[7]  # градусов

    angles = np.deg2rad(
        np.arange(0, 360, 360 / steps)
    )  # Углы в радианах от 0 до steps градусов

    x_A = np.full(steps, -delta_1_x)
    y_A = np.full(steps, delta_1_y)

    x_B = x_A + np.cos(angles) * l_AB
    y_B = y_A + np.sin(angles) * l_AB

    x_D = np.zeros(steps)
    y_D = np.zeros(steps)

    x_C = np.zeros(steps)
    y_C = np.zeros(steps)

    (x3, y3), (x4, y4) = get_intersections(x_B, y_B, l_BC, x_D, y_D, l_CD)
    if x3[0] > x4[0]:
        x_C[0] = x3[0]
    else:
        x_C[0] = x4[0]
    for i in range(1, steps):
        prev_x = x_C[i - 1]
        prev_y = y_C[i - 1]
        d3 = math.sqrt((prev_x - x3[i]) ** 2 + (prev_y - y3[i]) ** 2)
        d4 = math.sqrt((prev_x - x4[i]) ** 2 + (prev_y - y4[i]) ** 2)
        if d3 > d4:
            x_C[i] = x4[i]
            y_C[i] = y4[i]
        else:
            x_C[i] = x4[i]
            y_C[i] = y4[i]

    x_E = np.zeros(steps)
    x_E = np.zeros(steps)

    x_BC_e = x_C - x_B
    y_BC_e = y_C - y_B
    x_BC_e /= l_BC
    y_BC_e /= l_BC

    x_CE_e = x_BC_e * math.cos(alpha_1_angle) + y_BC_e * math.sin(alpha_1_angle)

    y_CE_e = -x_BC_e * math.sin(alpha_1_angle) + y_BC_e * math.cos(alpha_1_angle)

    x_CE = x_CE_e * l_CE
    y_CE = y_CE_e * l_CE

    x_E = x_C + x_CE
    y_E = y_C + y_CE

    xeMax = max(
        x_A.max(),
        x_B.max(),
        x_C.max(),
        x_D.max(),
        x_E.max(),
    )
    xeMin = min(
        y_A.min(),
        y_B.min(),
        y_C.min(),
        y_D.min(),
        y_E.min(),
    )

    x_F = np.full((steps), xeMin - (xeMax - xeMin) * 0.3)
    y_F = x_E * math.sin(-alpha_2_angle) + y_E + (math.sin(alpha_2_angle) * x_F)
    # Создаем матрицы
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F))

    return X, Y
