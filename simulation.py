import numpy as np
import math

from numba import njit

deg = math.pi / 180.0


@njit
def get_intersections(
    x0: np.array, y0: np.array, r0: float, x1: np.array, y1: np.array, r1: float
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

    return (x3, y3), (x4, y4)


@njit()
def simulateRevolutePress(params: np.array) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает симуляцию пресса с вращательной кинематической парой.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (360, 6).
    Y (numpy.ndarray): Матрица координат Y размером (360, 6).
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
    angles = np.deg2rad(np.arange(0, 360))  # Углы в радианах от 0 до 359 градусов

    x_A = np.full(360, A_x)
    y_A = np.full(360, A_y)

    x_B = A_x + AB_length * np.cos(angles)
    y_B = A_y + AB_length * np.sin(angles)

    x_D = np.full(360, 0)
    y_D = np.full(360, 0)

    x_C = np.zeros((360))
    y_C = np.zeros((360))

    (x3, y3), (x4, y4) = get_intersections(x_B, y_B, l_BC, x_D, y_D, l_CD)
    if x3[0] > x4[0]:
        x_C[0] = x3[0]
    else:
        x_C[0] = x4[0]
    for i in range(1, 360):
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

    x_E = np.zeros((360))
    x_E = np.zeros((360))

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

    x_F = np.full(360, x_E[0] - delta_2_x)

    y_F = y_E - np.sqrt(-((x_E - x_F) ** 2) + l_EF**2)

    # Создаем матрицы
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F))

    return X, Y


@njit
def simulatePrismaticPress(params: np.array):
    """
    Делает симуляцию пресса с поступательной кинематической парой.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (360, 6).
    Y (numpy.ndarray): Матрица координат Y размером (360, 6).
    Порядок точек (центров вращательных кинематических пар) по столбцам: A, B, C, D, E, F
    """
    A_x = params.DA_x
    A_y = params.DA_y
    AB_length = params.AB_length
    angles = np.deg2rad(np.arange(0, 360))  # Углы в радианах от 0 до 359 градусов

    x_A = np.full(360, A_x)
    y_A = np.full(360, A_y)

    x_B = A_x + AB_length * np.cos(angles)
    y_B = A_y + AB_length * np.sin(angles)

    x_D = np.full(360, 0)
    y_D = np.full(360, 0)

    x_C = np.zeros((360))
    y_C = np.zeros((360))

    (x3, y3), (x4, y4) = get_intersections(x_B, y_B, l_BC, x_D, y_D, l_CD)
    if x3[0] > x4[0]:
        x_C[0] = x3[0]
    else:
        x_C[0] = x4[0]
    for i in range(1, 360):
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

    x_E = np.zeros((360))
    x_E = np.zeros((360))

    x_BC_e = x_C - x_B
    y_BC_e = y_C - y_B
    x_BC_e /= params.BC_length
    y_BC_e /= params.BC_length

    x_CE_e = x_BC_e * math.cos(params.BCE_inner_angle) + y_BC_e * math.sin(
        params.BCE_inner_angle
    )

    y_CE_e = -x_BC_e * math.sin(params.BCE_inner_angle) + y_BC_e * math.cos(
        params.BCE_inner_angle
    )

    x_CE = x_CE_e * params.CE_length
    y_CE = y_CE_e * params.CE_length

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

    x_F = np.full((360), xeMin - (xeMax - xeMin) * 0.3)
    y_F = (
        x_E * math.sin(-params.alpha_angle) + y_E + (math.sin(params.alpha_angle) * x_F)
    )
    # Создаем матрицы
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F))

    return X, Y
