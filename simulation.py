import numpy as np
from generateGif import create_animated_gif
from geom import find_point_C
import math


class SimulationParams:
    DA_x: float
    DA_y: float
    AB_length: float
    DC_length: float
    BC_length: float
    CE_length: float
    BCE_inner_angle: float
    alpha_angle: float


def sim(params: SimulationParams):
    """
    Делает симуляцию
    Параметры:
    offset (tuple или list): Вектор смещения центра вращения (x0, y0).
    length (float): Длина отрезка.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (360, 6).
    Y (numpy.ndarray): Матрица координат Y размером (360, 6).
    Порядок точек по столбцам: A, B, C, D, E, F
    """
    A_x = params.DA_x
    A_y = params.DA_y
    AB_length = params.AB_length
    углы = np.deg2rad(np.arange(0, 360))  # Углы в радианах от 0 до 359 градусов

    x_A = np.full(360, A_x)
    y_A = np.full(360, A_y)

    x_B = A_x + AB_length * np.cos(углы)
    y_B = A_y + AB_length * np.sin(углы)

    x_C = np.zeros((360))
    y_C = np.zeros((360))

    x_D = np.full(360, 0)
    y_D = np.full(360, 0)

    C_init = find_point_C(params.DC_length, (x_B[0], y_B[0]), params.BC_length)
    x_C[0] = C_init[0]
    y_C[0] = C_init[1]

    for i in range(1, 360):
        prevC = (x_C[i - 1], y_C[i - 1])
        C_this = find_point_C(
            params.DC_length, (x_B[i], y_B[i]), params.BC_length, prevIteration=prevC
        )
        x_C[i] = C_this[0]
        y_C[i] = C_this[1]

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

    x_G = np.full((360), xeMax + (xeMax - xeMin) * 0.3)
    y_G = (
        x_E * math.sin(-params.alpha_angle) + y_E + (math.sin(params.alpha_angle) * x_G)
    )

    # Создаем матрицы с повторяющимися координатами центра и вычисленными концами
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F, x_G))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F, y_G))

    return X, Y
