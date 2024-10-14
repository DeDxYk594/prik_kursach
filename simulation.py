import numpy as np
import math
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from numba import njit

deg = math.pi / 180.0


class SimulationParams:
    """
    Схема, на которой указаны параметры:
    ./img/параметрыОптимизации.svg
    """

    l_AB: float
    l_BC: float
    l_CD: float
    l_CE: float
    l_EF: float
    delta_1_x: float
    delta_1_y: float
    delta_2_x: float
    alpha_1_angle: float  # градусов

    def toarray(self):
        return np.array(
            [
                self.l_AB,
                self.l_BC,
                self.l_CD,
                self.l_CE,
                self.l_EF,
                self.delta_1_x,
                self.delta_1_y,
                self.delta_2_x,
                self.alpha_1_angle,
            ]
        )


def find_point_C(radius_DC, B, distance_BC, prevIteration=None) -> list[float, float]:
    x_b, y_b = B
    center_okr = Point(0, 0)
    circle_okr = center_okr.buffer(radius_DC).exterior

    point_C = Point(x_b, y_b)
    circle_C = point_C.buffer(distance_BC).exterior

    # Найти пересечение окружностей
    intersection = circle_okr.intersection(circle_C)

    if intersection.is_empty:
        raise ValueError("Нет пересечений между окружностями.")

    if intersection.geom_type == "MultiPoint":
        points = list(intersection.geoms)
    else:
        raise ValueError(f"Неожиданный тип пересечения: {intersection.geom_type}")

    # Если prevIteration не задано, выбрать точку с большим x
    if prevIteration is None:
        D = max(points, key=lambda p: (p.x, p.y))
    else:
        prev_point = Point(prevIteration)
        # Найти ближайшую точку к prevIteration
        nearest = nearest_points(prev_point, intersection)[1]
        D = nearest

    return (D.x, D.y)


def find_F_y(E_x: float, E_y: float, l_EF: float, F_x: float) -> float:
    center_okr = Point(E_x, E_y)
    circle_okr = center_okr.buffer(l_EF).exterior

    line = LineString([Point(F_x, -100000), Point(F_x, 100000)])

    # Найти пересечение окружностей
    intersection = circle_okr.intersection(line)

    if intersection.is_empty:
        raise ValueError("Нет пересечений между окружностью и прямой")
    if intersection.geom_type == "MultiPoint":
        points = list(intersection.geoms)
    else:
        raise ValueError(f"Неожиданный тип пересечения: {intersection.geom_type}")

    D = max(points, key=lambda p: (-p.y))

    return D.y


def sim(prms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает симуляцию
    Параметры:
    offset (tuple или list): Вектор смещения центра вращения (x0, y0).
    length (float): Длина отрезка.

    Возвращает:
    X (numpy.ndarray): Матрица координат X размером (360, 6).
    Y (numpy.ndarray): Матрица координат Y размером (360, 6).
    Порядок точек (центров вращательных кинематических пар) по столбцам: A, B, C, D, E, F
    """

    l_AB = prms[0]
    l_BC = prms[1]
    l_CD = prms[2]
    l_CE = prms[3]
    l_EF = prms[4]
    delta_1_x = prms[5]
    delta_1_y = prms[6]
    delta_2_x = prms[7]
    alpha_1_angle = prms[8]

    A_x = delta_1_x
    A_y = delta_1_y
    AB_length = l_AB
    angles = np.deg2rad(np.arange(0, 360))  # Углы в радианах от 0 до 359 градусов

    x_A = np.full(360, A_x)
    y_A = np.full(360, A_y)

    x_B = A_x + AB_length * np.cos(angles)
    y_B = A_y + AB_length * np.sin(angles)

    x_C = np.zeros((360))
    y_C = np.zeros((360))

    x_D = np.full(360, 0)
    y_D = np.full(360, 0)

    C_init = find_point_C(l_CD, (x_B[0], y_B[0]), l_BC)
    x_C[0] = C_init[0]
    y_C[0] = C_init[1]

    for i in range(1, 360):
        prevC = (x_C[i - 1], y_C[i - 1])
        C_this = find_point_C(l_CD, (x_B[i], y_B[i]), l_BC, prevIteration=prevC)
        x_C[i] = C_this[0]
        y_C[i] = C_this[1]

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

    y_F = np.zeros(360)
    for i in range(360):
        y_F[i] = find_F_y(x_E[i], y_E[i], l_EF, x_F[0])

    # Создаем матрицы с повторяющимися координатами центра и вычисленными концами
    X = np.column_stack((x_A, x_B, x_C, x_D, x_E, x_F))
    Y = np.column_stack((y_A, y_B, y_C, y_D, y_E, y_F))

    return X, Y
