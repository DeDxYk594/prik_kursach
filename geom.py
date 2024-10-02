from shapely.geometry import Point
from shapely.ops import nearest_points
import math


def find_point_C(radius_B, C, distance_BC, prevIteration=None):
    x_c, y_c = C
    center_B = Point(0, 0)
    circle_B = center_B.buffer(radius_B)

    point_C = Point(x_c, y_c)
    circle_C = point_C.buffer(distance_BC)

    # Найти пересечение окружностей
    intersection = circle_B.intersection(circle_C)

    if intersection.is_empty:
        raise ValueError("Нет пересечений между окружностями.")

    # Получить все точки пересечения
    if intersection.geom_type == "Point":
        points = [intersection]
    elif intersection.geom_type == "MultiPoint":
        points = list(intersection)
    else:
        raise ValueError("Неожиданный тип пересечения.")

    if not points:
        raise ValueError("Нет пересечений между окружностями.")

    # Если prevIteration не задано, выбрать точку с большим x
    if prevIteration is None:
        D = max(points, key=lambda p: (p.x, p.y))
    else:
        prev_point = Point(prevIteration)
        # Найти ближайшую точку к prevIteration
        nearest = nearest_points(prev_point, intersection)[1]
        D = nearest

    return (D.x, D.y)
