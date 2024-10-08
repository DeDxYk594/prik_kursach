from shapely.geometry import Point
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import geopandas as gpd
import math


def find_point_C(radius_DC, B, distance_BC, prevIteration=None):
    x_b, y_b = B
    center_okr = Point(0, 0)
    circle_okr = center_okr.buffer(radius_DC).exterior

    point_C = Point(x_b, y_b)
    circle_C = point_C.buffer(distance_BC).exterior

    # Найти пересечение окружностей
    intersection = circle_okr.intersection(circle_C)

    if intersection.is_empty:
        raise ValueError("Нет пересечений между окружностями.")

    # Получить все точки пересечения
    if intersection.geom_type == "Point":
        points = [intersection]
    elif intersection.geom_type == "MultiPoint":
        points = list(intersection.geoms)
    else:
        raise ValueError(f"Неожиданный тип пересечения: {intersection.geom_type}")

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
