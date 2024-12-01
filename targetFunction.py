import numpy as np
import math


def targetFunction(X: np.ndarray, Y: np.ndarray) -> tuple[float, str]:
    """F_y - массив формы (360,) - каждому градусу угла поворота
    кривошипа соответствует координата шабота пресса y.
    @returns два значения:
    первое - сама целевая фукнция
    второе - описание результата str
    """
    F_y = Y[:, 5]
    travelm = travelMetric(F_y)
    critical_sine = (
        np.max(np.abs(X[:, 5] - X[:, 4]))
        / math.sqrt((X[0][5] - X[0][4]) ** 2 + (Y[0][5] - Y[0][4]) ** 2)
    ) * 0.01
    upperTime = upperTimeMetric(F_y)
    THR = 0.5
    if upperTime > THR:
        upperTime = THR + (upperTime - THR) ** 3
    upperTime *= 0.1
    tgt = critical_sine + travelm + upperTime
    ret = (
        tgt,
        f"travel={np.max(F_y)-np.min(F_y)}mm tgt={tgt} tm={travelm} cs={critical_sine} ut={upperTime}",
    )
    return ret


def upperTimeMetric(F_y: np.ndarray) -> float:
    """
    Calculates the proportion of time during which the coordinate y
    is less than the midpoint of its minimum and maximum values.

    Parameters:
    ----------
    F_y : np.ndarray
        Array of y-coordinates corresponding to each degree of crank angle (shape: (360,)).

    Returns:
    -------
    float
        The fraction of time y is below (y_min + y_max) / 2.
    """
    y_min = np.min(F_y)
    y_max = np.max(F_y)
    midpoint = (y_min + 3 * y_max) / 4
    count_above = np.sum(F_y > midpoint)
    total = F_y.size

    return 1 - count_above / total


def travelMetric(F_y: np.ndarray) -> float:
    """
    Computes a metric related to the press's travel (y_max - y_min).
    The metric is minimized when the travel is 30 mm. It increases sharply
    if the travel is less than 30 mm and increases gently if greater.

    Parameters:
    ----------
    F_y : np.ndarray
        Array of y-coordinates corresponding to each degree of crank angle (shape: (360,)).

    Returns:
    -------
    float
        The computed travel metric.
    """
    y_min = np.min(F_y)
    y_max = np.max(F_y)
    travel = y_max - y_min
    target_travel = 180.0  # mm
    if travel < target_travel:
        return (target_travel - travel) ** 4  # Sharp increase
    else:
        return (travel - target_travel) ** 2  # Gentle increase
