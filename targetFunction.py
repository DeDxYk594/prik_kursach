import numpy as np


def targetFunction(F_y: np.ndarray[float]):
    """F_y - массив формы (360,) - каждому градусу угла поворота
    кривошипа соответствует координата шабота пресса y.
    @returns два списка:
    первый список - список значений целевых подфункций, он будет просуммирован
    второй список - список меток для первого списка
    """
    ret = (
        (upperTimeMetric(F_y),),
        ("upperTimeM",),
    )
    # print(ret)
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
    midpoint = (y_min + y_max) / 2
    count_below = np.sum(F_y < midpoint)
    total = F_y.size
    if count_below / total > 0.45:
        return 999999
    return (count_below / total) ** 2


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
    target_travel = 50.0  # mm
    if travel < target_travel:
        return (target_travel - travel) ** 4  # Sharp increase
    else:
        return np.log1p(travel - target_travel) ** 2  # Gentle increase


def fallAccelerationMetric(F_y: np.ndarray) -> float:
    """
    Measures the closeness of the average acceleration within a specific y range
    to the acceleration due to gravity. The y range is from (y_min + 0.8 * travel)
    to (y_min + 0.2 * travel), handling cyclic wrapping of the array.

    Parameters:
    ----------
    F_y : np.ndarray
        Array of y-coordinates corresponding to each degree of crank angle (shape: (360,)).

    Returns:
    -------
    float
        The absolute difference between the average acceleration and gravity.
    """
    y_min = np.min(F_y)
    y_max = np.max(F_y)
    travel = y_max - y_min
    lower_bound = y_min + 0.2 * travel
    upper_bound = y_min + 0.8 * travel

    # Identify indices within the y range
    indices = np.where((F_y >= lower_bound) & (F_y <= upper_bound))[0]
    if indices.size == 0:
        return np.inf  # Penalize if no points are in the range

    # Calculate velocity and acceleration
    delta_t = 1 / 72  # seconds
    velocity = np.gradient(F_y, delta_t)
    acceleration = np.gradient(velocity, delta_t)

    # Handle cyclic indices
    acceleration_in_range = acceleration[indices]
    avg_acceleration = np.mean(acceleration_in_range)

    g = 9.81  # m/s^2
    return abs(avg_acceleration - g)


def riseAccelerationMetric(F_y: np.ndarray) -> float:
    """
    Evaluates the peak acceleration within a specific y range.
    Lower peak accelerations are better.

    Parameters:
    ----------
    F_y : np.ndarray
        Array of y-coordinates corresponding to each degree of crank angle (shape: (360,)).

    Returns:
    -------
    float
        The peak acceleration within the specified y range.
    """
    y_min = np.min(F_y)
    y_max = np.max(F_y)
    travel = y_max - y_min
    lower_bound = y_min + 0.2 * travel
    upper_bound = y_min + 0.8 * travel

    # Identify indices within the y range
    indices = np.where((F_y >= lower_bound) & (F_y <= upper_bound))[0]
    if indices.size == 0:
        return np.inf  # Penalize if no points are in the range

    # Calculate velocity and acceleration
    delta_t = 1 / 72  # seconds
    velocity = np.gradient(F_y, delta_t)
    acceleration = np.gradient(velocity, delta_t)

    # Handle cyclic indices
    acceleration_in_range = acceleration[indices]
    peak_acceleration = np.max(np.abs(acceleration_in_range))

    return peak_acceleration
