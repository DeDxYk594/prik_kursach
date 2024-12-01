import numpy as np
from simulation import simulatePrismaticPress, simulateRevolutePress
from generateGif import create_animated_gif
from targetFunction import targetFunction


class Press:
    @staticmethod
    def getBounds() -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def isRevolutable(params: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def simulate(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @staticmethod
    def generateGif(params: np.ndarray, saveTo: str) -> None:
        raise NotImplementedError()


class RevolutePairPress:
    """
    l_AB = params[0]           # мм
    l_BC = params[1]           # мм
    l_CD = params[2]           # мм
    l_CE = params[3]           # мм
    l_EF = params[4]           # мм
    delta_1_x = params[5]      # мм
    delta_1_y = params[6]      # мм
    delta_2_x = params[7]      # мм
    alpha_1_angle = params[8]  # градусов
    """

    @staticmethod
    def getBounds() -> np.ndarray:
        return np.array(
            [
                [10, 500],
                [10, 300],
                [10, 300],
                [10, 300],
                [10, 300],
                [10, 300],
                [10, 300],
                [10, 300],
                [-90, 90],
            ]
        )

    @staticmethod
    def isRevolutable(params: np.ndarray) -> bool:
        try:
            x, y = simulateRevolutePress(params, 360)
            if np.isnan(x).any():
                return False
            if np.isnan(y).any():
                return False
            return True
        except ValueError:
            return False

    @staticmethod
    def simulate(params: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        ret = simulateRevolutePress(params, steps)
        if np.isnan(ret[0]).any():
            raise ValueError("NaN somewhere in x")
        if np.isnan(ret[1]).any():
            raise ValueError("NaN somewhere in y")
        return ret

    @staticmethod
    def generateGif(params: np.ndarray, saveTo: str) -> None:
        X, Y = simulateRevolutePress(params, 360)
        tf, desc2 = targetFunction(X, Y)
        desc1 = ",".join(map(str, params))
        create_animated_gif(X, Y, [5], saveTo, desc1 + "\n" + desc2)


class PrismaticPairPress:
    """
    l_AB = params[0]           # мм
    l_BC = params[1]           # мм
    l_CD = params[2]           # мм
    l_CE = params[3]           # мм
    delta_1_x = params[4]      # мм
    delta_1_y = params[5]      # мм
    alpha_1_angle = params[6]  # градусов
    alpha_2_angle = params[7]  # градусов
    """

    @staticmethod
    def getBounds() -> np.ndarray:
        return np.array(
            [
                [10, 500],
                [10, 500],
                [10, 500],
                [10, 500],
                [10, 500],
                [10, 500],
                [-90, 90],
                [-90, 90],
            ]
        )

    @staticmethod
    def isRevolutable(params: np.ndarray) -> bool:
        try:
            simulatePrismaticPress(params, 360)
            return True
        except ValueError:
            return False

    @staticmethod
    def simulate(params: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        ret = simulatePrismaticPress(params, steps)
        if np.isnan(ret[0]).any():
            raise ValueError("NaN somewhere in x")
        if np.isnan(ret[1]).any():
            raise ValueError("NaN somewhere in y")
        return ret

    @staticmethod
    def generateGif(params: np.ndarray, saveTo: str) -> None:
        X, Y = simulatePrismaticPress(params, 360)
        tf, desc2 = targetFunction(X, Y)
        desc1 = ",".join(map(str, params))
        create_animated_gif(X, Y, [5], saveTo, desc1 + "\n" + desc2)
