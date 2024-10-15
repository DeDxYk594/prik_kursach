import numpy as np


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
    def generateGif(params: np.array, saveTo: str) -> None:
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
        raise NotImplementedError()

    @staticmethod
    def isRevolutable(params: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def simulate(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @staticmethod
    def generateGif(params: np.array, saveTo: str) -> None:
        raise NotImplementedError()


class PrismaticPairPress:
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
    def generateGif(params: np.array, saveTo: str) -> None:
        raise NotImplementedError()
