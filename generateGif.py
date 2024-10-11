import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import tempfile


def thin_matrix(matrix):
    matrix = np.array(matrix)  # Убедимся, что это NumPy массив
    return matrix[::10]


def create_animated_gif(
    _X, _Y, arr1, arr2, output_file="animation.gif", duration=2, text: str = ""
):
    """
    Создает анимированный GIF из координат точек и соединений между ними.

    :param X: numpy-массив координат X, форма (T, N)
    :param Y: numpy-массив координат Y, форма (T, N)
    :param arr1: список индексов первой точки для соединения
    :param arr2: список индексов второй точки для соединения
    :param output_file: имя выходного файла
    :param duration: продолжительность анимации в секундах
    """
    X = thin_matrix(_X)
    Y = thin_matrix(_Y)
    num_frames = X.shape[0]
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    # Настройка графика
    fig, axes = plt.subplots(3, 1)
    ax: plt.Axesa = axes[0]
    ax_godograph: plt.Axes = axes[1]
    ax_text: plt.Axes = axes[2]
    ax.set_aspect("equal")
    # Определение границ графика
    all_x = X.flatten()
    all_y = Y.flatten()
    yMin = all_y.min()
    yMax = all_y.max()
    margin = 0.2 * (yMax - yMin)

    def createText():
        ax_text.clear()
        ax_text.axis((0, 10, 0, 10))
        ax_text.text(0.2, 9, text, wrap=True)

    def setupMechanism():
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(yMin - margin, yMax + margin)
        ax.axis("off")

    def drawGodograph():
        ax_godograph.clear()
        ax_godograph.set_aspect("equal")
        ax_godograph.set_xlim(0, 360)
        scale = 360 / (yMax - yMin)
        ax_godograph.plot(timespace, y_F * scale)
        ax_godograph.axis("off")
        ax_godograph.grid(True, "both", "both")
        y_F_max = y_F.max()
        y_F_min = y_F.min()

        threshold = y_F_min + ((y_F_max - y_F_min) / 2)
        ax_godograph.plot([0, 360], [threshold * scale] * 2, c="black")

        lowerBound = np.full(360, threshold * scale)
        upperBound = np.maximum(lowerBound, y_F * scale)
        ax_godograph.fill_between(timespace, lowerBound, upperBound, color="#66ff77")

    timespace = np.linspace(0, 359, 360)
    y_F: np.ndarray = _Y[:, 5]

    for t in range(num_frames):
        setupMechanism()
        drawGodograph()
        createText()

        # Рисование линий
        for i, j in zip(arr1, arr2):
            ax.plot([X[t, i], X[t, j]], [Y[t, i], Y[t, j]], "k-", linewidth=1)

        # Рисование точек
        ax.scatter(X[t][:4], Y[t][:4], s=50, c="blue", edgecolors="k")
        ax.scatter(X[t][4], Y[t][4], s=50, c="red", edgecolors="k")

        ax_godograph.scatter(
            [timespace[t * 10]], [_Y[t * 10, 5] * 360 / (yMax - yMin)], c="blue"
        )

        # Сохранение кадра
        frame_path = os.path.join(temp_dir, f"frame_{t}.png")
        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        frame_files.append(frame_path)

    plt.close(fig)

    # Создание GIF
    images = []
    for frame in frame_files:
        images.append(imageio.imread(frame))
    imageio.mimsave(
        output_file,
        images,
        duration=duration / num_frames,
    )

    # Очистка временных файлов
    for frame in frame_files:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"Анимированный GIF сохранен как {output_file}.")
