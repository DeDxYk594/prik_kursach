import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import tempfile


def thin_matrix(matrix):
    matrix = np.array(matrix)  # Убедимся, что это NumPy массив
    return matrix[::10]


def create_animated_gif(X, Y, arr1, arr2, output_file="animation.gif", duration=2):
    """
    Создает анимированный GIF из координат точек и соединений между ними.

    :param X: numpy-массив координат X, форма (T, N)
    :param Y: numpy-массив координат Y, форма (T, N)
    :param arr1: список индексов первой точки для соединения
    :param arr2: список индексов второй точки для соединения
    :param output_file: имя выходного файла
    :param duration: продолжительность анимации в секундах
    """
    X = thin_matrix(X)
    Y = thin_matrix(Y)
    num_frames = X.shape[0]
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    # Настройка графика
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # Определение границ графика
    all_x = X.flatten()
    all_y = Y.flatten()
    margin = 1
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    plt.axis("off")  # Отключить оси

    for t in range(num_frames):
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        plt.axis("off")

        # Рисование линий
        for i, j in zip(arr1, arr2):
            ax.plot([X[t, i], X[t, j]], [Y[t, i], Y[t, j]], "k-", linewidth=1)

        # Рисование точек
        ax.scatter(X[t], Y[t], s=50, c="blue", edgecolors="k")

        # Сохранение кадра
        frame_path = os.path.join(temp_dir, f"frame_{t}.png")
        plt.savefig(frame_path, bbox_inches="tight", pad_inches=0)
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


# Пример использования
if __name__ == "__main__":
    # Пример данных
    T = 60  # Количество кадров (например, 60 кадров для 2 секунд при 30 FPS)
    N = 5  # Количество точек

    # Генерация случайных движений точек
    np.random.seed(0)
    X = np.cumsum(np.random.randn(T, N), axis=0)
    Y = np.cumsum(np.random.randn(T, N), axis=0)

    # Определение соединений
    arr1 = [0, 1, 2]
    arr2 = [1, 2, 3]

    create_animated_png(X, Y, arr1, arr2, output_file="animation.png", duration=2000)
