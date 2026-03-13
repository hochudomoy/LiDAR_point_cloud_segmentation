import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN

def ground_ransac(xyz,max_trials=1000,height_percentile=40, residual_threshold=0.2):
    """Перед применением фильтруем по высоте, выбираем 40% самых низких точек
       Алгоритм:
       1)Случайным образом выбирает 3 точки, по которым строит плоскость
       2)Считает, какие точкки относятся к построенной плоскости(расстояние меньше,чем порог residual_threshold)
       3)Повторяет шаги max_trials раз, находит плоскость с наибольшим количеством точек, относящихся к плоскости"""

    z_limit = np.percentile(xyz['z'], height_percentile)
    mask_z = (xyz['z'] <= z_limit)
    xyz_filtered = xyz[mask_z]
    xy = xyz_filtered[['x','y']]
    z = xyz_filtered['z']
    model = RANSACRegressor(
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )
    model.fit(xy, z)
    inliers_filtered = model.inlier_mask_
    inliers = np.zeros(len(xyz), dtype=bool)
    inliers[mask_z.values] = inliers_filtered
    return inliers


def ground_grid_ransac(xyz, grid_size=20.0, residual_threshold=0.2, max_trials=50):
    """Делим плоскость XY на блоки 20х20 в виде сетки, применяем RANSAC для каждого блока"""
    inliers = np.zeros(len(xyz), dtype=bool)
    x_min, y_min = xyz['x'].min(), xyz['y'].min()
    x_idx = ((xyz['x'] - x_min) // grid_size)
    y_idx = ((xyz['y'] - y_min) // grid_size)
    unique_blocks = set(zip(x_idx, y_idx))
    unique_blocks = set(zip(x_idx, y_idx))

    for bx, by in unique_blocks:
        block_mask = (x_idx == bx) & (y_idx == by)
        block_points = xyz[block_mask]

        if len(block_points) < 3:
            continue

        xy = block_points[['x', 'y']]
        z = block_points['z']

        model = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
        model.fit(xy, z)
        inliers_block = model.inlier_mask_
        inliers[block_mask.values] = inliers_block

    return inliers


def beaton_tukey_weights(residuals, c=1):
    w = np.zeros_like(residuals)
    mask = np.abs(residuals) <= c
    r = residuals[mask] / c
    w[mask] = (1 - r ** 2) ** 2
    return w


def iterative_ground_filtering(xyz, grid_size=5.0, max_iter=10, hag_threshold=0.2):
    """1) Облако точек разбивается на регулярную сетку
       2) В каждом блоке сохраняются точки с минимальной высотой, остальные точки временно игнорируются
       3) Для оставшихся точек строится предпологаемая плоскость z=ax+by+c(коэффициенты находятся методом наименьших квадратов)
       4) Для остальных точек считаем отклонение высоты от предпологаемой плоскости(HAG)
       5) В зависимости от HAG каждой точке назначается вес в соответсвии с функцией Beaton–Tukey robust weighting
       6) Фильтруем по значению HAG и значению веса
       7) Алгоритм повторяется пока не будет найдено устойчивое решение или пока не достигнуто макисмальное количество итераций"""
    n = len(xyz)
    inliers = np.zeros(len(xyz), dtype=bool)
    x_min, y_min = xyz['x'].min(), xyz['y'].min()
    x_idx = ((xyz['x'] - x_min) // grid_size)
    y_idx = ((xyz['y'] - y_min) // grid_size)
    unique_blocks = set(zip(x_idx, y_idx))
    for bx, by in unique_blocks:
        block_mask = (x_idx == bx) & (y_idx == by)
        block_points = xyz[block_mask]
        if len(block_points) == 0:
            continue
        block_indices = np.where(block_mask)[0]
        min_local_idx = np.argmin(xyz['z'][block_mask])
        min_global_idx = block_indices[min_local_idx]
        inliers[min_global_idx] = True

    for iteration in range(max_iter):
        Xg = np.column_stack((xyz['x'][inliers], xyz['y'][inliers], np.ones(inliers.sum())))
        Zg = xyz['z'][inliers]
        coeffs, _, _, _ = np.linalg.lstsq(Xg, Zg, rcond=None)
        a, b, c = coeffs
        Z_pred = a * xyz['x'] + b * xyz['y'] + c
        residuals = xyz['z'] - Z_pred

        weights = beaton_tukey_weights(residuals)

        ground_mask = (weights > 0.5) | (np.abs(residuals) < hag_threshold)

        if np.array_equal(ground_mask, inliers):
            break
        inliers = ground_mask
    return inliers


def ground_dbscan(xyz, eps=0.7, min_samples=30, height_percentile=35):
    """1) Выбираем 35% самых низких точек
       2) Их кластеризуем, используя DBSCAN
       3) Выбираем кластер с самым большим количеством точек в качестве земли"""
    inliers = np.zeros(len(xyz), dtype=bool)
    points = np.column_stack((xyz['x'], xyz['y'], xyz['z']))
    z_limit = np.percentile(xyz['z'], height_percentile)
    low_mask = (xyz['z'] <= z_limit)
    filtered_points = points[low_mask]
    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1).fit(filtered_points)
    labels = clustering.labels_
    valid = (labels != -1)
    unique_labels = np.unique(labels[valid])
    best_label = unique_labels[0]
    max_size = 0
    for l in unique_labels:
        cluster_size = np.sum(labels == l)
        if cluster_size > max_size:
            max_size = cluster_size
            best_label = l

    inliers[low_mask] = (labels == best_label)
    return inliers

def ground_neighbours_grid_filter(xyz, grid_size=1, threshold=0.2):
    """Разбиваем облако точек на сетку, в каждой ячейке сохраняем минимальное значение высоты
       Для каждой ячейки находится минимальное значение высот её соседей, вклюяая её саму.
       Каждая точка фильтруется по принципу, если её высота меньше равна минимального значения высот соседних ячеек + treshold, то точка принадлежит земле"""
    inliers=np.zeros(len(xyz), dtype=bool)
    x_min, y_min = xyz['x'].min(), xyz['y'].min()
    x_idx = ((xyz['x'] - x_min) // grid_size).astype(int)
    y_idx = ((xyz['y'] - y_min) // grid_size).astype(int)
    nx, ny = int(x_idx.max()) + 1, int(y_idx.max()) + 1
    z_grid = {}
    for xi, yi, zi in zip(x_idx, y_idx, xyz['z']):
        key = (int(xi), int(yi))
        if key not in z_grid or zi < z_grid[key]:
            z_grid[key] = zi
    neighbor_min_grid = {}
    for key in z_grid:
        i,j=key
        min_z = z_grid[key]
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                    neighbor = (i + di, j + dj)
                    if neighbor in z_grid:
                        min_z = min(min_z, z_grid[neighbor])
            neighbor_min_grid[key] = min_z
    for idx, (xi, yi, zi) in enumerate(zip(x_idx, y_idx, xyz['z'])):
        key = (int(xi), int(yi))
        if zi <= neighbor_min_grid[key] + threshold:
            inliers[idx] = True
    return inliers