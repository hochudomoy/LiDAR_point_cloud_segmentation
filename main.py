from velodyne_utils import read_velodyne_bin, read_label_file, metrics
from visualization import visualization_2D, visualization_3D
import ground_filtering
import SalsaNext.inference
import SegFormer.inference
import os
import numpy as np
import pandas as pd
import time

test_folders = [
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\07\\velodyne',
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\09\\velodyne',
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\10\\velodyne'
]
ground_classes=[40, 44, 48, 49]#40: "road" 44: "parking" 48: "sidewalk" 49: "other-ground"]
results=[]
for folder in test_folders:
    files=os.listdir(folder)
    miou, f1, precision, recall, total_time = 0, 0, 0, 0, 0
    for i in files:
        filename = f'{folder}\\{i}'
        lidar_df = read_velodyne_bin(filename)
        class_labels = read_label_file(
            f"C:\\Users\\User\\LiDAR_point_cloud_segmentation\\labels\\{folder.split('\\')[-2]}\\labels\\{i.split('.')[0]}.label")
        gt_mask = np.isin(class_labels, list(ground_classes))
        start = time.time()
        geometric_pred = ground_filtering.ground_neighbours_grid_filter(lidar_df)
        _,prob = SalsaNext.inference.SalsaNext(lidar_df)
        end = time.time()
        final_pred = geometric_pred
        final_pred[prob > 0.9] = 1
        final_pred[prob < 0.1] = 0

        total_time += (end - start)
        metric = metrics(final_pred, gt_mask)
        miou += metric[0]
        f1 += metric[1]
        precision += metric[2]
        recall += metric[3]
        break
    n = len(files)
    results.append([folder.split('\\')[-2], miou / n, f1 / n, precision / n, recall / n, total_time / n])
    df = pd.DataFrame(results, columns=[
        "Sequence",
        "mIoU",
        "F1-score",
        "Precision",
        "Recall",
        "Time per frame"
    ])
print(df)

visualization_2D(lidar_df, color=final_pred)
visualization_3D(lidar_df, final_pred)



