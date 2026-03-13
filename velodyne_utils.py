import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def read_velodyne_bin(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity'])
    return df

def read_label_file(file_path):
    raw_data = np.fromfile(file_path, dtype=np.uint32)
    class_labels = raw_data & 0xFFFF
    return class_labels

def metrics(pred, gt):
    intersection_g = np.logical_and(pred, gt).sum()
    union_g = np.logical_or(pred, gt).sum()
    iou_g = intersection_g / (union_g)

    pred_ng = ~pred
    gt_ng = ~gt
    intersection_ng = np.logical_and(pred_ng, gt_ng).sum()
    union_ng = np.logical_or(pred_ng, gt_ng).sum()
    iou_ng = intersection_ng / (union_ng)

    miou = (iou_g + iou_ng) / 2

    f1 = f1_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)

    return miou, f1, precision, recall