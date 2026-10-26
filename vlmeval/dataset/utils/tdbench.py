import re


def extract_bbox_from_string(bbox_str):
    bbox_str = bbox_str.replace('\n', '')
    parsed = re.findall(r'(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?)', bbox_str)
    if len(parsed) == 1:
        return float(parsed[0][0]), float(parsed[0][1]), float(parsed[0][2]), float(parsed[0][3])
    else:
        raise RuntimeError(f'Invalid VLM output: {bbox_str}. '
                           f'Correct coordinate should be [a, b, c, d], where each number is 0, 0.*, or 1. ')


def calculate_bbox_iou(pred_bbox, gt_bbox):
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_bbox
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox

    x_min_intersect = max(pred_x_min, gt_x_min)
    y_min_intersect = max(pred_y_min, gt_y_min)
    x_max_intersect = min(pred_x_max, gt_x_max)
    y_max_intersect = min(pred_y_max, gt_y_max)

    if x_max_intersect < x_min_intersect or y_max_intersect < y_min_intersect:
        return 0.0

    intersection_area = (x_max_intersect - x_min_intersect) * (y_max_intersect - y_min_intersect)

    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)

    union_area = pred_area + gt_area - intersection_area

    iou = intersection_area / union_area

    return iou


def calculate_centroid_containment(pred_bbox, gt_bbox):
    label_x0, label_y0, label_x1, label_y1 = gt_bbox
    pred_x0, pred_y0, pred_x1, pred_y1 = pred_bbox
    pred_center_x = (pred_x0 + pred_x1) / 2
    pred_center_y = (pred_y0 + pred_y1) / 2
    if (label_x0 <= pred_center_x <= label_x1) and (label_y0 <= pred_center_y <= label_y1):
        return 1
    else:
        return 0


def evaluate_bbox(pred_bbox, gt_bbox, method):
    if method == 'centroid':
        return calculate_centroid_containment(pred_bbox, gt_bbox)
    elif method == 'iou':
        return calculate_bbox_iou(pred_bbox, gt_bbox)
