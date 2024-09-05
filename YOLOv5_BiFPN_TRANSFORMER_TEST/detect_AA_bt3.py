import torch
import os
import numpy as np

predictions_folder = '/cs/home/psxss39/gwhd_results/runs/detect/exp_5x_bt3/labels'
targets_folder = '/cs/home/psxss39/gwhd_dataset/test/labels'

def load_labels_from_file(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, x, y, w, h  = map(float, parts[:5])
                boxes.append([class_id, x, y, w, h])
    return np.array(boxes)

def convert_to_xyxy(labels):
    converted_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        converted_labels.append([class_id, x1, y1, x2, y2])
    return np.array(converted_labels)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_boxes(predictions, targets):
    num_detections = len(predictions)
    num_ground_truths = len(targets)

    iou_matrix = np.zeros((num_detections, num_ground_truths))

    for i, dbox in enumerate(predictions):
        for j, gtbox in enumerate(targets):
            iou_matrix[i, j] = calculate_iou(dbox, gtbox)

    matched_indices = []

    for i in range(min(num_detections, num_ground_truths)):
        max_index = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        matched_indices.append((max_index[0], max_index[1]))  
        iou_matrix[max_index[0], :] = -1  
        iou_matrix[:, max_index[1]] = -1  

    return matched_indices

def compute_metrics(predictions, targets, matched_indices, iou_threshold=0.45):
    TP = FP = FN = 0
    matched_targets = set()

    for pred_idx, target_idx in matched_indices:
        p_box = predictions[pred_idx]
        t_box = targets[target_idx]
        iou = calculate_iou(p_box, t_box)
        if iou >= iou_threshold:
            TP += 1
            matched_targets.add(tuple(t_box))
        else:
            FP += 1

    unmatched_predictions = set(range(len(predictions))) - {pred_idx for pred_idx, _ in matched_indices}
    FP += len(unmatched_predictions)

    unmatched_targets = set(range(len(targets))) - {target_idx for _, target_idx in matched_indices}
    FN = len(unmatched_targets)

    return TP, FP, FN

def calculate_aa(predictions, targets, matched_indices):
    TP, FP, FN = compute_metrics(predictions, targets, matched_indices)
    if (TP + FP + FN) == 0:
        return 0.0
    return TP / (TP + FP + FN)
	
all_precisions, all_recalls, all_mAPs = [], [], []
overall_aa = []

pred_files = sorted(os.listdir(predictions_folder))
target_files = sorted(os.listdir(targets_folder))

for image_index, (pred_file, target_file) in enumerate(zip(pred_files, target_files)):
    pred_path = os.path.join(predictions_folder, pred_file)
    target_path = os.path.join(targets_folder, target_file)

    predictions = load_labels_from_file(pred_path)
    targets = load_labels_from_file(target_path)

    predictions_xyxy = convert_to_xyxy(predictions)
    targets_xyxy = convert_to_xyxy(targets)

    matched_indices = match_boxes(predictions_xyxy, targets_xyxy)

    # Now you can calculate IoU using the reordered_detected_boxes
    aa = calculate_aa(predictions_xyxy, targets_xyxy, matched_indices)
    print("Average Accuracy (AA):", aa)

    overall_aa.append(aa)

mean_aa = np.mean(overall_aa)

print("Mean Average Accuracy (AA):", mean_aa)