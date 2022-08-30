import YOLO as yolo
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse

# How to run
# python Evaluation.py -o '/Users/trunghieu/Desktop/HCMUS/HocThongKe/YOLO-3/' \
#                      -i 'images/' \
#                      -l 'labels/' \
#                      -c 'yolo-chess-data/yolov3.cfg' \
#                      -w 'yolo-chess-data/yolov3_1200.weights' \
#                      -cl 'yolo-chess-data/classes.txt' \

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--original', required=True,
                help='orginal absolute path with ending in "/" ')
ap.add_argument('-i', '--images', required=True,
                help='images relative path with ending in "/"')
ap.add_argument('-l', '--labels', required=True,
                help='labels relative path with ending in "/"')
ap.add_argument('-c', '--config', required=True,
                help='relative path  to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='relative path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='relative path to text file containing class names')
args = ap.parse_args()

def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
                              min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union

    return iou


# cần absolute path
origin_path = args.original
config_file = origin_path + args.config
weight_file = origin_path + args.weights
classes_file = origin_path + args.classes

a,b,c = yolo.load_model(config_file,weight_file,classes_file)

# Khởi tạo các biến cần thiết cho việc đánh giá
positive_prediction_class, negative_prediction_class = {}, {}
class_name = []
IoU_avg = []
with open(classes_file, 'r') as f:
    for line in f.readlines():
        positive_prediction_class[line.strip()] = 0
        negative_prediction_class[line.strip()] = 0
        class_name.append(line.strip())

# Duyệt từng file để đánh giá.
os.chdir(origin_path + args.images)
for _, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            img_name = f
            images = cv2.imread(img_name)
            h, w = images.shape[:2]

            gt_boxes = []  # ground true boxes
            labels_path = origin_path + args.labels
            with open(labels_path + f.replace('.jpg' or '.jpeg', '') + '.txt') as box:
                for line in box.readlines():
                    gt_boxes.append(line)

                    # prediction box
            _, _, prediction_box, confidences, class_numbers = yolo.prediction(img_name, a, b, c, predicted_labels_returns=True)

            IoU_avg_one_image = []

            if len(gt_boxes) == len(prediction_box):
                for i in range(len(gt_boxes)):
                    true_box = gt_boxes[i].strip().split(' ')[1:]
                    true_box = list(map(float, true_box))
                    x, y, w, h = np.array(true_box) * np.array([w, h, w, h])

                    left = x - int(w / 2)
                    top = y - int(h / 2)
                    right = x + int(w / 2)
                    bottom = y + int(h / 2)

                    ground_true_box = np.array([left, top, w, h])
                    ground_true_box.astype(int)
                    intersection = intersection_over_union(prediction_box[i], ground_true_box)
                    if intersection > 0:
                        # IoU score
                        IoU_avg.append(intersection)

                        # True positive and False positive
                        if intersection >= 0.5 and confidences[i] >= 0.5:
                            positive_prediction_class[class_name[class_numbers[i]]] += 1
                        if (0 <= intersection < 0.5) or confidences[i] < 0.5:
                            negative_prediction_class[class_name[class_numbers[i]]] += 1

print('Avg IoU: ',np.mean(IoU_avg))

width = 0.35
fig, ax = plt.subplots()

ax.barh(list(positive_prediction_class.keys()), list(positive_prediction_class.values()), width, label='True positive')
ax.barh(list(negative_prediction_class.keys()), list(negative_prediction_class.values()), width, label='False positive')

ax.set_ylabel('Class names')
ax.set_title('Count')

for i in ax.patches:
    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='grey')

ax.legend()

plt.show()





