import numpy as np
import cv2

def constraint(prob=0.5, thres=0.3):
    # set xác suất thấp nhất để loại bỏ dự đoán yếu
    probability_minium = prob
    # Cài đặt threshold để lọc các bouding box yếu với non-maximum suppression.
    threshold = thres
    return probability_minium, threshold

def get_layers_names(network):
    layers_names_all = network.getLayerNames()
    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    return layers_names_output

def load_model(config_file,weights_file, classes_file):
    network = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    layers_names_output = get_layers_names(network)
    with open(classes_file) as f:
        labels = [line.strip() for line in f]
    return network, layers_names_output, labels

def prediction(img_source, network, layers_names_output, labels, prob=0.5, threshold=0.3, predicted_labels_returns=False):
    images = cv2.imread(img_source)
    h, w = images.shape[:2]

    # blob from images - get blob after mean substraction, normalizing, RB channel swapping
    blob = cv2.dnn.blobFromImage(images, 1/255.0, (416,416), swapRB=True, crop=False)

    # cài đặt ràng buộc
    probability_minium, threshold = constraint(prob,threshold)
    colours = np.random.randint(0,255, size=(len(labels), 3), dtype='uint8')

    # dùng mô hình đã được load để predict.
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)

    # gán nhãn các bouding box
    bounding_boxes = []
    confidences = []
    class_numbers = []
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minium:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minium, threshold)

    # vẽ các bouding box lên hình.
    counter = 0
    if len(results) > 0:
        for i in results.flatten():
            counter += 1
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()

            cv2.rectangle(images, (x_min,y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)
            text_box_current = '{} : {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])
            cv2.putText(images, text_box_current, (x_min, y_min - 5),cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    if predicted_labels_returns:
        return images, counter, bounding_boxes, confidences, class_numbers
    return images, counter, bounding_boxes


# if __name__ == '__main__' :
#     img_url = 'images/8de03901c64a80070048ead3fb0d32bd_jpg.rf.d4cd4d7336cee96c7c01731c6f3b81cd.jpg'
#     config_file = 'yolo-chess-data/yolov3.cfg'
#     weight_file = 'yolo-chess-data/yolov3_1200.weights'
#     classes_file = 'yolo-chess-data/classes.txt'
#
#     network, layer_output, labels = load_model(config_file,weight_file,classes_file)
#     images, counter, _ = prediction(img_url,network, layer_output, labels)
#
#     # show hình ảnh
#     print('Number of object: ', counter)
#     cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
#     cv2.imshow('Detections', images)
#     cv2.waitKey(0)
#     cv2.destroyWindow('Detections')
