import string
import numpy as np
from predict import predict


OCR_CLASSES = list(string.ascii_uppercase) + [str(num) for num in range(10)]

# dictionary holding class_label:class_name mapping
OCR_CLASS_MAPPING = {k: v for k, v in enumerate(OCR_CLASSES)}


def calculate_iou(box1, box2):
    # calculate a vector of iou between a separate box(box1) and a set of boxes(box2)
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    # calculate coordinates of intersection
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # calculate the intersection area, taking into account no intersection case
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def torch_like_nms(boxes, scores, iou_threshold):
    # This implementation of non-max suppression follows the
    # description of the torchvision.ops.nms function, which
    # is used under the hood of ultralytics.utils.ops.non_max_suppression.

    # get indices of the scores array sorted in descending order.
    order = np.argsort(-scores)

    # generate array of indices from 0 to boxes.shape[0] - 1.
    indices = np.arange(boxes.shape[0])

    # create a boolean array of the same shape as indices initially filled with True.
    # i-th entry in this array indicates whether we keep i-th bounding box or not.
    keep = np.ones_like(indices, dtype=np.bool_)

    # loop over indices
    for i in indices:
        # enter only if we keep i-th bounding box
        if keep[i]:
            # retrieve bounding box, which corresponds to the i-th largest confidence score
            bbox = boxes[order[i]]

            # calculate iou of the obtained bounding box with all the others
            iou = calculate_iou(bbox[np.newaxis, :], boxes[order[i+1:]] * keep[i+1:][:, np.newaxis])

            # create array of indices which indicate locations where the specified condition holds True
            overlapped = np.nonzero(iou > iou_threshold)

            # filter out bounding boxes using the obtained indices
            keep[overlapped + i + 1] = 0

    # return indices of bounding boxes to keep
    return order[keep]


def non_max_suppression(prediction, conf_thresh=0.2, iou=0.7, max_det=300, nc=36):
    # get batch size, e.g. 1x40x336 -> bs = 1
    bs = prediction.shape[0]

    # get number of boxes + classes
    mi = 4 + nc

    # for prediction of shape 1x40x336, get only classes(4:40 slice)
    # and calc max values over this dimension. Next obtain a boolean
    # mask by applying '> conf_thresh' to the result. This step can
    # be referred to as confidence thresholding. The shape of xc
    # should be 1x336
    xc = prediction[:, 4:mi].max(axis=1) > conf_thresh

    # in the prediction matrix, permute the dimensions. This step
    # transforms 1x40x336 to 1x336x40.
    prediction = prediction.transpose(0, -1, -2)

    # convert xywh coordinates to xyxy. Since first 2 coordinates
    # are already in the desired format we only need to transform
    # w and h. That's why for every entry of [w, h] pair we add
    # corresponding [x, y] pair.
    prediction[..., 2:4] += prediction[..., :2]

    # create output matrix of size bsx6, which stores
    # (x_min, y_min, x_max, y_max, conf, class_label)
    output = [np.zeros((0, 6))] * bs

    # traverse the prediction matrix. xi corresponds to current
    # index e.g. xi = 0, 1, 2..., while x corresponds to contents
    # at index xi.
    for xi, x in enumerate(prediction):

        # extract predictions with confidence higher than conf_thresh.
        # For reference, during my run, the shape of x was 59x40. The
        # first dimension shouldn't necessarily be the same in your case,
        # but the second should.
        x = x[xc[xi]]

        # if matrix x has 0 rows, continue,
        # because there are no detections.
        if not x.shape[0]:
            continue

        # extract bounding box coordinates and class confidences from x.
        # If the shape of x is 59x40, the shapes of box and are 59x4 and
        # 59x36 correspondingly.
        box, cls = np.split(x, [4], 1)

        # get max class confidences over class dimension. If the shape of
        # cls is 59x36, the shape of conf will be 59x1. Vector j stores
        # indices of max confidences over class dimension.
        conf, j = cls.max(axis=1, keepdims=True), cls.argmax(axis=1, keepdims=True)

        # concatenate obtained boxes, class confidences and class labels
        # into a single matrix. If the shapes of box, conf and j are 59x4,
        # 59x1, 59x1, the result of concatenation would have shape of 59x6.
        # Finally, subsample rows from the resulting matrix, where class
        # confidences are greater than conf_thresh. This is done using
        # '[conf.flatten() > conf_thresh]'. This step doesn't necessarily
        # change the number of rows in the resulting matrix.
        x = np.concatenate((box, conf, j), axis=1)[conf.flatten() > conf_thresh]

        # extract bounding boxes and confidence scores
        scores = x[:, 4]
        boxes = x[:, :4]

        # apply non-max suppression to obtained boxes and scores.
        # The nms function returns indices of boxes to keep.
        i = torch_like_nms(boxes, scores, iou)

        # limit the number of detections to 300
        i = i[:max_det]

        # extract valid detections using the obtained indices
        output[xi] = x[i]
        return output


def order_classes(detections):
    # calculate differences between min and max
    # coordinates of detected boxes over x and y axes
    delta_x = detections[:, 2].max() - detections[:, 0].min()
    delta_y = detections[:, 3].max() - detections[:, 1].min()

    # boxes should be ordered horizontally
    if delta_x > delta_y:
        sorted_detections = detections[detections[:, 0].argsort()]
    # boxes should be ordered vertically
    else:
        sorted_detections = detections[detections[:, 1].argsort()]

    # decode class_labels into class_names and return the string
    return " ".join([OCR_CLASS_MAPPING[class_label] for class_label in sorted_detections[:, 5]])


if __name__ == '__main__':
    out = predict("dataset_0_train_1_SERIAL_NUMBER_H.jpg", "ocr_model.tflite")
    out = non_max_suppression(out)[0]
    print(order_classes(out))
