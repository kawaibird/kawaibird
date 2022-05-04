import numpy as np

np.random.seed(1)


def generate_NMS_sample(num):
    bbox = 100 * np.random.random((num, 4))
    bbox[:, 0] = bbox[:, 0] + 100
    bbox[:, 2] = bbox[:, 2] + 100
    confidence = np.random.random((num, 1))

    return np.concatenate((bbox, confidence), axis=1)


def get_area(box):
    return abs(box[:, 0] - box[:, 1]) * abs(box[:, 2] - box[:, 3])


def get_iou(abox, bbox):
    return 0


def get_batch_iou(batch_bbox, target_bbox):
    # xmin = np.min(batch_bbox[:,0], target_bbox[0], axis=0)
    # xmax = np.max(batch_bbox[:,1], target_bbox[1], axis=0)
    # ymin = np.min(batch_bbox[:,2], target_bbox[2], axis=0)
    # ymax = np.max(batch_bbox[:,3], target_bbox[3], axis=0)
    #

    return np.array([0.0] * len(batch_bbox))


def NMS_algorithm(bboxes, iou_thd=0.7):
    """
    :param bboxes: np.array N x 5 (xyxy conf)
    :param iou_thd: float
    :return:
        keep_bbox : list
    """
    keep_bbox = []
    current_store_bbox = bboxes
    current_store_bbox = current_store_bbox[np.argsort(current_store_bbox[:, 4])]

    while len(current_store_bbox) != 0:
        # conf_max_index = np.argmax(current_store_bbox[1:, 4])
        # max_bbox = bboxes[conf_max_index]
        # keep_bbox.append(max_bbox)

        batch_iou = get_batch_iou(current_store_bbox[1:], max_bbox)
        batch_iou[conf_max_index] = 1.0

        store_idx_list = batch_iou < iou_thd

        current_store_bbox = current_store_bbox[store_idx_list]
    return keep_bbox


if __name__ == '__main__':
    a = generate_NMS_sample(100)
    print(a.shape)
    result = NMS_algorithm(a)
    print(len(result))
    # help(NMS_algorithm)
