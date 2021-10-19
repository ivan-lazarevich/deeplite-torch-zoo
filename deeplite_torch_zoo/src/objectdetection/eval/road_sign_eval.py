import cv2
import numpy as np
import torch

from deeplite_torch_zoo.src.objectdetection.datasets.road_sign_detection import (
    RoadSignDetectionDataset,
)
from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.eval.metrics import MAP


class RoadSignDetectionEval(Evaluator):
    """Road sign detection dataset evaluator object"""

    def __init__(self, model, data_root, net="yolov3", img_size=448):
        data_path = "deeplite_torch_zoo/results/road_sign_detection/{net}".format(
            net=net
        )
        super(RoadSignDetectionEval, self).__init__(
            model=model, data_path=data_path, img_size=img_size, net=net
        )

        self.dataset = RoadSignDetectionDataset(data_root, split="val")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        self.model.eval()
        self.model.cuda()
        results = []
        for img_idx, _ in enumerate(self.dataset.img_info):
            _info = self.dataset.img_info[img_idx]
            img_path = _info["img_path"].as_posix().replace('\n', '')
            image = cv2.imread(img_path)

            print("Parsing batch: {}/{}".format(img_idx, len(self.dataset)), end="\r")
            bboxes_prd = self.get_bbox(image)

            if len(bboxes_prd) == 0:
                bboxes_prd = np.zeros((0, 6))

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            detections = {"bboxes": [], "labels": []}
            detections["bboxes"] = bboxes_prd[:, :5]
            detections["labels"] = bboxes_prd[:, 5]

            gt_bboxes = _info["annotations"]["bbox"]
            gt_bboxes = [
                (
                    x1 * image.shape[1],
                    y1 * image.shape[0],
                    x2 * image.shape[1],
                    y2 * image.shape[0],
                )
                for (x1, y1, x2, y2) in gt_bboxes
            ]
            gt_labels = _info["annotations"]["labels"]

            gt = {"bboxes": gt_bboxes, "labels": gt_labels}
            results.append({"detections": detections, "gt": gt})

        mAP = MAP(results, self.dataset.num_classes)
        mAP.evaluate()
        ap = mAP.accumlate()
        ap = ap[ap > 1e-6]
        _ap = np.mean(ap)
        print("mAP = {:.3f}".format(_ap))
        return _ap  # Average Precision  (AP) @[ IoU=050 ]


def yolo_eval_road_sign(
    model, data_root, device="cuda", net="yolov3", img_size=448, **kwargs
):

    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        mAP = RoadSignDetectionEval(
            model, data_root, net=net, img_size=img_size
        ).evaluate()
        result["mAP"] = mAP

    return result
