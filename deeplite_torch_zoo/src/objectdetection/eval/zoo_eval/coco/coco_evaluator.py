import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from deeplite_torch_zoo.src.objectdetection.eval.zoo_eval.evaluator import \
    Evaluator


class COCOEvaluator(Evaluator):
    def __init__(
        self,
        model,
        dataset,
        visualize=False,
        net="yolo3",
        img_size=448,
        gt=None,
        progressbar=False,
    ):
        super(COCOEvaluator, self).__init__(
            model=model, img_size=img_size
        )
        self.dataset = dataset
        self.progressbar = progressbar

        self.classes = self.dataset.classes
        self.__visiual = visualize
        self.__visual_imgs = 0

        self.cocoGt = gt

    def evaluate(self, multi_test=False, flip_test=False):
        results = []
        for img, _, _, img_ind in tqdm(self.dataset, disable=not self.progressbar):
            results += self.process_image(img, int(img_ind))

        results = np.array(results).astype(np.float32)
        if len(results) == 0:
            return {"mAP": 0}
        cocoDt = self.cocoGt.loadRes(results)
        E = COCOeval(self.cocoGt, cocoDt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))
        return {"mAP": E.stats[0]}

    def process_image(self, img, **kwargs):
        pass


class YoloCOCOEvaluator(COCOEvaluator):
    def __init__(self, model, dataset, gt=None, visualize=False,
        net="yolo3", img_size=448, progressbar=False):
        super().__init__(
            model=model,
            dataset=dataset,
            gt=gt,
            visualize=visualize,
            net=net,
            img_size=img_size,
            progressbar=progressbar,
        )

    def process_image(self, img, img_ind, multi_test=False, flip_test=False, **kwargs):

        bboxes_prd = self.get_bbox(img, multi_test, flip_test)

        results = []
        for bbox in bboxes_prd:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = self.classes[class_ind]

            xmin, ymin, xmax, ymax = coor
            results.append(
                [
                    img_ind,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    score,
                    self.dataset.add_coco_empty_category(class_ind),
                ]
            )
        return results


class SSDCOCOEvaluator(COCOEvaluator):
    def __init__(self, model, dataset, gt=None, net="ssd", predictor=None, img_size=300):
        super().__init__(
            model=model,
            dataset=dataset,
            net=net,
            img_size=img_size,
            gt=gt,
        )
        self.predictor = predictor

    def process_image(self, img, img_id):
        boxes, labels, probs = self.predictor.predict(img)
        results = []
        for bbox, label, prob in zip(boxes, labels, probs):
            xmin, ymin, xmax, ymax = bbox
            results.append(
                [
                    img_id,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    prob,
                    self.dataset.add_coco_empty_category(label) - 1,
                ]
            )
        return results


def ssd_eval_coco(model, data_loader, gt=None, predictor=None, device="cuda", net="ssd"):
    model.to(device)
    with torch.no_grad():
        return SSDCOCOEvaluator(
            model,
            data_loader.dataset,
            gt=gt,
            predictor=predictor,
            net=net
        ).evaluate()
