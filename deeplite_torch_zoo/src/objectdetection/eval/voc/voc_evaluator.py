import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET

import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_voc as cfg
from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY

from mean_average_precision import MetricBuilder


class VOCEvaluator(Evaluator):
    def __init__(
        self,
        model,
        data_root,
        num_classes=20,
        img_size=448,
        conf_thresh=0.001,
        nms_thresh=0.5,
        is_07_subset=False,
        progressbar=False,
        class_names=None,
    ):

        super(VOCEvaluator, self).__init__(
            model=model, img_size=img_size, conf_thresh=conf_thresh, nms_thresh=nms_thresh,
        )

        self.is_07_subset = is_07_subset
        self.test_file = "test.txt" if not self.is_07_subset else "val.txt"
        self.val_data_path = data_root

        img_inds_file = os.path.join(
            data_root, "ImageSets", "Main", self.test_file
        )
        with open(img_inds_file, "r") as f:
            lines = f.readlines()
            self.img_inds = [line.strip() for line in lines]

        self.progressbar = progressbar

        if class_names is None:
            self.classes = cfg.DATA["CLASSES"]
        else:
            self.classes = class_names

        self.predictions = {}
        self.ground_truth_boxes = {}

        self._parse_gt_boxes()
        self.metric_fn = MetricBuilder.build_evaluation_metric("map_2d",
            async_mode=True, num_classes=num_classes)

    def evaluate(self, multi_test=False, flip_test=False, iou_thresh=0.5):
        for img_ind in tqdm(self.img_inds, disable=not self.progressbar):
            img_path = os.path.join(self.val_data_path, "JPEGImages", img_ind + ".jpg")
            img = cv2.imread(img_path)
            self.process_image(
                img, img_ind=img_ind, multi_test=multi_test, flip_test=flip_test
            )
            self.metric_fn.add(np.array(self.predictions[img_ind]),
                np.array(self.ground_truth_boxes[img_ind]))

        metrics = self.metric_fn.value(iou_thresholds=iou_thresh)
        APs = {'mAP': metrics['mAP']}
        for cls_id, ap_dict in metrics[iou_thresh].items():
            APs[self.classes[cls_id]] = ap_dict['ap']
        return APs

    def process_image(self, img, img_ind, multi_test=False, flip_test=False):
        bboxes_prd = self.get_bbox(img, multi_test, flip_test)
        self.predictions[img_ind] = []
        for bbox in bboxes_prd:
            self.predictions[img_ind].append([*np.array(bbox[:4], dtype=np.int32),
                int(bbox[5]), bbox[4]])

    def _parse_gt_boxes(self):
        annopath = os.path.join(self.val_data_path, "Annotations", "{:s}.xml")
        for imagename in self.img_inds:
            parsed_boxes = self.parse_rec(annopath.format(imagename))
            for obj in parsed_boxes:
                if not obj['difficult']:
                    if imagename not in self.ground_truth_boxes:
                        self.ground_truth_boxes[imagename] = []
                    self.ground_truth_boxes[imagename].append([*obj["bbox"],
                        self.classes.index(obj["name"]), 0, 0])

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(float(bbox.find("xmin").text)),
                int(float(bbox.find("ymin").text)),
                int(float(bbox.find("xmax").text)),
                int(float(bbox.find("ymax").text)),
            ]
            objects.append(obj_struct)

        return objects


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='voc')
def yolo_eval_voc(
    model, data_root, num_classes=20, device="cuda", img_size=448,
    is_07_subset=False, progressbar=False, iou_thresh=0.5, conf_thresh=0.001,
    nms_thresh=0.5, **kwargs
):

    model.to(device)
    with torch.no_grad():
        ap_dict = VOCEvaluator(
            model, data_root, num_classes=num_classes, img_size=img_size,
            is_07_subset=is_07_subset, progressbar=progressbar, conf_thresh=conf_thresh,
            nms_thresh=nms_thresh).evaluate(iou_thresh=iou_thresh)

    return ap_dict


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='voc07')
def yolo_voc07_eval(
    model, data_root, num_classes=20, device="cuda",
    img_size=448, progressbar=True, conf_thresh=0.001,
    nms_thresh=0.5, iou_thresh=0.5, **kwargs
):
    return yolo_eval_voc(model, data_root, num_classes=num_classes, device=device,
        img_size=img_size, is_07_subset=True, progressbar=progressbar, conf_thresh=conf_thresh,
        nms_thresh=nms_thresh, iou_thresh=iou_thresh, **kwargs)
