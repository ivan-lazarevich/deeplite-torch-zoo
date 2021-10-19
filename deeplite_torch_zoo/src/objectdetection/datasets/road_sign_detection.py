import os
import random
from math import floor
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np
import torch

from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import (
    Mixup,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFilp,
    Resize,
)


class RoadSignDetectionDataset:
    def __init__(self, root, split='train', num_classes=None, img_size=416):
        """Road sign detection dataset in YOLO format"""
        self.root = Path(root)
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        self.split = split
        self.labels_set = set()

        self._parse_annotation_files()
        print(f"Number of samples in the '{split}' dataset split: {len(self.img_info)}")

        self.num_classes = len(self.labels_set)
        if num_classes is not None:
            self.num_classes = num_classes
        print(f'Number of classes in the dataset: {self.num_classes}')

        self._img_size = img_size

    def __getitem__(self, item):
        """
        return:
            bboxes of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """
        img_org, bboxes_org, img_id = self.__parse_annotation(self.img_info[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.img_info) - 1)
        img_mix, bboxes_mix, _ = self.__parse_annotation(self.img_info[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()
        return img, bboxes, bboxes.shape[0], img_id

    def __parse_annotation(self, _info):
        img_path = _info["img_path"].as_posix().replace('\n', '')
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, "File Not Found " + img_path

        bboxes = np.array(_info["annotations"]["bbox"])

        bboxes = [
            (
                floor(x1 * img.shape[1]),
                floor(y1 * img.shape[0]),
                floor(x2 * img.shape[1]),
                floor(y2 * img.shape[0]),
            )
            for (x1, y1, x2, y2) in bboxes
        ]
        labels = np.array(_info["annotations"]["labels"])[:, np.newaxis]
        bboxes = np.concatenate((bboxes, labels), axis=1)

        if len(bboxes) == 0:
            bboxes = np.array(np.zeros((0, 5)))
        else:
            img, bboxes = RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
            img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
            img, bboxes = RandomAffine()(np.copy(img), np.copy(bboxes))

        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes, str(Path(img_path).stem)

    def collate_img_label_fn(self, sample):
        images = []
        labels = []
        lengths = []
        labels_with_tail = []
        max_num_obj = 0

        for image, label, length, img_id in sample:
            images.append(image)
            labels.append(label)
            lengths.append(length)
            max_num_obj = max(max_num_obj, length)
        for label in labels:
            num_obj = label.size(0)
            zero_tail = torch.zeros(
                (max_num_obj - num_obj, label.size(1)),
                dtype=label.dtype,
                device=label.device,
            )
            label_with_tail = torch.cat((label, zero_tail), dim=0)
            labels_with_tail.append(label_with_tail)
        image_tensor = torch.stack(images)
        label_tensor = torch.stack(labels_with_tail)
        length_tensor = torch.tensor(lengths)
        return image_tensor, label_tensor, length_tensor, None

    def __len__(self) -> int:
        return len(self.img_info)

    def _parse_annotation_files(self) -> None:
        filename = self.split + '.txt'
        filepath = os.path.join(self.root, filename)

        with open(filepath, "r") as f:
            lines = f.readlines()
            for aline in lines:
                labels = []
                img_path = Path(os.path.join(self.root, os.path.basename(aline)))
                anno_path = img_path.with_suffix('.txt')
                with open(anno_path, "r") as anno_file:
                    annotation = anno_file.readlines()
                    num_boxes = len(annotation)
                    if num_boxes == 0:
                        continue
                    labels, bboxes = [], []
                    for line in annotation:
                        label, *bbox_values = [float(v) for v in line.split()]
                        x1, y1, dx, dy = bbox_values
                        bbox_values = (
                            x1 - 0.5 * dx,
                            y1 - 0.5 * dy,
                            x1 + 0.5 * dx,
                            y1 + 0.5 * dy,
                        )
                        labels.append(label)
                        self.labels_set.add(label)
                        bboxes.append(bbox_values)
                    self.img_info.append(
                        {
                            "img_path": img_path,
                            "annotations": {
                                "bbox": bboxes,
                                "labels": labels,
                            },
                        }
                    )
