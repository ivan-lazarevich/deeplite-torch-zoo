import os
from os.path import expanduser
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)
from torchvision.datasets.vision import VisionDataset

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms, get_vanilla_transforms)
from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY

__all__ = ["get_imagenette", "get_imagenette_320", "get_imagenette_160"]

IMAGENETTE_IMAGENET_CLS_LABEL_MAP = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)


@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenette")
def get_imagenette(
    data_root="",
    batch_size=64,
    val_batch_size=None,
    img_size=224,
    num_workers=4,
    fp16=False,
    download=True,
    device="cuda",
    distributed=False,
    augmentation_mode='imagenet',
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    _URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenette.zip"

    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(
            f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"'
        )

    if augmentation_mode == 'imagenet':
        train_transforms, val_transforms = get_imagenet_transforms(img_size)
    else:
        train_transforms, val_transforms = get_vanilla_transforms(img_size)

    train_dataset = Imagenette(
        root=data_root,
        split='train',
        download=download,
        transform=train_transforms,
        url=_URL,
    )

    val_dataset = Imagenette(
        root=data_root,
        split='val',
        download=download,
        transform=val_transforms,
        url=_URL,
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=not distributed,
        device=device,
    )

    val_batch_size = batch_size if val_batch_size is None else val_batch_size
    val_loader = get_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=False,
        device=device,
    )

    return {"train": train_loader, "test": val_loader}


@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenette_320")
def get_imagenette_320(
    data_root="",
    batch_size=64,
    val_batch_size=None,
    img_size=224,
    num_workers=4,
    fp16=False,
    download=True,
    device="cuda",
    distributed=False,
    augmentation_mode='imagenet',
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    _URL = (
        "https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenette320.zip"
    )

    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(
            f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"'
        )

    if augmentation_mode == 'imagenet':
        train_transforms, val_transforms = get_imagenet_transforms(img_size)
    else:
        train_transforms, val_transforms = get_vanilla_transforms(img_size)

    train_dataset = Imagenette(
        root=data_root,
        split='train',
        download=download,
        transform=train_transforms,
        url=_URL,
    )

    val_dataset = Imagenette(
        root=data_root,
        split='val',
        download=download,
        transform=val_transforms,
        url=_URL,
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=not distributed,
        device=device,
    )

    val_batch_size = batch_size if val_batch_size is None else val_batch_size
    val_loader = get_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=False,
        device=device,
    )

    return {"train": train_loader, "test": val_loader}


@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenette_160")
def get_imagenette_160(
    data_root="",
    batch_size=64,
    val_batch_size=None,
    img_size=160,
    num_workers=4,
    fp16=False,
    download=True,
    device="cuda",
    distributed=False,
    augmentation_mode='imagenet',
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    _URL = (
        "https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenette160.zip"
    )

    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(
            f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"'
        )

    if augmentation_mode == 'imagenet':
        train_transforms, val_transforms = get_imagenet_transforms(img_size)
    else:
        train_transforms, val_transforms = get_vanilla_transforms(img_size)

    train_dataset = Imagenette(
        root=data_root,
        split='train',
        download=download,
        transform=train_transforms,
        url=_URL,
    )

    val_dataset = Imagenette(
        root=data_root,
        split='val',
        download=download,
        transform=val_transforms,
        url=_URL,
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=not distributed,
        device=device,
    )

    val_batch_size = batch_size if val_batch_size is None else val_batch_size
    val_loader = get_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=False,
        device=device,
    )

    return {"train": train_loader, "test": val_loader}


class Imagenette(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        download: bool = False,
        url: str = "",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val"))
        self.url = url
        if self.url[-7:] == "320.zip":
            self._base_folder = Path(self.root) / "imagenette320"
        elif self.url[-7:] == "160.zip":
            self._base_folder = Path(self.root) / "imagenette160"
        else:
            self._base_folder = Path(self.root) / "imagenette"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._labels = []
        self._image_files = []

        self.classes = sorted(
            entry.name
            for entry in os.scandir(self._base_folder / "train")
            if entry.is_dir()
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self._base_folder / f"{split}", target_class)
            for folder, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(folder, fname)
                    self._labels.append(class_index)
                    self._image_files.append(path)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(
            folder.exists() and folder.is_dir() for folder in (self._base_folder,)
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self.url, download_root=self.root)
