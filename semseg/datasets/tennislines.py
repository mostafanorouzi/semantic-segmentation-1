import glob
import os
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class TennisLines(Dataset):
    """
    num_classes: 2
    all_num_classes: 2
    """
    CLASSES = ["bg", "line"]
    CLASSES_ALL = ["bg", "line"]
    PALETTE = torch.tensor([[0, 0, 0], [128, 128, 128]])
    PALETTE_ALL = torch.tensor([[0, 0, 0], [128, 128, 128]])

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        random.seed(0)
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = os.path.join(root, "images")
        self.files = glob.glob(os.path.join(img_path, "*" + ".jpg"))
        random.shuffle(self.files)
        if split == "train":
            self.files = self.files[:int(len(self.files) * .7)]
        elif split == "test":
            self.files = self.files[int(len(self.files) * .7):int(len(self.files) * .9)]
        else:
            self.files = self.files[int(len(self.files) * .9):]

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', "masks")

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label).long()

    def encode(self, label: Tensor) -> Tensor:
        label = label.permute(1, 2, 0).squeeze()
        label = label / 255
        mask = torch.zeros(label.shape[:-1])

        # for index, color in enumerate(self.PALETTE):
        #     bool_mask = torch.eq(label, color)
        #     class_map = torch.all(bool_mask, dim=-1)
        #     mask[class_map] = index + 1
        return label


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample

    visualize_dataset_sample(TennisLines, '/home/mostafa/mehdi_data')
