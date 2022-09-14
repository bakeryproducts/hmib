import os.path as osp
from pathlib import Path

import cv2
from torch.utils.data import Dataset


class LungFilterDataset(Dataset):
    def __init__(self, items, transform=None, load_all_fields=False, rate=1):
        self.items = items
        self.transform = transform
        self.load_all_fields = load_all_fields
        self.rate = rate

    def __len__(self):
        return len(self.items) * self.rate

    def __getitem__(self, index):
        index %= len(self.items)

        item = self.items[index]
        sample = {
            "image": self.load_image(item["image_file"]),
            "mask": self.load_image(item["mask_file"], fmt="gray"),
        }

        if self.transform:
            sample = self.transform(**sample)

        sample["target"] = sample["mask"].float().mean()

        return sample

    @classmethod
    def create(cls, images_dir, cache=False, debug=False, **init_kwargs):
        items = cls.load_items(images_dir, cache, debug)
        return cls(items, **init_kwargs)

    @classmethod
    def load_items(cls, images_dir, cache=False, debug=False):
        items = []
        for image_file in Path(images_dir).glob("*_image.png"):
            image_name = osp.splitext(osp.basename(image_file))[0]
            assert image_name.endswith("_image")
            image_name = image_name[:-6]

            item = {
                "image_file": str(image_file),
                "mask_file": str(image_file).replace("_image", "_mask"),
                "image_name": image_name
            }

            if cache:
                item["image"] = cls.load_image(item["image_file"])
                item["mask"] = cls.load_image(item["mask_file"])

            items.append(item)

        return items

    @classmethod
    def load_image(cls, image_file: str, fmt: str = "rgb", image_size=None):
        if fmt == "gray":
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        elif fmt == "rgb":
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image format: {fmt}. Supported are: gray, rgb")

        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = cv2.resize(image, image_size)

        return image


if __name__ == "__main__":
    images_dir = "../../input/hmib/lung_filter"
    items = LungFilterDataset.load_items(images_dir, debug=True)
    dataset = LungFilterDataset(items)
    sample = dataset[12]
    print(1)
