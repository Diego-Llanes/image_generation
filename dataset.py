from pathlib import Path
import pickle
import os
from typing import Tuple, Union, List, Dict, Callable
from functools import cached_property

import torch
from torch.utils.data import Dataset


class DebugDataset(Dataset):

    def __init__(self, num_samples: int = 10_000, num_classes: int = 10) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.rand(1, 28, 28), torch.randint(0, self.num_classes, (1,))

    @cached_property
    def get_in_out_size(self):
        return (1, 28, 28), self.num_classes


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        normalize: bool = False,
    ) -> None:
        self.csv_path = os.path.join(
            data_dir,
            "fashion-mnist_train.csv" if split == "train" else "fashion-mnist_test.csv",
        )

        self.data = torch.tensor(
            [
                [float(x) for x in line.split(",")]
                for line in open(self.csv_path).readlines()[1:]
            ]
        )
        self.labels = self.data[:, 0].long()
        if normalize:
            self.data /= 255.0

        self.num_classes = 10

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # return image, label
        return self.data[idx][1:].reshape(1, 28, 28), self.labels[idx]

    @cached_property
    def get_in_out_size(self):
        return self[0][0].shape, self.num_classes


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        normalizer: Callable = None,
        augs: List[Callable] = None,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.normalizer = normalizer
        self.data, self.labels = self._load_data()
        self.augs = augs
        self.num_classes = 10

    def unpickle(self, file) -> Dict:
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    def _load_data(self):
        if self.split == "train":
            data = []
            labels = []
            for i in range(1, 6):
                batch = self.unpickle(os.path.join(self.data_dir, f"data_batch_{i}"))
                data.append(batch[b"data"])
                labels.append(batch[b"labels"])
            data = [torch.tensor(d) for d in data]
            data = torch.cat(data)
            labels = [torch.tensor(l) for l in labels]
            labels = torch.cat(labels)
        else:
            batch = self.unpickle(os.path.join(self.data_dir, "test_batch"))
            data = batch[b"data"]
            labels = batch[b"labels"]

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return data, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.data[idx].reshape(3, 32, 32).float()
        if self.normalizer:
            img = self.normalizer(img)
        if self.augs:
            for aug in self.augs:
                img = aug(img)
        # return image, label
        return img, self.labels[idx]

    def get_in_out_size(self):
        return self[0][0].shape, self.num_classes


class MixedDataset(Dataset):
    def __init__(
        self,
        datasets: List[Dataset],
        transforms: List[Callable] = None,
        split: str = "train",
    ) -> None:
        self.datasets = datasets
        self.transforms = transforms
        self.num_classes = sum([dataset.num_classes for dataset in datasets])

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # find the dataset that idx belongs to
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx]):
            idx -= len(self.datasets[dataset_idx])
            dataset_idx += 1

        img, label = self.datasets[dataset_idx][idx]
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)

        # update label to be the global label
        label += sum(dataset.num_classes for dataset in self.datasets[:dataset_idx])
        img = img.reshape(self.get_in_out_size[0])
        return img, label

    @cached_property
    def get_in_out_size(self):
        return self.datasets[0][0][0].shape, self.num_classes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_FashionMNISTDataset():
        dataset = FashionMNISTDataset(
            data_dir=Path("data/fashion_mnist/"), split="train", normalize=True
        )
        print(len(dataset))
        print(dataset[0])
        plt.imshow(dataset[0][0].reshape(28, 28), cmap="gray")
        plt.show()

    def test_CIFAR10Dataset():
        dataset = CIFAR10Dataset(
            data_dir=Path("data/cifar-10-batches-py/"),
            split="train",
            normalizer=lambda x: (x - x.min()) / (x.max() - x.min()),
        )
        print(len(dataset))
        print(dataset[0])
        plt.imshow(dataset[0][0].permute(1, 2, 0))
        plt.show()

    # Test MixedDataset
    def test_MixedDataset(count=10):
        dataset1 = FashionMNISTDataset(
            data_dir=Path("data/fashion_mnist/"), split="train", normalize=True
        )
        dataset2 = CIFAR10Dataset(
            data_dir=Path("data/cifar-10-batches-py/"),
            split="train",
            normalizer=lambda x: (x - x.min()) / (x.max() - x.min()),
            augs=[
                # interpolate to 28x28
                lambda x: torch.nn.functional.interpolate(
                    x.unsqueeze(0), size=(28, 28), mode="bilinear", align_corners=False
                ).squeeze(0),
                # greyscale
                lambda x: x.mean(dim=0),
            ],
        )
        dataset = MixedDataset(
            datasets=[dataset1, dataset2],
            transforms=[],
        )
        for i in range(count):
            item = dataset[i]
            print(item)
            plt.imshow(item[0].squeeze().numpy(), cmap="gray")
            # show the label
            plt.title(item[1])
            plt.show()

    # test_FashionMNISTDataset()
    # test_CIFAR10Dataset()
    test_MixedDataset(count=10)
