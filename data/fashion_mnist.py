import torchvision
import torchvision.transforms as v_transforms

from torch.utils.data import DataLoader, ConcatDataset

from data.base_data_loader import BaseDataLoader


# TODO

class FashionMNISTDataLoader(BaseDataLoader):
    def __init__(self, config):
        self.config = config

        transform = v_transforms.Compose(
            [v_transforms.Resize((self.config.img_height, self.config.img_width)),
             v_transforms.ToTensor(),
             v_transforms.Normalize(mean=(0.5,), std=(0.5,))])

        train_dataset = torchvision.datasets.FashionMNIST(root=self.config.data_folder, train=True,
                                                          download=True,
                                                          transform=transform)

        test_dataset = torchvision.datasets.FashionMNIST(root=self.config.data_folder, train=False,
                                                         download=True,
                                                         transform=transform)

        dataset = ConcatDataset([train_dataset, test_dataset])

        self.dataset_len = len(dataset)

        self.num_iterations = (self.dataset_len + config.batch_size - 1) // config.batch_size

        self.train_loader = DataLoader(dataset,
                                 batch_size=self.config.batch_size,
                                 shuffle=True,
                                 num_workers=self.config.data_loader_workers,
                                 pin_memory=self.config.pin_memory)
