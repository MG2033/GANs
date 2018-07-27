import torchvision.transforms as v_transforms
import torchvision.datasets as v_datasets

from torch.utils.data import DataLoader

from data.base_data_loader import BaseDataLoader


class CelebADataLoader(BaseDataLoader):
    def __init__(self, config):
        self.config = config

        transform = v_transforms.Compose(
            [v_transforms.Resize((self.config.img_height, self.config.img_width)),
             v_transforms.ToTensor(),
             # Mean is set to 0.5 to get 2x-1 which makes the images between [-1, 1]
             v_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        dataset = v_datasets.ImageFolder(self.config.celebA_datapath, transform=transform)

        self.dataset_len = len(dataset)

        self.num_iterations = (self.dataset_len + self.config.batch_size - 1) // config.batch_size

        self.train_loader = DataLoader(dataset,
                                 batch_size=self.config.batch_size,
                                 shuffle=True,
                                 num_workers=self.config.data_loader_workers,
                                 pin_memory=self.config.pin_memory)
