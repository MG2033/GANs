import sys
sys.path.extend(['.'])

from data.celebA import CelebADataLoader
from DCGAN.model import Generator
from DCGAN.model import Discriminator
from DCGAN.train import DCGANTrainer
from utils import parse_args, create_experiment_dirs

import torch


def main():
    # Parse the JSON arguments
    config_args = parse_args()

    # Create the experiment directories
    experiment_dir, summary_dir, checkpoint_dir, output_dir = create_experiment_dirs(
        config_args.experiment_dir)

    generator = Generator(config_args.Z_dim, config_args.dim_multiplier, config_args.img_channels)
    discriminator = Discriminator(config_args.dim_multiplier, config_args.img_channels, config_args.leaky)

    data_loader = CelebADataLoader(config_args)

    device = torch.device(config_args.device)

    trainer = DCGANTrainer(generator, discriminator, data_loader, device, summary_dir, checkpoint_dir, output_dir,
                           config_args)

    if config_args.mode == "train":
        trainer.train()
    else:
        raise ValueError("Choose from the following modes: (train)")


if __name__ == "__main__":
    main()
