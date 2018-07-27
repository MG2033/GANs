from easydict import EasyDict as edict
import argparse
import json
import sys
import os
from pprint import pprint

import torch.nn as nn


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="GANs PyTorch Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    output_dir = experiment_dir + 'out/'

    dirs = [summary_dir, checkpoint_dir, output_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir, output_dir
        return experiment_dir, summary_dir, checkpoint_dir, output_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
