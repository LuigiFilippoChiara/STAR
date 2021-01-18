import argparse
import ast
import os

import torch
import yaml

from src.processor import processor

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(description='STAR')
    parser.add_argument('--dataset', default='eth5')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2,'
                             ' univ] for ETH-univ, ETH-hotel, UCY-zara01, '
                             'UCY-zara02, UCY-univ')
    parser.add_argument('--base_dir', default='.',
                        help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/',
                        help='Directory for saving caches and models.')
    # default='train'
    parser.add_argument('--phase', default='train',
                        help='Set this value to "train" or "test"')
    parser.add_argument('--train_model', default='star', help='Model name')
    # TODO: Check if I can load pre-trained model and continue learning
    parser.add_argument('--load_model', default=None, type=int,
                        help="Load pre-trained model for test or training. "
                             "Specify the epoch to load. Defalut=None means "
                             "do not load.")
    parser.add_argument('--model', default='star.STAR', help="Which model")
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int,
                        help="=Desired number of pedestrians in a batch - at least")
    # TODO: What does this?
    parser.add_argument('--batch_size', default=8, type=int, help="???")
    # TODO: This is not used at all.
    parser.add_argument('--test_batch_size', default=4, type=int)
    # TODO: change! default=300
    parser.add_argument('--num_epochs', default=3, type=int)
    # TODO: change! default=10
    parser.add_argument('--start_test', default=1, type=int)
    parser.add_argument('--show_step', default=100, type=int,
                        help="Intermediate results during training are shown "
                             "after show_step batches")
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval,
                        help="True: show intermediate epoch results during "
                             "training")
    parser.add_argument('--ifsave_results', default=False,
                        type=ast.literal_eval, help="True: save best model")
    # TODO: experiment with random rotation, True/False in Train/Test sets
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="True: random rotation of each trajectory fragment")
    # TODO: change and test threshold distance to be considered neighbors
    parser.add_argument('--neighbor_thred', default=10, type=int,
                        help="Threshold distance to be considered neighbors (meters)")
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int, help="Gradient clip")

    return parser


def load_arg(p_arg):
    """
    Load args from config file and confront them with parsed args.
    p_arg are the entered parsed arguments for this run, while saved_args
    are the previously saved config arguments.

    The priority is:
    command line > configuration files > default values in script.
    """
    with open(p_arg.config, 'r') as f:
        saved_args = yaml.full_load(f)
    for k in saved_args.keys():
        if k not in vars(p_arg).keys():
            raise KeyError('WRONG ARG: {}'.format(k))
    assert set(saved_args) == set(vars(p_arg)), \
        "Entered args and config saved args are different"
    parser.set_defaults(**saved_args)
    return parser.parse_args()


def save_arg(arg):
    """
    Save args to config file
    """
    arg_dict = vars(arg)
    if not os.path.exists(arg.model_dir):
        os.makedirs(arg.model_dir)
    with open(arg.config, 'w') as f:
        yaml.dump(arg_dict, f)


def add_default_paths_and_device(arg):
    """
    Add default paths and device to parsed args
    """
    arg.save_dir = os.path.join(
        arg.save_base_dir, str(arg.test_set))
    arg.model_dir = os.path.join(
        arg.save_dir, arg.train_model)
    arg.config = os.path.join(
        arg.model_dir, 'config_' + arg.phase + '.yaml')
    arg.using_cuda = arg.using_cuda and torch.cuda.is_available()
    return arg


if __name__ == '__main__':
    parser = get_parser()
    pars_args = parser.parse_args()
    pars_args = add_default_paths_and_device(pars_args)

    # configuration files are created at the first run
    if not os.path.exists(pars_args.config):
        save_arg(pars_args)
    args = load_arg(pars_args)

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    elif args.phase == 'train':
        trainer.train()
    else:
        raise ValueError(
            "Unsupported phase! args.phase need to be train or test")
    print("Program finished.")
