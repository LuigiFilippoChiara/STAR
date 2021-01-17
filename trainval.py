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
    parser.add_argument('--train_model', default='star', help='Your model name')
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


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.full_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    arg_dict['using_cuda'] = arg_dict['using_cuda'] and torch.cuda.is_available()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)
    args = load_arg(p)

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    elif args.phase == 'train':
        trainer.train()
    else:
        raise ValueError("Unsupported phase! args.phase should be train or "
                         "test")
    print("Program finished.")
