"""Module docstring.

continue
"""
import logging
import argparse
from pathlib import Path
import datetime
import sys
import json

import numpy as np
import torch


def parse_args():
    """Parse arguments.

    continue
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('metadata_path', type=Path, help='Path to.')
    parser.add_argument('--no_cuda', action='store_true',
                        help="Set if you DON'T want to use cuda.")
    parser.add_argument('--out_dir', type=Path, default='./',
                        help='Output directiory.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Choose verbosity levels, based on how often the argument is given.')
    parser.add_argument('--log_file', action='store_true', help='Create log file.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Randomgenerator seed for reproducibility.')
    return parser.parse_args()


def set_seed(seed):
    """Set seed for reproducebility.

    PyTorch and Numpy seed.
    """
    print(f'Setting seed to {seed}.')
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def config_logger(verbosity, log_file, log_dir):
    """Configure logging.

    Set log level and log file path.
    """

    log_level = 10*(3-max(0, min(verbosity, 2)))
    print(f'Log level set to: {logging.getLevelName(log_level)}')

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    if log_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
        log_path = log_dir.joinpath(f'{timestamp}-{Path(sys.argv[0]).stem}.log')
        print(f'Saving logs to: {log_path.absolute()}')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


def main(args):
    """
    """
    if args.seed is not None:
        set_seed(args.seed)
    start_datetime = datetime.datetime.now()
    print(f'Arguments passed to the script: {args}')
    config_logger(verbosity=args.verbose, log_file=args.log_file, log_dir=args.out_dir)
    logging.info('Arguments passed to the script: %s', args)

    if torch.cuda.is_available() and not args.no_cuda:
        print('Using CUDA.')
        logging.info('Using CUDA.')

    with open(args.metadata_path, 'r') as fp:
        metadata = json.load(fp)

    print(metadata.keys())
    sets = [set(metadata['ion_channel']['drug']['index']) , set(metadata['gpcr']['drug']['index']) , set(metadata['enzyme']['drug']['index']), set(metadata['nuclear_receptor']['drug']['index'])]
    for s in sets:
        print(len(s))

    print(set.intersection(*sets))
    # print(metadata['ego_networks'].keys())

    print(f'Script run time: {datetime.datetime.now() - start_datetime}')


if __name__ == '__main__':
    main(parse_args())
