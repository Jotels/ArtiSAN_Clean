import json
import logging
import os
import pickle
import sys
from typing import Optional, List, Tuple

import ase.formula
import numpy as np
import scipy.signal
import torch
from datetime import date
import ase.io

from torch import nn


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def combined_shape(length: int, shape: Optional[tuple] = None) -> tuple:
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: torch.nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_formulas(formulas: str) -> List[ase.formula.Formula]:
    return [ase.formula.Formula(s.strip()) for s in formulas.split(',')]


def get_tag() -> str:
    # today = date.today()
    # return 'The_ArtiSAN_{date}'.format(date=today)
    return 'ArtiSAN'


def save_config(config: dict, directory: str, tag: str, verbose=True):
    formatted = json.dumps(config, indent=4, sort_keys=True)

    if verbose:
        logging.info(formatted)

    path = os.path.join(directory, tag + '.json')
    with open(file=path, mode='w') as f:
        f.write(formatted)


def create_directories(directories: List[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_logger(config: dict, directory, tag: str):
    logger = logging.getLogger()

    logger.setLevel(config['log_level'])

    name = ''

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d ' + name + '%(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join(directory, tag + '.log')


def setup_simple_logger(path: str, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class ModelIO:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.root_name = tag
        self._suffix = '.model'
        self._iter_suffix = '.txt'

    def _get_model_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._suffix)

    def _get_info_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._iter_suffix)

    def save(self, module: nn.Module):
        # Save model
        model_path = self._get_model_path()
        print(f'Saving model: {model_path}')
        torch.save(obj=module, f=model_path)

    def load(self) -> nn.Module:
        # Load model
        model_path = self._get_model_path()
        logging.info(f'Loading model: {model_path}')
        model = torch.load(f=model_path)

        return model
