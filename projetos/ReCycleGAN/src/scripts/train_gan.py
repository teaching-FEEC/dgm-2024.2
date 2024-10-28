# pylint: disable=C0413,E0401
"""Trains and Tests CycleGAN"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.run import init_cyclegan_train, train_cyclegan
from src.utils.test_cases import TEST_CASES

def train(parameters):
    """Trains the CycleGAN model."""
    model, data_loaders = init_cyclegan_train(parameters)
    model = train_cyclegan(model, data_loaders, parameters)

if __name__ == '__main__':

    base_folder = Path(__file__).resolve().parent.parent.parent
    params = TEST_CASES["1"]

    train(params)
