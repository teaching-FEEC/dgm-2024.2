from model import Generator, Discriminator
from datasets import lungCTData
from transforms import AddGaussianNoise, AddUniformNoise
import torch
from save_models_and_training import SaveBestModel, SaveTrainingLosses


FACTORY_DICT = {
    "model_gen": {
        "Generator": Generator,
    },
    "model_disc": {
        "Discriminator": Discriminator,
    },
    "dataset": {
        "lungCTData": lungCTData,
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    },
    "criterion": {
        "BCELoss": torch.nn.BCELoss,
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
        "MSELoss": torch.nn.MSELoss,
    },
    "transforms": {
        "AddGaussianNoise": AddGaussianNoise,
        "AddUniformNoise": AddUniformNoise,
    },
    "savebest":{
        "SaveBestModel": SaveBestModel
    },
    "savelosses":{
        "SaveTrainingLosses": SaveTrainingLosses
    }
}
