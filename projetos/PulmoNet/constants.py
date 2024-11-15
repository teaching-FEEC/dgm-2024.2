from model import Generator, Discriminator
from datasets import lungCTData, processedCTData
from transforms import AddGaussianNoise, AddUniformNoise
import torch
from save_models_and_training import SaveBestModel, SaveTrainingLosses, SaveBestUnetModel, SaveUnetTrainingLosses
from losses import DiceLoss


FACTORY_DICT = {
    "model_gen": {
        "Generator": Generator,
    },
    "model_disc": {
        "Discriminator": Discriminator,
    },
    "model_unet": {
        "Unet": Generator,
    },
    "dataset": {
        "lungCTData": lungCTData,
        "processedCTData" : processedCTData
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    },
    "criterion": {
        "BCELoss": torch.nn.BCELoss,
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
        "MSELoss": torch.nn.MSELoss,
        "DiceLoss" : DiceLoss
    },
    "transforms": {
        "AddGaussianNoise": AddGaussianNoise,
        "AddUniformNoise": AddUniformNoise
    },
    "savebest":{
        "SaveBestModel": SaveBestModel,
        "SaveBestUnetModel" : SaveBestUnetModel
    },
    "savelosses":{
        "SaveTrainingLosses": SaveTrainingLosses, 
        "SaveUnetTrainingLosses" : SaveUnetTrainingLosses
    }
}
