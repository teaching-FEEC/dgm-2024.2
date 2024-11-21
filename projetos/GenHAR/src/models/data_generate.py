from models.gans.doppelganger.dgan_generator import DCGANGenerator
from models.gans.timeganpt.timegan_generator import TimeGANGenerator
from models.vae.vae_generator import VRAEGenerator
from models.diffusion.diffusion import DiffusionGenerator
from utils import log
import os


class DataGenerate:
    def __init__(self, m_config, dataset, transformation):
        self.m_config = m_config
        self.losses = {}


        if self.m_config["name"] == "timeganpt":
            self.generator = TimeGANGenerator(m_config)

        elif self.m_config["name"] == "Doppelgangerger":
            self.generator = DCGANGenerator(m_config)

        elif self.m_config["name"] in ["cond_diffusion_unet1d", "uncond_diffusion_unet1d"]:
            self.generator = DiffusionGenerator(m_config)

        self.n_gen_samples = m_config["n_gen_samples"]

    def train(self, X_train, y_train):
        #try:
            X_train_ = X_train.copy()
            log.print_debug(f"-----train----{self.m_config['name']}")
            self.model, loss_hist = self.generator.train(X_train_, y_train)
            self.losses = loss_hist
        #except Exception as e:
        #    log.print_err(f"Error in trainning synthetic data: {e}")

    def generate(self):
        #try:
            log.print_debug(f"-----generate ----{self.m_config['name']}")
            self.synthetic_df = self.generator.generate(self.n_gen_samples)
#            if self.folder_save is not None:
#                self.save_data(self.folder_save)
            return self.synthetic_df
        #except Exception as e:
        #    log.print_err(f"Error in generating synthetic data: {e}")


