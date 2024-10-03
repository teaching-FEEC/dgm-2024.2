from  models.gans.doppelganger.dgan_generator  import DCGANGenerator
from  models.gans.timeganpt.timegan_generator  import TimeGANGenerator
from  models.vae.vae_generator  import VRAEGenerator
from utils import log



class DataGenerate:
    def __init__(self, m_config):
        self.m_config = m_config
        if(self.m_config['name']=='vrae'):
            self.generator= VRAEGenerator(m_config)    

        elif(self.m_config['name']=='timeganpt'):
           self.generator= TimeGANGenerator(m_config)
         
        elif(self.m_config['name']=='Doppelgangerger'):
            self.generator=DCGANGenerator(m_config)
        self.n_gen_samples=m_config['n_gen_samples']
        self.folder_save=m_config['folder_save_generate_df']

       
    def train(self, X_train,y_train):
        try:
            log.print_debug(f"-----train----")  
            self.model=self.generator.train(X_train,y_train)
        except Exception as e:  
                log.print_err(f"Error in trainning synthetic data: {e}")  
           
            
    def generate(self):
        try:
            log.print_debug(f"-----generate ----")  
            self.synthetic_df=self.generator.generate(self.n_gen_samples)
            if(self.folder_save is not None):
                self.save_data(self.folder_save)
            return self.synthetic_df
        except Exception as e:  
                log.print_err(f"Error in generating synthetic data: {e}") 


    def save_data(self, folder, filename="synthetic_data.csv"):
        try:
            if self.synthetic_df is not None:
                file_path = f"{folder}/{filename}"
                self.synthetic_df.to_csv(file_path, index=False)
                log.info(f"Data saved to {file_path}")
            else:
                log.warning("No synthetic data to save.")
        except Exception as e:
            log.print_err(f"Error saving synthetic data: {e}")
          

           
           