import pickle
from  models.vae.vrae import VRAE
class VRAEGenerator:
    def __init__(self, m_config):
        self.m_config = m_config
        self.model = None  # O modelo VRAE será armazenado aqui

    def train(self, X_train, y_train):
        # Definindo parâmetros do VRAE
        parameters_ = self.m_config["parameters"]
        sequence_length = parameters_["sequence_length"]
        number_of_features = parameters_["number_of_features"]
        n_epochs = parameters_["n_epochs"]
        
        # Treinando o modelo VRAE
        self.model = VRAE(sequence_length=sequence_length, number_of_features=number_of_features, n_epochs=n_epochs)
        self.model.fit(X_train, y_train, save=True)

    def generate(self, num_samples_per_class):
        # Gerar amostras sintéticas a partir do VRAE
        if self.model is None:
            raise Exception("O modelo VRAE não foi treinado. Por favor, treine o modelo antes de gerar amostras.")

        # Gerar dados sintéticos
        synthetic_data = self.model.generate(num_samples=num_samples_per_class)
        return synthetic_data

    def save_model(self, file_path):
        # Salvar o modelo treinado como um arquivo .pkl
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path):
        # Carregar o modelo a partir de um arquivo .pkl
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)