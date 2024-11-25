import argparse
from train import train_model
from test import test_gail
import torch
from environment import DressEnvGAIL
from models import GeneratorPolicy, Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute o treinamento ou teste do modelo GAIL.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Modo de execução: train ou test")
    parser.add_argument("--generator_model", type=str, help="Caminho para o modelo gerador salvo (para o modo test)")
    parser.add_argument("--discriminator_model", type=str, help="Caminho para o modelo discriminador salvo (opcional no modo test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DressEnvGAIL()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    generator = GeneratorPolicy(state_dim, action_dim).to(device)
    discriminator = Discriminator(state_dim, action_dim, 128).to(device)

    if args.mode == "train":
        train_model(env, generator, discriminator, device)
        torch.save(generator.state_dict(), "generator_model.pth")
    elif args.mode == "test":
        generator.load_state_dict(torch.load(args.generator_model, map_location=device))
        test_gail(generator, env, device)
