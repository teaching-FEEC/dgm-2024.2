import argparse
import torch
from environment.dress_env_gail import DressEnvGAIL
from models.generator_policy import GeneratorPolicy
from models.discriminator import Discriminator
from train.train_gail import train_gail
from train.test_gail import test_gail

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Modo: train ou test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DressEnvGAIL()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    generator = GeneratorPolicy(state_dim, action_dim).to(device)
    discriminator = Discriminator(state_dim, action_dim, 128).to(device)

    if args.mode == "train":
        expert_states = torch.tensor(np.load("expert_states.npy"), dtype=torch.float32).to(device)
        expert_actions = torch.tensor(np.load("expert_actions.npy"), dtype=torch.float32).to(device)
        train_gail(generator, discriminator, env, expert_states, expert_actions, device)
        torch.save(generator.state_dict(), "generator_model.pth")
    elif args.mode == "test":
        generator.load_state_dict(torch.load("generator_model.pth", map_location=device))
        test_gail(generator, env, device)

if __name__ == "__main__":
    main()
