import torch

def test_gail(generator, env, device, num_episodes=5):
    for ep in range(num_episodes):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False

        while not done:
            with torch.no_grad():
                action = generator(state.unsqueeze(0))
            next_state, _, done, _ = env.step(action.cpu().numpy().squeeze(0))
            state = torch.tensor(next_state, dtype=torch.float32).to(device)
