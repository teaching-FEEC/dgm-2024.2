import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def train_gail(generator, discriminator, env, expert_states, expert_actions, device, num_epochs=150, batch_size=256):
    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        idx = np.random.randint(0, expert_states.shape[0], batch_size)
        real_states = expert_states[idx]
        real_actions = expert_actions[idx]

        fake_states = []
        fake_actions = []
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32).to(device)

        for _ in range(batch_size):
            action = generator(state.unsqueeze(0)).detach()
            next_state, _, done, _ = env.step(action.cpu().numpy().squeeze(0))
            fake_states.append(state)
            fake_actions.append(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)
            if done:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(device)

        fake_states = torch.stack(fake_states)
        fake_actions = torch.cat(fake_actions)

        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        real_preds = discriminator(real_states, real_actions)
        fake_preds = discriminator(fake_states, fake_actions)

        loss_disc = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        action = generator(state.unsqueeze(0))
        disc_reward = torch.log(discriminator(state.unsqueeze(0), action) + 1e-8)
        loss_gen = -torch.mean(disc_reward)

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss Disc: {loss_disc.item()}, Loss Gen: {loss_gen.item()}")
