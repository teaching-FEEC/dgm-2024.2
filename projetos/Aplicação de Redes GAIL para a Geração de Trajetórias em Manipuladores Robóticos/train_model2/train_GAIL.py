import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import entropy
from environment import DressEnvGAIL
from models import GeneratorPolicy, Discriminator

def train_model(env, generator, discriminator, device):
    expert_states = torch.tensor(np.load("data/expert_states.npy"), dtype=torch.float32).to(device)
    expert_actions = torch.tensor(np.load("data/expert_actions.npy"), dtype=torch.float32).to(device)

    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

    batch_size = 128
    num_epochs = 150

    # --- Treinamento Inicial do Discriminador ---
    initial_disc_epochs = 15
    for epoch in range(initial_disc_epochs):
        idx = np.random.randint(0, expert_states.shape[0], batch_size)
        real_states = expert_states[idx]
        real_actions = expert_actions[idx]

        fake_states = []
        fake_actions = []

        for _ in range(batch_size):
            action = generator(real_states).detach()
            next_state, _, done, _ = env.step(action.cpu().numpy().squeeze(0))
            fake_states.append(real_states)
            fake_actions.append(action)
            real_states = torch.tensor(next_state, dtype=torch.float32).to(device)
            if done:
                real_states, _ = env.reset()
                real_states = torch.tensor(real_states, dtype=torch.float32).to(device)

        fake_states = torch.stack(fake_states)
        fake_actions = torch.cat(fake_actions)

        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        real_preds = discriminator(real_states, real_actions)
        fake_preds = discriminator(fake_states, fake_actions)

        loss_disc_real = criterion(real_preds, real_labels)
        loss_disc_fake = criterion(fake_preds, fake_labels)
        loss_disc = loss_disc_real + loss_disc_fake

        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        print(f"[Pretrain Disc Epoch {epoch + 1}/{initial_disc_epochs}] Loss D: {loss_disc.item()}")

    # --- Treinamento Principal ---
    losses_disc = []
    losses_gen = []
    js_divergences = []

    for epoch in range(num_epochs):
        # Gerar um batch de dados de treino
        idx = np.random.randint(0, expert_states.shape[0], batch_size)
        real_states = expert_states[idx]
        real_actions = expert_actions[idx]

        # Gerar ações falsas a partir do gerador
        fake_states = []
        fake_actions = []

        for _ in range(batch_size):
            action = generator(real_states).detach()
            fake_states.append(real_states)
            fake_actions.append(action)
            real_states = torch.tensor(next_state, dtype=torch.float32).to(device)

        fake_states = torch.stack(fake_states)
        fake_actions = torch.cat(fake_actions)

        # Treinando o discriminador
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        real_preds = discriminator(real_states, real_actions)
        fake_preds = discriminator(fake_states, fake_actions)

        loss_disc_real = criterion(real_preds, real_labels)
        loss_disc_fake = criterion(fake_preds, fake_labels)
        loss_disc = loss_disc_real + loss_disc_fake

        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # Treinando o gerador
        fake_preds = discriminator(fake_states, fake_actions)
        loss_gen = criterion(fake_preds, real_labels)

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        losses_disc.append(loss_disc.item())
        losses_gen.append(loss_gen.item())
        js_divergences.append(js_divergence(real_preds.detach().cpu().numpy(), fake_preds.detach().cpu().numpy()))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss Disc: {loss_disc.item():.4f} | Loss Gen: {loss_gen.item():.4f} | JSD: {js_divergences[-1]:.4f}")

    # Salvando os modelos
    torch.save(generator.state_dict(), "generator_model.pth")
    torch.save(discriminator.state_dict(), "discriminator_model.pth")

  