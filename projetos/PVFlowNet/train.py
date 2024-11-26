import torch.optim as optim
import torch

def train_maf_with_loss_tracking(model, data, conditions, num_epochs=500, batch_size=64, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = data.shape[0] // batch_size

        indices = torch.randperm(data.shape[0])
        data = data[indices]
        conditions = conditions[indices]

        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            condition_batch = conditions[i * batch_size:(i + 1) * batch_size]

            loss = -model(batch, condition_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete!")
    return model, losses
