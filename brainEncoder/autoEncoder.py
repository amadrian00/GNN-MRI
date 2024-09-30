"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import torch
import matplotlib.pyplot as plt


class AE(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

        # Linear encoder reducing the data to 64 dimensions
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dims, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Linear encoder reducing the data to 64 dimensions
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dims),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def training(self, x):
        self.train()

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

        outputs = []
        losses = []

        epochs = 20
        for epoch in range(epochs):
            for (image, _) in x:
                # Reshaping the image to (-1, 784)
                image = image.reshape(-1, self.dims)

                # Output of Autoencoder
                reconstructed = self(image)

                # Calculating the loss function
                loss = loss_function(reconstructed, image)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss)
                outputs.append((epochs, image, reconstructed))

        # Defining the Plot Style
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Plotting the last 100 values
        plt.plot(losses[-100:])

        self.eval()