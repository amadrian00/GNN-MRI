"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import torch
from datetime import datetime
import matplotlib.pyplot as plt

class AE(torch.nn.Module):
    def __init__(self, available_device, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.available_device = available_device

        dims = [self.in_channels[0], 128, 32]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=dims[0], out_features=dims[1]),
            torch.nn.BatchNorm1d(dims[1]),
            torch.nn.LeakyReLU(0.2),


            torch.nn.Linear(in_features=dims[1], out_features=dims[2]),
            torch.nn.BatchNorm1d(dims[2]),
            torch.nn.LeakyReLU(0.2),
        )

        # Linear encoder reducing the data to 64 dimensions
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=dims[2], out_features=dims[1]),
            torch.nn.BatchNorm1d(dims[1]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[1], out_features=dims[0]),
            torch.nn.BatchNorm1d(dims[0]),
            torch.nn.LeakyReLU(0.2),
        )


    """ Input:  x: Instance of data to process by the model.
        Output: Model result for the given input.

        Function that encodes and decodes the given input."""
    def forward(self, x):
        return self.decoder(self.encoder(x))

    """ Input:  x: Instance of data to process by the model.
        Output: Model code (intermediate result) for the given input.

        Function that encodes the given input."""
    def encode(self, x):
        return self.encoder(x)

    """ Input:  data_loader: Instance of data to train the model.
                batch_size: Integer indicating the batch size.
        Output: -
        Function that trains the model with the given data."""
    def train_loop(self, data_loader, batch_size):
        self.train()
        print(f"\nStarted training at {datetime.now().strftime("%H:%M:%S")}.")

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

        losses = []
        epochs = 20
        for epoch in range(epochs):
            i = 0
            for batch in data_loader:
                batch.to(self.available_device)
                elements_reconstructed = self(batch)

                # Calculating the loss function
                loss = loss_function(elements_reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"{'':<10} Epoch: {epoch} {'':<20} Batch Start Index: {i} {'':<20} Loss: {loss:.4f}")
                i += batch_size

                # Storing the losses in a list for plotting
                losses.append(loss.detach().item())

        print(f"Finished training at {datetime.now().strftime("%H:%M:%S")}.\n")
        self.eval()

        # Defining the Plot Style
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Plotting the last 100 values
        plt.plot(losses[-100:])