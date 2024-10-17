"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import torch
import numpy as np
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

    """ Input:  train_loader: Instance of data to train the model.
                val_dataloader: Instance of data to validate the model.
                epochs: Integer indicating the number of epochs to train the model.
                batch_size: Integer indicating the batch size.
        Output: -
        Function that trains the model with the given data."""
    def train_loop(self, train_loader, val_dataloader, epochs, batch_size):
        print(f"\nStarted training at {datetime.now().strftime("%H:%M:%S")}.")

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.train()
            i = 0
            batch_losses = []
            for batch in train_loader:
                batch_data, _ = batch
                batch_data.to(self.available_device)
                elements_reconstructed = self(batch_data)

                loss = loss_function(elements_reconstructed, batch_data) # Calculating the loss function

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"{'':<10} Epoch: {epoch} {'':<20} Batch Start Index: {i} {'':<20} Loss: {loss:.4f}")
                i += batch_size

                batch_losses.append(loss.detach().item()) # Storing the losses in a list for plotting
            train_losses.append(sum(batch_losses)/len(batch_losses))

            self.eval()
            val_batch_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_data, _ = val_batch
                    val_batch_data.to(self.available_device)
                    val_elements_reconstructed = self(val_batch_data)

                    val_loss = loss_function(val_elements_reconstructed, val_batch_data)
                    val_batch_losses.append(val_loss.detach().item())  # Storing the losses in a list for plotting
                    print(f"{'':<40}Validation loss for epoch: {epoch} {'':<20} {val_loss:.4f}")
            val_losses.append(sum(val_batch_losses)/len(val_batch_losses))

        print(f"Finished training at {datetime.now().strftime("%H:%M:%S")}.\n")

        #Plot
        epoch_list = np.arange(0, epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(epoch_list, val_losses, label='Validation Loss', color='orange', marker='x')

        plt.xticks(np.arange(0, epochs, max(1,epochs // 10)))
        plt.ylim(min(train_losses)-0.025, max(train_losses)+0.025)

        # Label the axes and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()