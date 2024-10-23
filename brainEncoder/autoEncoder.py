"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import sys
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

class AE(torch.nn.Module):
    def __init__(self, available_device, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.available_device = available_device

        dims = [self.in_channels[0], 4056, 1024, 256, 64]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=dims[0], out_features=dims[1]),
            torch.nn.BatchNorm1d(dims[1]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[1], out_features=dims[2]),
            torch.nn.BatchNorm1d(dims[2]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[2], out_features=dims[3]),
            torch.nn.BatchNorm1d(dims[3]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[3], out_features=dims[4]),
            torch.nn.BatchNorm1d(dims[4]),
            torch.nn.LeakyReLU(0.2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=dims[4], out_features=dims[3]),
            torch.nn.BatchNorm1d(dims[3]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[3], out_features=dims[2]),
            torch.nn.BatchNorm1d(dims[2]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[2], out_features=dims[1]),
            torch.nn.BatchNorm1d(dims[1]),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(in_features=dims[1], out_features=dims[0]),
            torch.nn.BatchNorm1d(dims[0]),
            torch.nn.Tanh(),
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
    def fit(self, train_loader, val_dataloader, epochs, batch_size):
        print(f"\nStarted training at {datetime.now().strftime("%H:%M:%S")}.")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)

        train_losses = []
        val_losses = []
        best_val = sys.maxsize
        best_epoch = 0
        best_encoder_state_dict = None
        for epoch in range(epochs):
            self.train()
            i = 0

            batch_losses = []
            pq = tqdm(train_loader)
            pq.set_description(f"Epoch {epoch}")
            for batch in pq:
                batch_data, labels = batch

                batch_data.to(self.available_device)
                elements_reconstructed = self(batch_data)

                loss = criterion(elements_reconstructed, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += batch_size

                batch_losses.append(loss.detach().item()) # Storing the losses in a list for plotting
            train_losses.append(sum(batch_losses)/len(batch_losses))

            self.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_data, val_labels = val_batch

                    val_batch_data.to(self.available_device)
                    val_elements_reconstructed = self(val_batch_data)

                    val_loss = criterion(val_elements_reconstructed, val_batch_data)
            val_losses.append(val_loss.detach().item())

            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                best_encoder_state_dict = self.encoder.state_dict()
                best_epoch = epoch

        print(f'Finished training at {datetime.now().strftime("%H:%M:%S")}.\n'
              f'Best epoch {best_epoch}')

        torch.save(best_encoder_state_dict, 'brainEncoder/encoder.pt')
        #Plot
        epoch_list = np.arange(0, epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(epoch_list, val_losses, label='Validation Loss', color='orange', marker='x')

        plt.xticks(np.arange(0, epochs+1, max(1,epochs // 10)))

        # Label the axes and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()
        self.encoder.load_state_dict(best_encoder_state_dict)