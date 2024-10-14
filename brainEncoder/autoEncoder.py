"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import torch
from gc import get_objects
import matplotlib.pyplot as plt
from collections import defaultdict

class AE(torch.nn.Module):
    def __init__(self, available_device, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.available_device = available_device

        self.conv1 = (3,1,1)
        self.conv2 = (3,1,1)

        self.pool1 = (1,2,2)
        self.pool2 = (1,2,2)

        self.encoded_shape = self._calculate_encoded_shape(self.in_channels, [self.conv1, self.conv2], [self.pool1, self.pool2])

        # squeeze?

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.in_channels[0], out_channels=64, kernel_size=self.conv1[0], stride=self.conv1[1], padding=self.conv1[2]),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(self.pool1),
            torch.nn.BatchNorm3d(64),

            torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=self.conv1[0], stride=self.conv1[1], padding=self.conv1[2]),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(self.pool2),
            torch.nn.BatchNorm3d(32),
        )

        # Linear encoder reducing the data to 64 dimensions
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=32,out_channels=64, kernel_size=self.conv1[0], stride=self.conv1[1], padding=self.conv1[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(64),
            torch.nn.Upsample(size=(self.encoded_shape[0] * self.pool1[0],  self.encoded_shape[1] * self.pool1[1],
                                    self.encoded_shape[2] * self.pool1[2]), mode='trilinear', align_corners=False),

            torch.nn.ConvTranspose3d(in_channels=64, out_channels=self.in_channels[0], kernel_size=self.conv1[0], stride=self.conv1[1], padding=self.conv1[2]),
            torch.nn.Upsample(size=(self.encoded_shape[0] * self.pool1[0]*self.pool2[0],
                                    self.encoded_shape[1] * self.pool1[1]*self.pool2[1],
                                    self.encoded_shape[2] * self.pool1[2]*self.pool2[2]),
                              mode='trilinear', align_corners=False),
            torch.nn.ReLU()
        )

        self.to(available_device)

    """ Input:  input_shape: Array indicating the input dimensions of the data.
                conv_layers: Tuple indicating the kernel dimensions of each convolution layer.
                pool_layers: Tuple indicating the kernel dimensions of each pool layer.
        Output: Array indicating the output dimensions of the data.

        Function that calculates the output dimensions of the model."""
    @staticmethod
    def _calculate_encoded_shape(input_shape, conv_layers, pool_layers):
        d, h, w = input_shape[1], input_shape[2], input_shape[3]

        for (kernel_size, stride, padding) in conv_layers:
            d = (d + 2 * padding - kernel_size) // stride + 1
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1

        for kernel_size in pool_layers:
            d //= kernel_size[0]
            h //= kernel_size[1]
            w //= kernel_size[2]

        return d, h, w

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

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

        before = defaultdict(int)
        for i in get_objects():
            before[type(i)] += 1

        losses = []
        epochs = 20
        for epoch in range(epochs):
            i = 0
            for batch in data_loader:
                batch.to(self.available_device )
                elements_reconstructed = self(batch)
                # Calculating the loss function
                loss = loss_function(elements_reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch} {'':<20} Batch Start Index: {i} {'':<20} Loss: {loss:.4f}")
                i += batch_size

                # Storing the losses in a list for plotting
                losses.append(loss.detach().item())

                after =  defaultdict(int)

                for j in get_objects():
                    after[type(j)] += 1

                reps = [(k, after[k] - before[k]) for k in after if after[k] - before[k]]
                print (reps)
                before = after

        self.eval()

        # Defining the Plot Style
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Plotting the last 100 values
        plt.plot(losses[-100:])