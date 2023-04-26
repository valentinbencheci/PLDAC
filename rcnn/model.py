from data import load_data, load_merged_data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_auc_score


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First
            nn.Conv2d(1, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(kernel_size=(5, 1)),
            # Second
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Third
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Fourth
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Gated Recurrent Unit (GRU) layers
        self.gru_layers = nn.GRU(96, 96, num_layers=2, batch_first=True)

        # Temporal max-pooling layer
        self.temporal_max_pooling = nn.AdaptiveMaxPool1d(1)

        # Feedforward layer
        self.feedforward_layer = nn.Sequential(nn.Linear(96, 1), nn.Sigmoid())

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)
        x = self.conv_layers(x)

        # Rearrange dimensions for GRU input
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # GRU layers
        x, _ = self.gru_layers(x)

        # Temporal max-pooling
        x = x.permute(0, 2, 1)
        x = self.temporal_max_pooling(x)
        x = x.squeeze()

        # Feedforward layer
        x = self.feedforward_layer(x)

        return x


def train(batch_size):
    train_dataset, train_loader, val_dataset, val_loader = load_merged_data(batch_size)

    # Create an instance of the model
    model = CRNN()

    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # define number of epochs and early stopping
    num_epochs = 100
    patience = 50
    best_auc = 0.0
    counter = 0

    # training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("Training...")
        # set model to train mode
        model.train()

        # iterate over batches in the training set
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # reshape labels
            labels = labels.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate loss and backpropagate
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
        print("Training done!")

        # evaluate model on validation set
        print("Evaluating...")
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # reshape labels
                labels = labels.unsqueeze(1)

                # forward pass
                outputs = model(inputs)

                # calculate loss
                loss = criterion(outputs.float(), labels.float())
                val_loss += loss.item() * inputs.size(0)

                # save predictions and labels
                val_preds += outputs.cpu().numpy().tolist()
                val_labels += labels.cpu().numpy().tolist()

            # calculate validation metrics
            val_loss /= len(val_dataset)
            val_auc = roc_auc_score(val_labels, val_preds)

            # print validation metrics
            print(f"=> Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            print("Evaluating done!")

            # check for early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping after {patience} epochs without improvement."
                    )
                    break


if __name__ == "__main__":
    train(batch_size=16)


def debug_shapes(model, input_shape):
    # Define hook function to print shape of input and output tensors
    def print_shape(module, input, output):
        print(f"Module: {module.__class__.__name__}")
        print(f"Shape of input tensor: {input[0].shape}")
        for i, o in enumerate(output):
            print(f"Shape of output tensor {i}: {o.shape}")
        print("=" * 50)

    # Add hook to each module in the model
    for module in model.modules():
        module.register_forward_hook(print_shape)

    # Pass input tensor through the model
    input_tensor = torch.randn(input_shape)
    output_tensor = model(input_tensor)

    # Remove hooks from model
    for module in model.modules():
        module._forward_hooks.clear()


debug = False
if debug == True:
    model = CRNN()
    debug_shapes(model, (32, 40, 500))
