from data import load_data, load_merged_data
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
            nn.Conv2d(1, 96, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            # Second
            nn.Conv2d(96, 96, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Third
            nn.Conv2d(96, 96, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Fourth
            nn.Conv2d(96, 96, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Gated Recurrent Unit (GRU) layers
        self.gru_layers = nn.GRU(
            96, 96, num_layers=2, batch_first=True, bidirectional=True
        )

        # Temporal max-pooling layer
        self.temporal_max_pooling = nn.AdaptiveMaxPool1d(1)

        # Feedforward layer
        self.feedforward_layer = nn.Sequential(
            nn.Dropout(p=0.25), nn.BatchNorm1d(192), nn.Linear(192, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional layers      
        print(x.shape)
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


def train():
    train_dataset, train_loader, val_dataset, val_loader = load_merged_data(32)

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
        # set model to train mode
        model.train()

        # iterate over batches in the training set
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # evaluate model on validation set
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(inputs)

                # calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # save predictions and labels
                val_preds += outputs.cpu().numpy().tolist()
                val_labels += labels.cpu().numpy().tolist()

            # calculate validation metrics
            val_loss /= len(val_dataset)
            val_auc = roc_auc_score(val_labels, val_preds)

            # print validation metrics
            print(
                "Epoch [{}/{}], Val Loss: {:.4f}, Val AUC: {:.4f}".format(
                    epoch + 1, num_epochs, val_loss, val_auc
                )
            )

            # check for early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    print(
                        "Early stopping after {} epochs without improvement.".format(
                            patience
                        )
                    )
                    break

train()