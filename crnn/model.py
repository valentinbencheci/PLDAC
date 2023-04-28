from data import load_data, load_merged_data
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_auc_score, f1_score


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


def train(
    model, criterion, optimizer, scheduler, batch_size, split_size, num_epochs, patience
):
    train_loader, val_loader = load_merged_data(batch_size, split_size)

    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_auc = 0.0
    counter = 0

    # training loop
    since = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-" * 50)

        for phase in ["train", "eval"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataset = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataset = val_loader

            val_loss = 0.0
            val_preds = []
            val_labels = []

            # iterate over batches
            for inputs, labels in tqdm(dataset):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # reshape labels
                labels = labels.unsqueeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = outputs.round()
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                val_loss += loss.item() * inputs.size(0)
                val_preds += preds.detach().cpu().numpy().tolist()
                val_labels += labels.detach().cpu().numpy().tolist()

            # learning rate scheduler
            if phase == "train":
                scheduler.step()

            # statistics
            val_loss /= len(train_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)

            print(
                f"[{phase}] Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}"
            )

        print()
        
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

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val AUC: {best_auc:4f}")

    return model


if __name__ == "__main__":
    # Create an instance of the model
    model = CRNN()

    # define optimizer, learning scheduler and loss function
    optimizer = optim.Adam(model.parameters())
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.BCELoss()

    # define batch size, number of epochs and early stopping
    batch_size = 64
    split_size = 0.6
    num_epochs = 100
    patience = 50

    train(
        model,
        criterion,
        optimizer,
        scheduler,
        batch_size,
        split_size,
        num_epochs,
        patience,
    )


def debug_shapes(model, input_shape):
    # Define hook function to print shape of input and output tensors
    def print_shape(module, input, output):
        print(f"Module: {module.__class__.__name__}")
        print(f"Shape of input tensor: {input[0].shape}")
        for i, o in enumerate(output):
            print(f"Shape of output tensor {i}: {o.shape}")
        print("-" * 50)

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
