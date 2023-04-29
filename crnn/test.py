from data import AudioDataset
from model import CRNN
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


def eval(test_dataset, batch_size, best_model):
    # create a DataLoader object to iterate over the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    model.load_state_dict(torch.load(best_model))

    # Evaluate the model on the validation set
    model.eval()
    val_preds = []
    val_targets = []

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.round()
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    # Metrics
    metrics = {}
    metrics["auc"] = roc_auc_score(val_targets, val_preds)
    metrics["acc"] = accuracy_score(val_targets, val_preds)
    metrics["f1"] = f1_score(val_targets, val_preds)
    metrics["cm"] = confusion_matrix(val_targets, val_preds)

    return metrics


if __name__ == "__main__":
    # create a new instance of the AudioDataset class for the test set
    test_dataset = AudioDataset(csv_file="../data/warblrb10k_public_metadata.csv", audio_dir="../data/warblrb10k_public/wav/", data_slice=10)

    metrics = eval(test_dataset, 32, "best_model.pt")

    print(metrics)
