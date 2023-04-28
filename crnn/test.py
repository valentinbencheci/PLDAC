from data import AudioDataset
from model import CRNN
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

# create a new instance of the AudioDataset class for the test set
test_dataset = AudioDataset(
    csv_file="../data/BirdVoxDCASE20k_csvpublic.csv",
    audio_dir="../data/BirdVox-DCASE-20k/wav",
)

# create a DataLoader object to iterate over the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN().to(device)
model.load_state_dict(torch.load("best_model.pt"))

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
val_auc = roc_auc_score(val_targets, val_preds)
print(f"=> Val AUC: {val_auc:.4f}")

cm = confusion_matrix(val_targets, val_preds)
print(cm)
