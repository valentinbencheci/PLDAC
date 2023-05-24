import torch
import rnn_tr1
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(test_dataset, model_path, modelFlag=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = test_dataset[0][0].shape[-1]
    
    if (modelFlag == 0):
        model = rnn_tr1.AudioClassificationModel(input_size=input_size).to(device)
    elif (modelFlag == 1):
        model = rnn_tr1.AudioClassificationModelDense(input_size=input_size).to(device)
    elif (modelFlag == 2):
        model = rnn_tr1.AudioClassificationModelGRU(input_size=input_size).to(device)
    else:
        model = rnn_tr1.AudioClassificationModelBiLSTM(input_size=input_size).to(device)
        
    model.load_state_dict(torch.load(model_path))

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model.eval()
    test_predictions = []
    test_targets = []
    test_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            probs = softmax(outputs, dim=1)

            test_scores.extend(probs.cpu().numpy()[:, 1]) 
            test_predictions.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    f1 = f1_score(test_targets, test_predictions, average='weighted')
    auc = roc_auc_score(test_targets, test_scores)

    return f1, auc