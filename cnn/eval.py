import os
import torch
import cnn_tr1
import cnn_tr2
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(test_path, model_path, modelFlag=0, csv="../../dataset/bird_dataset.csv"):
    dataset = cnn_tr1.create_dataset(test_path, csv)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (modelFlag == 0):
        model = cnn_tr1.AudioClassificationModel().to(device)
    elif (modelFlag == 1):
        model = cnn_tr1.AudioClassificationModelDense().to(device)
    elif (modelFlag == 2):
        model = cnn_tr1.AudioClassificationModelAvgPool().to(device)
    else:
        model = model = cnn_tr1.AudioClassificationModelSoftmax().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    f1 = f1_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)

    return f1, auc

def evaluate_model2(test_path, model_path, csv="../../dataset/bird_dataset.csv"):
    dataset = []
    df = pd.read_csv(csv)

    for i in range(len(test_path)):
        files = os.listdir(test_path[i])

        for file_i in range(len(files)):
            audio_path = os.path.join(test_path[i], files[file_i])
            spectrogram = cnn_tr2.preprocess_audio_model(audio_path)

            target_item_id = files[file_i].split('.')[0]
            filtered_df = df[df['itemid'] == target_item_id]
            dataset.append((spectrogram.unsqueeze(0), filtered_df['hasbird'].iloc[0]))

    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn_tr2.AudioClassificationModel(input_size=(1, 128, 862))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(label.numpy())

    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)

    return f1, auc

# print('\n-- CNN | TR_2 | default')
# print("Dataset : BirdVoxDCASE20k")
# f1, auc = evaluate_model2(['../../dataset/test/pre_traitement_2/BirdVoxDCASE20k/'], "cnn_tr2.pth")
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : ff1010bird")
# f1, auc = evaluate_model2(['../../dataset/test/pre_traitement_2/ff1010bird/'], "cnn_tr2.pth")
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : warblrb10k")
# f1, auc = evaluate_model2(['../../dataset/test/pre_traitement_2/warblrb10k/'], "cnn_tr2.pth")
# print("F1 Score: ", f1)
# print("AUC: ", auc)

# print('\n-- CNN | TR_1 | default')
# print("Dataset : BirdVoxDCASE20k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/BirdVoxDCASE20k/'], "cnn_tr1.pth", modelFlag=0)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : ff1010bird")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/ff1010bird/'], "cnn_tr1.pth", modelFlag=0)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : warblrb10k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/warblrb10k/'], "cnn_tr1.pth", modelFlag=0)
# print("F1 Score: ", f1)
# print("AUC: ", auc)

# print('\n-- CNN | TR_1 | dense')
# print("Dataset : BirdVoxDCASE20k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/BirdVoxDCASE20k/'], "cnn_tr1_dense.pth", modelFlag=1)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : ff1010bird")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/ff1010bird/'], "cnn_tr1_dense.pth", modelFlag=1)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : warblrb10k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/warblrb10k/'], "cnn_tr1_dense.pth", modelFlag=1)
# print("F1 Score: ", f1)
# print("AUC: ", auc)

# print('\n-- CNN | TR_1 | avg_pool')
# print("Dataset : BirdVoxDCASE20k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/BirdVoxDCASE20k/'], "cnn_tr1_avg_pool.pth", modelFlag=2)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : ff1010bird")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/ff1010bird/'], "cnn_tr1_avg_pool.pth", modelFlag=2)
# print("F1 Score: ", f1)
# print("AUC: ", auc)
# print("Dataset : warblrb10k")
# f1, auc = evaluate_model(['../../dataset/test/pre_traitement_1/warblrb10k/'], "cnn_tr1_avg_pool.pth", modelFlag=2)
# print("F1 Score: ", f1)
# print("AUC: ", auc)