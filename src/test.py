import torch
import model
import file_manager

import file_manager

def classify_audio(audio_path, model, device):
    model.eval()
    with torch.no_grad():
        audio_tensor = file_manager.preprocess_audio_model(audio_path)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)

        if (audio_tensor.numel()>0):
            output = model(audio_tensor)

            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

            return predicted_class.item(), probabilities

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
model = model.AudioClassificationModel()
model.load_state_dict(torch.load("audio_classifier_model_1.pth"))
model = model.to(device)

# Path to the audio file
birds_path = '../data/birds/signal_chunks/'
other_path = '../data/other/signal_chunks/'
dataset = []
correct = 0
count = 0

dataset = file_manager.create_dataset(testFlag=True)
print(dataset)
for (file, cl) in dataset:
    count += 1
    predicted_class, probabilities = classify_audio(file, model, device)
    if (predicted_class == cl):
        correct += 1

print("Total:", count)
print("Correct:", correct)
print('Rapport:', correct/count)
