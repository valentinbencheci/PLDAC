# PLDAC "Chants d’oiseaux"

Il s'agit de concevoir un système qui, à partir d'un court enregistrement audio, donne une décision binaire sur la présence ou l'absence d'un bruit d'oiseau (quel qu'il soit).

Les étapes de ce projet seront les suivantes :

- mise au point d’un protocole expérimental
- études des différents pré-traitements possibles pour les données audio
- construction de baselines à l’aide d’algorithme non-deep learning (`sklearn`)
- prise en main d’une plateforme deep learning (`pytorch`)
- réflexion sur les contraintes spécifiques
- pour les "chunks" il faudra faire une fonction qui prend la moyenne des predictions
- on peut faire des autre modeles: Recurrent Neural Networks (RNNs); 1D Convolutional Neural Networks (1D CNNs); Time-Distributed CNN-LSTM; Transformers; WaveNet.
