# Handwriting-Recognition
This project-repo contains code that implements this [paper](https://arxiv.org/pdf/1507.05717.pdf) which was originally for scene-text-recognition for the handwriting recognition task. 

# Data
The data is taken from [Iam-Offline-DataBase](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and [Cvl-DataBase](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/).

# Implementation details
The architecture takes in a handwritten word and spits out machine-readable string.
### Model Details
1. The model contains a 6-layered Cnn that spits out a tensor batch_size x 32 x 1 x 512.
2. Then 2 bidirectional-layer of 512 Lstm Units are used to get the tensor batch_size x time_stamps x chars.
3. Finally the CTC-Loss is used to find the loss at the output and then we backprop the gradient of loss.

### Results
The model was trained on a total of approx(200,000) words from Iam + Cvl Dataset alongwith Data-Augmentation with probablity(p==0.5). And a accuracy of char-accuracy of 9.34% was achieved on the dataset. The data was divided into 90%-10% train-test dataset. And edit-distance between predicted string (s2) and ground-truth (s1) to define the character error-rate. 

