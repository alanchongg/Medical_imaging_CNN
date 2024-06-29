# Medical_imaging
Using Convoluitonal neural network done using PyTorch to classifiy lung diseases  
soruce of dataset: https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data  
This default hyperparatemer has the best results based on experimental testing done concidering both model parameter and training parameters.  

The downloaded dataset can be placed in the same folder, using the prep.py will split the data into 3 different sets: trianing, validation, testing.  
The Main file can be used to change the hyperparameters of the training process prior to running it.  
Training details will be saved iin a txt file which can be used by visualisation.py to view training and validation loss, training and validation F1 score.
Trained model will be saved in the Cnnstate.pt which can be used by the tk.py which provides a intuative and funcational GUI for making prediction.  
