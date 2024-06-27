import torch
import os
import matplotlib.pyplot as plt
from Cnn import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchinfo import summary

path=(os.getcwd())
test_ds= ImageFolder(root=path+'/test', transform=ToTensor())

#Paste logged details here
training_loss_details=  {'Cnn4A1LR1': [1.04921, 0.66649, 0.57215, 0.53678, 0.50899, 0.4539, 0.43337, 0.4129, 0.41451, 0.40145, 0.39829, 0.38711, 0.36118, 0.34262, 0.33223, 0.29454, 0.31699, 0.30822, 0.26366, 0.2445, 0.23411, 0.23242, 0.23635, 0.21824]}
val_loss_details=  {'Cnn4A1LR1': [0.80226, 0.53901, 0.47204, 0.48295, 0.45022, 0.41286, 0.42044, 0.42067, 0.38248, 0.37992, 0.39506, 0.34473, 0.34868, 0.34211, 0.33082, 0.43883, 0.3215, 0.34935, 0.30383, 0.29612, 0.30512, 0.29316, 0.26812, 0.30686]}
training_f1_details=  {'Cnn4A1LR1': [0.4728, 0.7039, 0.764, 0.7798, 0.7808, 0.8233, 0.8318, 0.8403, 0.8354, 0.8411, 0.8409, 0.8384, 0.8515, 0.8666, 0.8624, 0.8845, 0.8692, 0.871, 0.8958, 0.905, 0.903, 0.9083, 0.903, 0.9158]}
val_f1_details=  {'Cnn4A1LR1': [0.5608, 0.7822, 0.7974, 0.8169, 0.8437, 0.8289, 0.8365, 0.8218, 0.8363, 0.8454, 0.8571, 0.8637, 0.85, 0.8523, 0.8531, 0.8037, 0.8671, 0.8646, 0.8675, 0.8879, 0.8876, 0.885, 0.8904, 0.8794]}

def seeloss(graph1, graph2, model):
    num_epochs = len(graph1)
    steps = range(1, num_epochs + 1)
    scaled_steps = [step * 41 for step in steps]
    plt.figure(figsize=(12,8))
    plt.plot(scaled_steps, graph1, label='Training Loss')
    plt.plot(scaled_steps, graph2, linestyle='--', label='Validation Loss')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.title(f'{model} Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def seef1(graph1, graph2, model):
    num_epochs = len(graph1)
    steps = range(1, num_epochs + 1)
    scaled_steps = [step * 41 for step in steps]
    plt.figure(figsize=(12,8))
    plt.plot(scaled_steps, graph1, label='Training f1')
    plt.plot(scaled_steps, graph2, linestyle='--', label='Validation f1')
    plt.xlabel('steps')
    plt.ylabel('f1')
    plt.title(f'{model} Training and Validation f1 Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

for model in training_f1_details.keys():
    seeloss(training_loss_details[model], val_loss_details[model], model)
    seef1(training_f1_details[model], val_f1_details[model], model)

path=(os.getcwd()+'/Cnn_model/Cnnstate.pt')
model=Cnn4A1LR1()
checkpoint=torch.load(path)
model.load_state_dict(checkpoint['Cnn4A1LR1'])

print(f'Model 4 state dict size: {sum(p.numel() for p in model.parameters()) * 4 / 1e6} MB')
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
summary(model, (3,299,299))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print('model summary')
summary(model, (3,299,299))