# Historical source export. See archive/README.md; use esdl modules to run the project.

# %% Cell 1 (code)
# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import random
import requests
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% Cell 2 (code)
# Parameters
landa = 1 # unit meters
Dobjlength = 1 * landa # length and width of the object domain
Ni = 20 # Number of transmiters
M = 20 # Number of receivers
RD = 20 * landa # Distance of the receivers from origin
k0 = 2 * np.pi / landa # propagation constant
E0 = 1 # amplitude of the electric field
Z0 = 376.7303 # Intrinsic impedance of free space

# %% Cell 3 (code)
Domains = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Domains.npy')).type(torch.cuda.FloatTensor)
Field_Data = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Field_Data.npy')).type(torch.cuda.FloatTensor)

Domains_test = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Domains_test.npy')).type(torch.cuda.FloatTensor)
Field_Data_test = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Field_Data_test.npy')).type(torch.cuda.FloatTensor)

# %% Cell 4 (code)
# from torch.utils.data import Dataset
# import pickle

# class NumpyDataset(Dataset):
#   def __init__(self, data, labels):
#     self.data = data
#     self.labels = labels
  
#   def __len__(self):
#     return len(self.data)
  
#   def __getitem__(self, index):
#     data_item = self.data[index]
#     label_item = self.labels[index]
#     return data_item, label_item

# filename = "/content/drive/MyDrive/Colab Notebooks/dataset_train.pkl"
# with open(filename, "rb") as file:
#   dataset_train = pickle.load(file)
    
# filename = "/content/drive/MyDrive/Colab Notebooks/dataset_test.pkl"
# with open(filename, "rb") as file:
#   dataset_test = pickle.load(file)

# %% Cell 5 (code)
# Make PyTorch DataLoader
# batch_size = 100

# # Dataset Used to Train the Model
# train_dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False)

# # Dataset Used to Test the Model
# test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

# %% Cell 6 (code)
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %% Cell 7 (code)
# Make Model 4
# The structure can be seen below
class AlexNet(nn.Module):  
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels=2, out_channels=50, kernel_size=2, stride=(2,2),padding_mode='zeros', padding=0, bias=True),
      nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(in_channels=50, out_channels=100, kernel_size=2, stride=(1,1),padding_mode='zeros', padding='same', bias=True),
      nn.ReLU(),
    )

    self.fc1 = nn.Linear(14*14*100, 28*28*25, bias=True)
    self.fc2 = nn.Linear(28*28*25, M*Ni*2, bias=True)

  def forward(self, x):
    out = self.conv1(x)
    # print(1,out.shape)
    out = self.conv2(out)
    # print(2,out.shape)
    out = out.view(out.size(0), -1)
    # print(3,out.shape)

    out = self.fc1(out)
    # print(4,out.shape)
    out = self.fc2(out)
    # print(5,out.shape)
    return out

# %% Cell 8 (code)
model = AlexNet().to(device)
model

# %% Cell 9 (code)
# Getting helper functions from my friend daniel bourke

if Path("helper_functions.py").is_file():
  print("Python script already downloaded")
else:
  print("Downloading helper_functions.py script")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# %% Cell 10 (code)
from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer

# Creating and choosing a loss function and an optimizer
# Change the learning rate to make the network learn faster (and generally less accurate) or slower (and generally more accurate)
LEARNING_RATE = 5e-5

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# %% Cell 11 (code)
def accuracy_func(y_true, y_pred):
  return torch.sum(np.abs(y_test.cpu()- test_pred.cpu()))

# %% Cell 12 (code)
# Start Timer
train_time_start_on_cpu = timer()

# Number of Epochs - Change this to train the network for longer
EPOCHS = 400

bestloss = 5
# Training Loop
for epoch in tqdm(range(EPOCHS)):
  print(f"Epoch: {epoch}")
  train_loss = 0

  X = Domains.to(device)
  # print(int(i*batch_size),int((i+1)*batch_size))
  y = Field_Data.to(device)
  y = y.view(y.size(0), -1)
  # Set model to train
  model.train()
  # Perform the forward pass
  y_pred = model(X)
  # Calculate the loss (how bad our model is at predicting the right values)
  loss = loss_fn(y_pred, y)
  # Accumulate the loss
  train_loss = loss
  # Set the gradients of the optimizer to zero
  optimizer.zero_grad()
  # Perform backpropagation
  loss.backward()
  # Performs a single optimization step
  optimizer.step()

  # Test the model on unseen data
  test_loss, test_acc = 0,0
  mesqermax = 0
  model.eval()

  with torch.inference_mode():
    X_test = Domains_test.to(device)
    y_test = Field_Data_test.to(device)
    y_test = y_test.view(y_test.size(0), -1)
    # Our model predicts on test data
    test_pred = model(X_test)
    # Calculate loss and accuracy
    test_loss = loss_fn(test_pred, y_test)
    # print('test pred: ',test_pred.shape)
    # print('y test: ',y_test.shape)
    test_acc += accuracy_func(y_true=y_test, y_pred=test_pred)
    mesqermax += torch.max(torch.sum(np.square(np.abs(y_test.cpu()-test_pred.cpu())),axis=1)/len(test_pred[0].cpu()))

  if bestloss > train_loss:
    bestloss = train_loss
    print('Better')
  print(f"Train loss: {train_loss:.10f} | Test loss: {test_loss:.10f} | Accuracy: {test_acc:.10f} | MSEmax: {mesqermax:.10f}\n")

  # nnn = 0
  # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  # Dobj = ax1.imshow(Domains_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  # fig.colorbar(Dobj, ax=ax1)
  # fd1 = ax2.imshow(Field_Data_test[nnn,0].cpu(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd1, ax=ax2)
  # fd2 = ax3.imshow(Field_Data_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd2, ax=ax3)
  # plt.show()

  # xvis = torch.unsqueeze(Domains_test[nnn],0)
  # yvis = model(xvis)
  # y_pr = yvis.view(2, 20, 20)

  # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  # Dobj = ax1.imshow(Domains_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  # fig.colorbar(Dobj, ax=ax1)
  # fd1 = ax2.imshow(y_pr[0].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd1, ax=ax2)
  # fd2 = ax3.imshow(y_pr[1].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd2, ax=ax3)
  # plt.show()
  # print('')
# Stop timer
train_time_end_on_cpu = timer()
# Calculate timer difference
total_train_time_model = print_train_time(start=train_time_start_on_cpu, 
                                            end=train_time_end_on_cpu,
                                            device=str(next(model.parameters()).device))

# %% Cell 13 (code)
# def save_checkpoint(state, filename='/content/drive/MyDrive/Colab Notebooks/my_checkpoint.pth.tar'):
#   print('saveing checkpoint')
#   torch.save(state, filename)

# def load_checkpoint(checkpoint)
#   print('loading checkpoint')
#   model.load_state_dict(checkpoint['state_dict'])
#   optimizer.load_state_dict(checkpoint['optimizer'])

# load_checkpoint(torch.load('/content/drive/MyDrive/Colab Notebooks/my_checkpoint.pth.tar'))
# checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
# save_checkpoint(checkpoint)

# %% Cell 14 (code)
for nnn in range(0,2,2):
  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  Dobj = ax1.imshow(Domains_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  fig.colorbar(Dobj, ax=ax1)
  fd1 = ax2.imshow(Field_Data_test[nnn,0].cpu(), cmap="viridis", interpolation='none')
  fig.colorbar(fd1, ax=ax2)
  fd2 = ax3.imshow(Field_Data_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  fig.colorbar(fd2, ax=ax3)
  plt.show()

  xvis = torch.unsqueeze(Domains_test[nnn],0)
  yvis = model(xvis)
  y_pr = yvis.view(2, 20, 20)

  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  Dobj = ax1.imshow(Domains_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
  fig.colorbar(Dobj, ax=ax1)
  fd1 = ax2.imshow(y_pr[0].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  fig.colorbar(fd1, ax=ax2)
  fd2 = ax3.imshow(y_pr[1].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  fig.colorbar(fd2, ax=ax3)
  plt.show()
  print('accuracy is: ',accuracy_func(Field_Data_test[nnn], y_pr))
  print('')
