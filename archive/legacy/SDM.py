# Historical source export. See archive/README.md; use esdl modules to run the project.

# %% Cell 1 (code)
# Import libraries

import torch
from torch import nn
import random
import requests
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy
import os
import gc
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% Cell 2 (code)
from google.colab import drive
drive.mount('/content/drive')

# %% Cell 3 (code)
# Parameters

landa = 1 # unit meters
Dobjlength = 1 * landa # length and width of the object domain
Ni = 20 # Number of transmiters
M = 20 # Number of receivers
RD = 20 * landa # Distance of the receivers from origin
k0 = 2 * np.pi / landa # propagation constant
E0 = 1 # amplitude of the electric field
Z0 = 376.7303 # Intrinsic impedance of free space

# %% Cell 4 (code)
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %% Cell 5 (code)
# The structure of out DConvNet can be seen below
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
    out = self.conv2(out)
    out = out.view(out.size(0), -1)

    out = self.fc1(out)
    out = self.fc2(out)
    return out

# %% Cell 6 (code)
model = AlexNet().to(device)
model

# %% Cell 7 (code)
# Getting helper functions for visualizing and error evaluation

if Path("helper_functions.py").is_file():
  print("Python script already downloaded")
else:
  print("Downloading helper_functions.py script")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# %% Cell 8 (code)
from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer

# %% Cell 9 (code)
# This cell is for evaluating the output of the model

def RMS_X(delXk, Xt):
  return torch.mean(torch.div(torch.square(torch.sum(torch.pow(delXk,2),1)), torch.square(torch.sum(torch.pow(Xt,2),1))))

def RMS_E(delEk, Et):
  return torch.mean(torch.div(torch.square(torch.sum(torch.pow(delEk,2),1)), torch.square(torch.sum(torch.pow(Et,2),1))))

def trainloss(Xk, Xt):
  return torch.mean(torch.sum(torch.square(torch.sub(Xk,Xt)), dim=0))

def testloss(Xk, Xt):
  return torch.sum(torch.abs(torch.sub(Xk,Xt)))

# %% Cell 10 (code)
#This cell is for loading and saving the model
def load_checkpoint(checkpoint):
  print('loading checkpoint')
  model.load_state_dict(checkpoint['state_dict'])

load_checkpoint(torch.load('/content/drive/MyDrive/Colab Notebooks/my_checkpoint.pth.tar'))

# %% Cell 11 (markdown)
# #SDM offline training

# %% Cell 12 (code)
# This cell is for loading the required files and handeling the files needed for training the SDM
Xt = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Domains.npy')).type(torch.cuda.FloatTensor)
Et = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Field_Data.npy')).type(torch.cuda.FloatTensor)
Xk = torch.load('/content/drive/MyDrive/Colab Notebooks/Xk.pt')
Et = Et.view(len(Et),len(Et[0])*len(Et[0,0])*len(Et[0,0,0]))
Xk = Xk.view(len(Xk),len(Xk[0])*len(Xk[0,0])*len(Xk[0,0,0]))
Xt = Xt.view(len(Xt),len(Xt[0])*len(Xt[0,0])*len(Xt[0,0,0]))
iterations = 800
counter = 0
alpha = 0.5
Ks = torch.zeros((iterations, len(Et[0]), len(Xt[0])), dtype=torch.float)

# %% Cell 13 (code)
# This cell is the core of the SDM training
model.eval()
bestMSE = 99999999
# ploting the initial value of the contrast
plt.imshow(Xk[0].view(2,28,28)[1].cpu())
plt.colorbar()
plt.show()

for i in range(iterations):
  with torch.inference_mode():
    Ek = model(Xk.view(len(Xk), 2, 28, 28))
    delEk = torch.sub(Ek, Et)
    delXk = torch.sub(Xt, Xk)
    U, S, V = torch.svd(delEk)
    Kk = torch.matmul(V, torch.matmul(torch.inverse(torch.transpose(torch.diag(torch.pow(S,2)),0,1)+np.power(alpha,2)*torch.diag(torch.squeeze \
    (torch.ones((1,len(S)), device=device, dtype=torch.float)))), torch.matmul(torch.transpose(torch.diag(S),0,1), torch.matmul(torch.transpose(U,0,1),delXk))))
    Ks[i] = Kk
    Xk = Xk + torch.matmul(delEk, Kk)

  # j = 0
  # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  # Dobj = ax1.imshow(Xk.view(5000, 2, 28, 28)[j,1].cpu(), cmap="viridis", interpolation='none')
  # fig.colorbar(Dobj, ax=ax1)
  # fd1 = ax2.imshow(Ek[j].view(2,20,20)[0].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd1, ax=ax2)
  # fd2 = ax3.imshow(Ek[j].view(2,20,20)[1].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  # fig.colorbar(fd2, ax=ax3)
  # plt.show()

  MSEE = trainloss(Xk,Xt)
  print(f"i: {i} | RMS X: {RMS_X(delXk, Xt)*1e6:.10f} | RMS E: {RMS_E(delEk, Et)*1e6:.10f} | MMSE: {MSEE:.10f} | diff: {bestMSE - MSEE:.10f}\n")

  if bestMSE - MSEE <= 0.02:
    Ks = Ks[0:i+1]
    break
  if MSEE < bestMSE:
    bestMSE = MSEE

# %% Cell 14 (code)
# This cell is for saving or loading the calculated decent directions

torch.save(Ks,'/content/drive/MyDrive/Colab Notebooks/Ks.pt')
# Ks = torch.load('/content/drive/MyDrive/Colab Notebooks/Ks.pt')

# %% Cell 15 (markdown)
# #SDM online testing

# %% Cell 16 (code)
# This part is for loading the MNIST testing dataset
Xt = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Domains_test.npy')).type(torch.cuda.FloatTensor)
Et = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Field_Data_test.npy')).type(torch.cuda.FloatTensor)
Xk = torch.load('/content/drive/MyDrive/Colab Notebooks/Xk.pt').type(torch.cuda.FloatTensor)

# This part is for loading the Cylindrical testing dataset
# Xt = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Domains_cy.npy')).type(torch.cuda.FloatTensor)
# Et = torch.from_numpy(np.load('/content/drive/MyDrive/Colab Notebooks/Field_Data_cy.npy')).type(torch.cuda.FloatTensor)
# Xk = torch.load('/content/drive/MyDrive/Colab Notebooks/Xk.pt').type(torch.cuda.FloatTensor)

Et = Et.view(len(Et),len(Et[0])*len(Et[0,0])*len(Et[0,0,0]))
Xk = Xk.view(len(Xk),len(Xk[0])*len(Xk[0,0])*len(Xk[0,0,0]))
Xt = Xt.view(len(Xt),len(Xt[0])*len(Xt[0,0])*len(Xt[0,0,0]))

# This loop is for operating the traind SDM for muliple observed scattered fields
for j in range(20):
  # Start Timer
  train_time_start_on_cpu = timer()

  Xk_temp = Xk[j]
  Et_temp = Et[j]
  Xt_temp = Xt[j]

  for i in tqdm(range(len(Ks))):
    with torch.inference_mode():
      Ek = model(Xk_temp.view(1, 2, 28, 28))
      delEk = torch.sub(Ek, Et_temp)
      Xk_temp = Xk_temp + torch.matmul(delEk, Ks[i].to(device))
    # if i%5 == 0:
    #   plot = plt.imshow(Xk_temp.view(2, 28, 28)[1].cpu(), cmap="viridis", interpolation='none')
    #   plot.set_clim(vmin=1)
    #   plt.title('Iteration = '+str(i))
    #   plt.show()
    #   time.sleep(0.2)
  # with torch.inference_mode():
  #   Xk_temp = torch.mul((Xk_temp >= 1), Xk_temp) + (Xk_temp < 1)

  print(testloss(Xk_temp, Xt_temp))
  print(trainloss(Xk_temp, Xt_temp))

  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
  Dobj = ax1.imshow(Xk_temp.view(2, 28, 28)[1].cpu(), cmap="viridis", interpolation='none')
  fig.colorbar(Dobj, ax=ax1)
  Dobj.set_clim(vmin=1, vmax=2)
  ax1.axes.get_xaxis().set_ticks([])
  ax1.axes.get_yaxis().set_ticks([])
  fd1 = ax2.imshow(Xt_temp.view(2,28,28)[1].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  fig.colorbar(fd1, ax=ax2)
  ax2.axes.get_xaxis().set_ticks([])
  ax2.axes.get_yaxis().set_ticks([])
  fd2 = ax3.imshow(Ek.view(2,20,20)[0].cpu().detach().numpy(), cmap="viridis", interpolation='none')
  fig.colorbar(fd2, ax=ax3)
  ax3.axes.get_xaxis().set_ticks([])
  ax3.axes.get_yaxis().set_ticks([])
  plt.show()

  # Stop timer
  train_time_end_on_cpu = timer()
  # Calculate timer difference
  total_train_time_model = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model.parameters()).device))
  print("")

# %% Cell 17 (code)
# yvis = model(Xt.view(5000, 2, 28, 28))
# print(yvis.shape)
# for nnn in range(5):
#   # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
#   # Dobj = ax1.imshow(Domains_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
#   # fig.colorbar(Dobj, ax=ax1)
#   # fd1 = ax2.imshow(Field_Data_test[nnn,0].cpu(), cmap="viridis", interpolation='none')
#   # fig.colorbar(fd1, ax=ax2)
#   # fd2 = ax3.imshow(Field_Data_test[nnn,1].cpu(), cmap="viridis", interpolation='none')
#   # fig.colorbar(fd2, ax=ax3)
#   # plt.show()

#   A = yvis[nnn]
#   y_pr = A.view(2, 20, 20)

#   fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
#   Dobj = ax1.imshow(Xt.view(5000, 2, 28, 28)[nnn,1].cpu(), cmap="viridis", interpolation='none')
#   fig.colorbar(Dobj, ax=ax1)
#   fd1 = ax2.imshow(y_pr[0].cpu().detach().numpy(), cmap="viridis", interpolation='none')
#   fig.colorbar(fd1, ax=ax2)
#   fd2 = ax3.imshow(y_pr[1].cpu().detach().numpy(), cmap="viridis", interpolation='none')
#   fig.colorbar(fd2, ax=ax3)
#   plt.show()
#   # print('MSE is: ' , torch.mean(torch.square(Field_Data_test[nnn].view(-1) - yvis[0])).cpu().detach().numpy())
#   # print('')

# %% Cell 18 (code)
# This cell is for making the initial contrst

# Xk = torch.zeros((5000, 2, 28, 28), device=device, dtype=torch.float)
# for a in range(len(Xk)):
#   Xk[a,1,5:23,5:23] = torch.ones(Xk[a,1,5:23,5:23].shape)*2
# torch.save(Xk,'/content/drive/MyDrive/Colab Notebooks/Xk.pt')

# Xk = torch.ones((5000, 2, 28, 28), device=device, dtype=torch.float)*2
# torch.save(Xk,'/content/drive/MyDrive/Colab Notebooks/Xk.pt')

# %% Cell 19 (code)

