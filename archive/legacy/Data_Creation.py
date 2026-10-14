# Historical source export. See archive/README.md; use esdl modules to run the project.

# %% Cell 1 (code)
# Import libraries

from IPython.display import display, HTML
display(HTML("<style>.container { width:95% ! important; }<style>"))
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import random
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% Cell 2 (code)
AlexTransform = transforms.Compose([
    transforms.ToTensor(),
])

# %% Cell 3 (code)
# Creating and downloading the MNIST dataset

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=AlexTransform,
    target_transform=None
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=AlexTransform,
    target_transform=None
)

# type(train_data[0])
# print(train_data[0])

# %% Cell 4 (code)
# This cell creates the desired scatteres

if os.path.exists("Domains.npy") == False:
    NumRandDomain = 5000;
    random_integers = np.random.randint(low=0, high=60000, size=NumRandDomain)

    Domains = np.zeros((NumRandDomain, 28, 28))
    labels = np.zeros((NumRandDomain))
    for v in range(len(random_integers)):
        image, label = train_data[random_integers[v]]
        image = image + np.ones(image.shape)
        Domains[v,:,:] = image.squeeze()
        labels[v] = label
        print(v)

      # plt.imshow(Domains[v].squeeze(), cmap="viridis")
      # plt.colorbar()
      # plt.title("Number "+str(label))
      # plt.show()

    np.save("Domains.npy", Domains)
    np.save("Labels.npy", labels)
else:
    print('The file already exists')
    Domains = np.load("Domains.npy")
    Labels = np.load("Labels.npy")

# %% Cell 5 (code)
# This cell is for the number of times a digit was used in the domain
count = np.zeros((10))
indi = range(10)

for a in indi:
    count[a] = len(Labels[Labels == a])
    print('number of ' + str(a) + ' is : ' + str(count[a]))

# %% Cell 6 (markdown)
# # Calculating the scattered wave with MoM From different directions of incidented wave

# %% Cell 7 (code)
# Functions

# Calculates the incidented electric field
def E_Inc(x, y, E0, phi, k0):
    return E0 * np.exp(1j * k0 * (x * np.cos(phi) + y * np.sin(phi)))

# Calculates the Zmn matrix for the MoM
def Z_mn_func(m, n, xm, ym, xn, yn, k0, Z0, s):
    gama = 1.7180724
    if m == n:
        return k0 * Z0 * s / 4 * (1 - 2j / np.pi * np.log(k0*gama*s/4/np.exp(1)))
    if m != n:
        return k0 * Z0 * s / 4 * scipy.special.hankel2(0, k0*np.sqrt(np.power(xm-xn,2) + np.power(ym-yn,2)))

# Calculates the Zmn_h matrix for the MoM
def Z_mn_h_func(k0, eps_r, Z0):
        return -1j * Z0 / (k0 * (eps_r - 1))

# Calculates the scattered electric field
def E_Sct(xs, ys, xo, yo, Jn, k0, landa, s):
    Z0 = 376.7303
    return -k0 * Z0 / 4 * Jn * s *\
        scipy.special.hankel2(0, k0*np.sqrt(np.power(xs-xo,2) + np.power(ys-yo,2)))

# %% Cell 8 (code)
# Parameters

landa = 1 # unit meters
Dobjlength = 1 * landa # length and width of the object domain
Ni = 20 # Number of transmiters
M = 20 # Number of receivers
RD = 20 * landa # Distance of the receivers from origin
k0 = 2 * np.pi / landa # propagation constant
E0 = 1 # amplitude of the electric field
Z0 = 376.7303 # Intrinsic impedance of free space

# %% Cell 9 (code)
# This cell calculates the scattered fields in each reciver using MoM

if os.path.exists("Field_Data.npy") == False:
    phi_transmiter = np.arange(0, 2*np.pi, 2*np.pi/Ni)
    phi_reciever = np.arange(0, 2*np.pi, 2*np.pi/M)
    xR = RD * np.cos(phi_reciever)
    yR = RD * np.sin(phi_reciever)
    Field_Data = np.zeros((len(Domains), M, Ni), dtype=complex)
    for i in range(len(Domains)):
        count = len(Domains[i][Domains[i]>1])
        delxy = Dobjlength / len(Domains[i])
        s = np.power(delxy, 2)
        xmin = -Dobjlength / 2

        x_pos = np.zeros(count)
        y_pos = np.zeros(count)
        eps_r = np.zeros(count)
        count = 0
        
        # This for loop calculates the center position of each fragment
        for j in range(len(Domains[i])):
            for k in range(len(Domains[i,0])):
                if Domains[i,j,k] > 1:
                    x_pos[count] = xmin + (k + 0.5) * delxy
                    y_pos[count] = -xmin - (j + 0.5) * delxy
                    eps_r[count] = Domains[i,j,k]
                    count += 1

        # This part calculates the scattered fields
        Iplot = np.zeros((len(Domains[i]),len(Domains[i,0])), dtype=complex)
        FD = np.zeros((M, Ni), dtype=complex)
        for j in range(Ni):
            I = np.zeros(count, dtype=complex)
            V = np.zeros(count, dtype=complex)
            for k in range(len(V)):
                V[k] = E_Inc(x_pos[k], y_pos[k], E0, phi_transmiter[j], k0)
            Zmn = np.zeros((count, count), dtype=complex)
            Zmn_h = np.zeros((count, count), dtype=complex)
            for m in range(count):
                for n in range(count):
                    Zmn[m,n] = Z_mn_func(m, n, x_pos[m], y_pos[m], x_pos[n], y_pos[n], k0, eps_r[m], s) #check eps_r
            for m in range(count):
                Zmn_h[m,m] = Z_mn_h_func(k0, eps_r[m], Z0)
            I = np.matmul(np.linalg.inv(Zmn + Zmn_h), V)
            for k in range(M):
                for m in range(len(I)):
                    FD[k,j] += E_Sct(x_pos[m], y_pos[m], xR[k], yR[k], I[m], k0, landa, s)

        
        # This part plots the scatterer and the corresponding fields
        FDplot = np.zeros((M,2*Ni))
        for a in range(len(FD)):
            FDplot[a, :Ni] = FD[a].real
            FDplot[a, Ni:] = FD[a].imag
        fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
        Dobj = ax1.imshow(Domains[i], cmap="viridis", interpolation='none')
        fig.colorbar(Dobj, ax=ax1)
        fd = ax2.imshow(FDplot, cmap="viridis", interpolation='none')
        fig.colorbar(fd, ax=ax2)
        plt.show()

        Field_Data[i] = FD
        print(i/len(Domains)*100)

    np.save("Field_Data.npy", Field_Data)
else:
    print('The file already exists')
    Field_Data = np.load('Field_Data.npy')

# %% Cell 10 (code)
# here you can observe the calculated results

for b in range(0,10):
    FD = Field_Data[b]
    FDplot = np.zeros((M,2*Ni))
    for a in range(len(FD)):
        FDplot[a, :Ni] = FD[a].real
        FDplot[a, Ni:] = FD[a].imag
    fig, (ax1, ax2) = plt.subplots(figsize=(16, 3), ncols=2)
    Dobj = ax1.imshow(Domains[b], cmap="viridis", interpolation='none')
    fig.colorbar(Dobj, ax=ax1)
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax1.title.set_text('Permittivity')
    fd = ax2.imshow(FDplot, cmap="viridis", interpolation='none')
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.title.set_text('Scattered E')
    fig.colorbar(fd, ax=ax2)
    plt.show()

# %% Cell 11 (code)
# This cell is for creating the Object Domain for the cylindrical scatterer
if os.path.exists("Domains_cy.npy") == False:
    NumDomain = 1000
    Domains_cy = np.ones((NumDomain,28,28), dtype=int)
    for i in range(NumDomain):
        x = np.random.randint(low=4, high=24, size=2).squeeze()
        y = np.random.randint(low=4, high=24, size=2).squeeze()
        for j in range(2):
            Domains_cy[i, x[j]-4, y[j]-2:y[j]+3] = 2*np.ones(5)
            Domains_cy[i, x[j]-3, y[j]-3:y[j]+4] = 2*np.ones(7)
            Domains_cy[i, x[j]-2:x[j]+3, y[j]-4:y[j]+5] = 2*np.ones((5,9))
            Domains_cy[i, x[j]+3, y[j]-3:y[j]+4] = 2*np.ones(7)
            Domains_cy[i, x[j]+4, y[j]-2:y[j]+3] = 2*np.ones(5)
#         plt.imshow(Domains_cy[i])
#         plt.colorbar()
#         plt.show()
    np.save("Domains_cy.npy", Domains_cy)
else:
    print('The file already exists')
    Domains_cy = np.load("Domains_cy.npy")

# %% Cell 12 (code)
# This cell calculates the scattered fields in each reciver using MoM

if os.path.exists("Field_Data_cy.npy") == False:
    phi_transmiter = np.arange(0, 2*np.pi, 2*np.pi/Ni)
    phi_reciever = np.arange(0, 2*np.pi, 2*np.pi/M)
    xR = RD * np.cos(phi_reciever)
    yR = RD * np.sin(phi_reciever)
    Field_Data_cy = np.zeros((len(Domains_cy), M, Ni), dtype=complex)
    for i in range(len(Domains_cy)):
        count = len(Domains_cy[i][Domains_cy[i]>1])
        delxy = Dobjlength / len(Domains_cy[i])
        s = np.power(delxy, 2)
        xmin = -Dobjlength / 2

        x_pos = np.zeros(count)
        y_pos = np.zeros(count)
        eps_r = np.zeros(count)
        count = 0
        
        # This for loop calculates the center position of each fragment
        for j in range(len(Domains_cy[i])):
            for k in range(len(Domains_cy[i,0])):
                if Domains_cy[i,j,k] > 1:
                    x_pos[count] = xmin + (k + 0.5) * delxy
                    y_pos[count] = -xmin - (j + 0.5) * delxy
                    eps_r[count] = Domains_cy[i,j,k]
                    count += 1

        # This part calculates the scattered fields
        Iplot = np.zeros((len(Domains_cy[i]),len(Domains_cy[i,0])), dtype=complex)
        FD = np.zeros((M, Ni), dtype=complex)
        for j in range(Ni):
            I = np.zeros(count, dtype=complex)
            V = np.zeros(count, dtype=complex)
            for k in range(len(V)):
                V[k] = E_Inc(x_pos[k], y_pos[k], E0, phi_transmiter[j], k0)
            Zmn = np.zeros((count, count), dtype=complex)
            Zmn_h = np.zeros((count, count), dtype=complex)
            for m in range(count):
                for n in range(count):
                    Zmn[m,n] = Z_mn_func(m, n, x_pos[m], y_pos[m], x_pos[n], y_pos[n], k0, eps_r[m], s) #check eps_r
            for m in range(count):
                Zmn_h[m,m] = Z_mn_h_func(k0, eps_r[m], Z0)
            I = np.matmul(np.linalg.inv(Zmn + Zmn_h), V)
            for k in range(M):
                for m in range(len(I)):
                    FD[k,j] += E_Sct(x_pos[m], y_pos[m], xR[k], yR[k], I[m], k0, landa, s)

        
        # This part plots the scatterer and the corresponding fields
        FDplot = np.zeros((M,2*Ni))
        for a in range(len(FD)):
            FDplot[a, :Ni] = FD[a].real
            FDplot[a, Ni:] = FD[a].imag
        fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
        Dobj = ax1.imshow(Domains_cy[i], cmap="viridis", interpolation='none')
        fig.colorbar(Dobj, ax=ax1)
        fd = ax2.imshow(FDplot, cmap="viridis", interpolation='none')
        fig.colorbar(fd, ax=ax2)
        plt.show()

        Field_Data_cy[i] = FD
        print(i/len(Domains_cy)*100)

    np.save("Field_Data_cy.npy", Field_Data_cy)
else:
    print('The file already exists')
    Field_Data_cy = np.load('Field_Data_cy.npy')
