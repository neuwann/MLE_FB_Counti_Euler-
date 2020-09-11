import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import defaultdict


from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=64)

train_dataset = datasets.MNIST('./data',train=True,download=True,transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data',train=False,download=True,transform=transforms.ToTensor())

class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        
        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
            
        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))  #x is size 256
        x = self.activation(self.fc_e1(x))  #x is size 128
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)            #z_mean is size 5
            z_var = F.softplus(self.fc_var(x))  #z_var is size 5
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)                #z_mean is size 5
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1  #z_var is size 1#
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        
        return x
        
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var)) 
            #p_zの平均と分散を同じ次元の０ベクトルと１ベクトルにしている
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var), (q_z, p_z), z, x_
              
def latent_vars(model):  #output the latent variable z with size 5
    latent_vars=[]
    for i, (x_mb, y_mb) in enumerate(train_loader):
        z_mean, z_var = model.encode(x_mb.reshape(-1, 784))
        q_z, p_z = model.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        latent_vars.append(z)
    np.save('latent_vars.npy', np.array(latent_vars))
    #print(latent_vars)

def label(model,number):
    label_latent_vars=[]
    with torch.no_grad():
        for i, (x_mb, y_mb) in enumerate(train_loader):
            if (y_mb.abs()==number):
                z_mean, z_var = model.encode(x_mb.reshape(-1, 784))
                q_z, p_z = model.reparameterize(z_mean, z_var)
                z = q_z.rsample()
                t=z.cpu().numpy()
                t=t.flatten()
                label_latent_vars.append(t)
    #npz=label_latent_vars.cpu().numpy
    #torch.save(label_latent_vars,"label_latent_vars.csv")
    np.savetxt('label_latent_vars.csv', label_latent_vars) 
    
        


modelS=torch.load("modelS")
#latent_vars(modelS)
label(modelS,1)

#modelN=torch.load("modelN")
#latent_vars(modelN)

#t=np.load("label_latent_vars.npy")
#print(t)