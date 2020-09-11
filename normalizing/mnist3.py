
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
    transform=transforms.ToTensor()), batch_size=64, shuffle=True)
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
        #self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim*2, z_dim)
            self.fc_var =  nn.Linear(h_dim*2, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim*2, z_dim)
            self.fc_var = nn.Linear(h_dim*2, 1)
        else:
            raise NotImplemented
            
        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim*2)
        #self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))  #x is size 256
        #x = self.activation(self.fc_e1(x))  #x is size 128
        
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
        #x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        x=torch.sigmoid(x)
        
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
    
    
def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.reshape(-1, 784))
    
    """
    x is the mnist image 
    reshape(-1.784) convert the image(28*28=784) into vector 784
    """
    
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def train(model, optimizer):
    
    model.train()
    train_loss=0

    for i, (x_mb, y_mb) in enumerate(train_loader):
            x_mb = x_mb.view(-1, 28 * 28)
            x_mb = x_mb

            optimizer.zero_grad()
            
            # dynamic binarization
            #x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()

            (z_mean,z_var) , (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))

            loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 784)).sum(-1).mean()

            if model.distribution == 'normal':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplemented

            loss = loss_recon + loss_KL

            loss.backward()
            train_loss += loss.item()

            optimizer.step()
    return train_loss

def test(model, optimizer):

    model.eval()
    test_loss=0

    #print_ = defaultdict(list)
    with torch.no_grad():
        for x_mb, y_mb in test_loader:
            x_mb = x_mb.view(-1, 28 * 28)
            x_mb = x_mb
            
            # dynamic binarization
            #x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        
            _, (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))
        
            #print_['recon loss'].append(float(nn.BCEWithLogitsLoss(reduction='none')(x_mb_,x_mb.reshape(-1, 784)).sum(-1).mean().data))
            loss_recon=nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 784)).sum(-1).mean()
        
            if model.distribution == 'normal':
                #print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data))
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                #print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data))
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplemented
        
            loss=loss_recon+loss_KL
            test_loss += loss.item()
            #print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])
            #print_['LL'].append(float(log_likelihood(model, x_mb).data))
    
        #print({k: np.mean(v) for k, v in print_.items()})
    return test_loss


# hidden dimension and dimension of latent space
H_DIM = 128
Z_DIM = 5
epochs=100
early_stopping=10

"""
# normal VAE
modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
optimizerN = optim.Adam(modelN.parameters(), lr=1e-3)

print('##### Normal VAE #####')

N_loss_list = []
N_test_loss_list = []
best_test_loss = float('inf')
# training for 20 epoch
for e in range(epochs):
    train_loss=train(modelN, optimizerN)
    test_loss=test(modelN,optimizerN)
  
    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    N_loss_list.append(train_loss)
    N_test_loss_list.append(test_loss)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    if best_test_loss > test_loss:
         best_test_loss = test_loss
         patience_counter = 1
    else:
         patience_counter += 1

    if patience_counter > early_stopping:
         break

#np.save('N_loss_list.npy', np.array(N_loss_list))
#np.save('N_test_loss_list.npy', np.array(N_test_loss_list))
torch.save(modelN, 'modelN')
#latent_vars(modelN)

# test
#test(modelN, optimizerN)

print()
"""
# hyper-spherical  VAE
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)

print('##### Hyper-spherical VAE #####')

S_loss_list = []
S_test_loss_list = []
best_test_loss = float('inf')
# training for 20 epoch
for e in range(epochs):
    train_loss=train(modelS, optimizerS)
    test_loss=test(modelS,optimizerS)
  
    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    S_loss_list.append(train_loss)
    S_test_loss_list.append(test_loss)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > early_stopping:
        break

#np.save('S_loss_list.npy', np.array(S_loss_list))
#np.save('S_test_loss_list.npy', np.array(S_test_loss_list))
torch.save(modelS,'modelS')
# test
#test(modelS, optimizerS)
"""
N_loss_list = np.load('N_loss_list.npy')
N_test_loss_list = np.load('N_test_loss_list.npy')
S_loss_list = np.load('S_loss_list.npy')
S_test_loss_list = np.load('S_test_loss_list.npy')

plt.plot(N_loss_list,label="N_loss")
plt.plot(N_test_loss_list,label="N_test_loss")
plt.plot(S_loss_list,label="S_loss")
plt.plot(S_test_loss_list,label="S_test_loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig("loss.png")
plt.show()
"""