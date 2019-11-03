# In [0]:
# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits


#========================================================================================================
# In [1]:
# import training set
# restore data
hdulist = fits.open('../Catalog_Apogee_Payne.fits.gz')

Teff = hdulist[1].data["Teff"]
Logg = hdulist[1].data["Logg"]
FeH = hdulist[1].data["FeH"]

CFe = hdulist[1].data["CH"] - hdulist[1].data["FeH"]
NFe = hdulist[1].data["NH"] - hdulist[1].data["FeH"]
OFe = hdulist[1].data["OH"] - hdulist[1].data["FeH"]
MgFe = hdulist[1].data["MgH"] - hdulist[1].data["FeH"]

AlFe = hdulist[1].data["AlH"] - hdulist[1].data["FeH"]
SiFe = hdulist[1].data["SiH"] - hdulist[1].data["FeH"]
SFe = hdulist[1].data["SH"] - hdulist[1].data["FeH"]
KFe = hdulist[1].data["KH"] - hdulist[1].data["FeH"]

CaFe = hdulist[1].data["CaH"] - hdulist[1].data["FeH"]
TiFe = hdulist[1].data["TiH"] - hdulist[1].data["FeH"]
CrFe = hdulist[1].data["CrH"] - hdulist[1].data["FeH"]
MnFe = hdulist[1].data["MnH"] - hdulist[1].data["FeH"]

NiFe = hdulist[1].data["NiH"] - hdulist[1].data["FeH"]
CuFe = hdulist[1].data["CuH"] - hdulist[1].data["FeH"]

# make training catalog
y_tr = np.vstack([Teff,Logg,FeH,\
                  CFe, NFe, OFe, MgFe,\
                  AlFe, SiFe, SFe, KFe,\
                  CaFe, TiFe, CrFe, MnFe,\
                  NiFe, CuFe]).T

# convert into torch
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)

# standardize
mu_y = y_tr.mean(dim=0)
std_y = y_tr.std(dim=0)
y_tr = (y_tr - mu_y) / std_y


#=======================================================================================================
# In [2]:
# define normalizing flow
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, z):
        x = self.g(z)
        return x



#=======================================================================================================
# In [3]:
# define network
device = torch.device("cuda")
num_neurons = 300

# input dimension
dim_in = y_tr.shape[-1]

nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in), nn.Tanh()).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

# define mask
num_layers = 10
masks = []
for i in range(num_layers):
    mask_layer = np.random.randint(2,size=(dim_in))
    masks.append(mask_layer)
    masks.append(1-mask_layer)
masks = torch.from_numpy(np.array(masks).astype(np.float32))
masks.to(device)

# set prior
prior = distributions.MultivariateNormal(torch.zeros(dim_in, device='cuda'),\
                                         torch.eye(dim_in, device='cuda'))

# intiate flow
flow = RealNVP(nets, nett, masks, prior)
flow.cuda()


#=======================================================================================================
# In [4]
# number of epoch and batch size
num_epochs = 2001
batch_size = 2048

# break into batches
nsamples = y_tr.shape[0]
nbatches = nsamples // batch_size

# optimizing flow models
optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)

#-------------------------------------------------------------------------------------------------------
# train the network
for e in range(num_epochs):

    # randomly permute the data
    perm = torch.randperm(nsamples)
    perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        loss = -flow.log_prob(y_tr[idx]).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # the average loss.
    if e % 10 == 0:
        print('iter %s:' % e, 'loss = %.3f' % loss)

#========================================================================================================
# save models
torch.save(flow, 'flow_final.pt')

# sample results
z1 = flow.f(y_tr)[0].detach().cpu().numpy()
x1 = y_tr
z2 = np.random.multivariate_normal(np.zeros(dim_in), np.eye(dim_in), x1.shape[0])
x2 = flow.sample(torch.from_numpy(z2).type(torch.cuda.FloatTensor))

# rescale the results
x1 = x1*std_y + mu_y
x2 = x2*std_y + mu_y

# convert back to numpy
x1 = x1.detach().cpu().numpy()
x2 = x2.detach().cpu().numpy()

# save results
np.savez("../real_nvp_results.npz",\
         z1 = z1,\
         z2 = z2,\
         x1 = x1,\
         x2 = x2)
