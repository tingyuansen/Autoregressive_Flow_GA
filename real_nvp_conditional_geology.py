# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits


#=========================================================================================================
# import training set
temp = np.loadtxt("geology_data.txt")

# velocities (1-10), depths of voronoi cells (1-10),
# and then the outputs are 11 predicted Love wave velocities (one for each of 11 frequencies)
x_tr = temp[:,:20]
y_tr = temp[:,20:]


x_tr = torch.from_numpy(x_tr).type(torch.cuda.FloatTensor)
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)

# standardize
mu_x = x_tr.mean(dim=0)
std_x = x_tr.std(dim=0)
x_tr = (x_tr - mu_x) / std_x


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
        self.s2 = torch.nn.ModuleList([nets2() for _ in range(len(masks))])
        self.t2 = torch.nn.ModuleList([nett2() for _ in range(len(masks))])

    def g(self, z, y):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s1 = self.s[i](x_)
            s2 = self.s2[i](y)
            s = s1*s2*(1 - self.mask[i])
            t1 = self.t[i](x_)
            t2 = self.t2[i](y)
            t = t1*t2*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x, y):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s1 = self.s[i](z_)
            s2 = self.s2[i](y)
            s = s1*s2*(1 - self.mask[i])
            t1 = self.t[i](z_)
            t2 = self.t2[i](y)
            t = t1*t2*(1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x,y):
        z, logp = self.f(x,y)
        return self.prior.log_prob(z) + logp

    def sample(self, z,y):
        x = self.g(z,y)
        return x



#=======================================================================================================
# In [3]:
# define network
device = torch.device("cuda")
num_neurons = 100

# input dimension
dim_in = y_tr.shape[-1]

nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in), nn.Tanh()).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

# conditional layer
dim_cond = x_tr.shape[-1]
nets2 = lambda: nn.Sequential(nn.Linear(dim_cond, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, dim_in)).cuda()
nett2 = lambda: nn.Sequential(nn.Linear(dim_cond, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, dim_in)).cuda()


#=======================================================================================================
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
# load previous models
flow = torch.load('flow_final.pt')
flow.eval();


#=======================================================================================================
# In [4]
# number of epoch and batch size
num_epochs = 2001
batch_size = 1024

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
        loss = -flow.log_prob(y_tr[idx], x_tr[idx]).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # the average loss.
    if e % 10 == 0:
        print('iter %s:' % e, 'loss = %.3f' % loss)

#========================================================================================================
# save models
torch.save(flow, 'flow_final.pt')
