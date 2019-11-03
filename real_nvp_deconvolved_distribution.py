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
temp = np.load("../mock_gaussians.npz")
y_tr = temp["age_noised"] # with noise
y_noise = temp["noise"]

# convert into torch
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)
y_noise = torch.from_numpy(y_noise).type(torch.cuda.FloatTensor)

# standardize
#mu_y = y_tr.mean(dim=0)
#std_y = y_tr.std(dim=0)
#y_tr = (y_tr - mu_y) / std_y


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
num_neurons = 100

# input dimension
dim_in = y_tr.shape[-1]

nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in), nn.Tanh()).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

# define mask
num_layers = 5
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * num_layers).astype(np.float32))

#masks = []
#for i in range(num_layers):
#    mask_layer = np.random.randint(2,size=(dim_in))
#    masks.append(mask_layer)
#    masks.append(1-mask_layer)
#masks = torch.from_numpy(np.array(masks).astype(np.float32))

masks.to(device)

# set prior
prior = distributions.MultivariateNormal(torch.zeros(dim_in, device='cuda'),\
                                         torch.eye(dim_in, device='cuda'))


#=======================================================================================================
# restore models
flow = torch.load("flow_final.pt")
flow.eval()

# disable gradient for the previous flow
for p in flow.parameters():
    p.requires_grad = False


#=======================================================================================================
# another flow for deconvolved distribution
flow2 = RealNVP(nets, nett, masks, prior)
flow2.cuda()

#-------------------------------------------------------------------------------------------------------
# In [4]
# number of epoch and batch size
num_epochs = 201
batch_size = 1028

# break into batches
nsamples = y_tr.shape[0]
nbatches = nsamples // batch_size

# optimizing flow models
optimizer = torch.optim.Adam([p for p in flow2.parameters() if p.requires_grad==True], lr=1e-4)

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

        # map it to the devolved space
        x, logp2 = flow2.f(y_tr[idx])
        #x += torch.randn(size=y_tr[idx].shape).type(torch.cuda.FloatTensor)
        x += y_noise[idx]

        # convolve it back to the observed space
        z, logp = flow.f(x)
        loss = -(flow.prior.log_prob(z) + logp + logp2).mean()

        # gradient descent
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # the average loss.
    if e % 10 == 0:
        print('iter %s:' % e, 'loss = %.3f' % loss)

#-------------------------------------------------------------------------------------------------------
# save models
torch.save(flow2, 'flow2_final.pt')
torch.save(flow, 'flow3_final.pt')


#========================================================================================================
# sample results
z1 = flow.f(y_tr)[0].detach().cpu().numpy()
x1 = y_tr
z2 = np.random.multivariate_normal(np.zeros(dim_in), np.eye(dim_in), x1.shape[0])

# map from the observed space to the normal space
x2, _ = flow2.f(flow.sample(torch.from_numpy(z2).type(torch.cuda.FloatTensor)))

# rescale the results
#x1 = x1*std_y + mu_y
#x2 = x2*std_y + mu_y

# convert back to numpy
x1 = x1.detach().cpu().numpy()
x2 = x2.detach().cpu().numpy()

# save results
np.savez("../real_nvp_deconvolution_results.npz",\
         z1 = z1,\
         z2 = z2,\
         x1 = x1,\
         x2 = x2)
