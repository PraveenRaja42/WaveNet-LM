import torch 
import torch.nn as nn
import torch.nn.functional as F
import random 


# hyperparameters 
max_steps = 2000
batch_size = 32 
block_size = 8 # context length: how many characters do we take to predict the next one?
n_embd = 24 # the dimensionality of the character embedding vector 
n_hidden = 128 


random.seed(42)
torch.manual_seed(42)
# read in all the words 
words = open(r"names.txt", "r").read().splitlines()
random.shuffle(words)

# character vocabulary and mappings to/from integers 
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)


# build the dataset


def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
#   print(X.shape, Y.shape)
  return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]


class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

# -----------------------------------------------------------------------------------------------
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight] 


# -----------------------------------------------------------------------------------------------
class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]



model = Sequential([
    nn.Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), nn.Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
    FlattenConsecutive(2), nn.Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
    FlattenConsecutive(2), nn.Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
    nn.Linear(n_hidden, vocab_size),]
)

with torch.no_grad():
    model.layers[-1].weight * 0.1 # last layer make less confident

parameters = model.parameters()

for p in parameters:
    p.requires_grad = True



lossi = []

for i in range(max_steps):

    # minibatch construct 
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass 
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward pass 
    for p in parameters: 
        p.grad = None 
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01 # step learning rate decay 
    for p in parameters:
        p.data += -lr * p.grad 

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item()) 

# put layers into eval mode (needed for batchnorm especially)
for layer in model.layers: 
    layer.training = False 

# sample from the model 
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ... 
    while True: 
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix] # shift the context window (crop and append)
        out.append(ix) # track the samples 
        # break if we sample the special '.' token. 
        if ix == 0: 
            break 
    print(''.join(itos[i] for i in out))