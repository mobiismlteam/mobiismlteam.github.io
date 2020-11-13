---
layout: article
title: NonParam
mathjax: true
toc : true
tags : UnsupervisedFeatureLearning
---



<h1>Zhirong Wu et al. Unsupervised Feature Learning via Non-Parametric Instance Discrimination (2018)$^{(1)}$ - Implementation</h1>

<h3>Motivation</h3>
<br />
<img src="/assets/images/nonparam_files/np_motivation.png" width="600">
<br />
<h5>Observation of a classification model</h5>
When the image of a leopard is tested on a supervised learning model, the responses are produced as output above.<br />
The highest responses are leopard, and those that look similar to a leopard,<br />
the lowest responses were those that look nothing like a leopard.
<br />
<br />
<h5>Theory Crafting</h5>
Typical discriminative learning methods do not instruct the model to learn the similarity among semantic categories,<br />
but they appear to discover the apparent similarity automatically when learning.
<br />
<br />
<h5>Reforming The Theory Crafting Done Above</h5>
As the semantic annotations are mere classes that are independent of each other by principle,<br />
the apparent similarity must be learned from the visual data themselves.
<br />
<br />
<h5>Question we would like to answer</h5>
Following the last paragraph..<br />
Can we learn to discriminate the individual instances, without any notion of semantic categories?
<br />

<h3>The Result of the Application, and Our Objective</h3>

<br />
<img src="/assets/images/nonparam_files/np_interest.png" width="1000">
<br />

We define the input (in other words, testing data) as to the trained model as query.<br />
Given the query, the 10 closest instances from the training set is extracted above.<br />
<br />
For the successful cases, all top 10 results show the same entity (same category as we say for the classification model) as the query. <br />
We observe that even in the failure cases, there are some features that are similar e.g. color, texture, pattern etc. <br />

<h3>The Pipeline</h3>
<br />
<img src="/assets/images/nonparam_files/np_implementation_pipeline.png" width="1000">
<br />
Above is the training network. <br />

In testing, we calculate the similarity between the query and each element in the memory bank and output top "k" candidates. <br />

<h3>Result of Our Training (sample)</h3>
<br />
<img src="/assets/images/nonparam_files/np_res_norecompute.png" >
<br />

<h3>Implementions</h3>


```python
# Required common libraries and global variables

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os

device = 'cuda'
low_dim = 128
```

<h4>Data Preparation</h4>


```python
from PIL import Image
import torchvision.datasets as datasets

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
```


```python
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```


```python
trainset = CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
ndata = trainset.__len__()

testset = CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
```

    Files already downloaded and verified
    Files already downloaded and verified


<h4>Backbone model (ResNet18)</h4>


```python

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, low_dim)
        self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) #
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.l2norm(out)
        return out


def ResNet18(low_dim=128):
    return ResNet(BasicBlock, [2,2,2,2], low_dim)

def ResNet34(low_dim=128):
    return ResNet(BasicBlock, [3,4,6,3], low_dim)

def ResNet50(low_dim=128):
    return ResNet(Bottleneck, [3,4,6,3], low_dim)

def ResNet101(low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], low_dim)

def ResNet152(low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], low_dim)
```


```python
import torch.backends.cudnn as cudnn

net = ResNet18(low_dim)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
```

    /home/gorilla/.virtualenvs/jylee_py27/local/lib/python2.7/site-packages/torch/nn/parallel/data_parallel.py:24: UserWarning: 
        There is an imbalance between your GPUs. You may want to exclude GPU 1 which
        has less than 75% of the memory or cores of GPU 0. You can do so by setting
        the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
        environment variable.
      warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))



```python
print(net)
```

    DataParallel(
      (module): ResNet(
        (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (layer1): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential()
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential()
          )
        )
        (layer2): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential()
          )
        )
        (layer3): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential()
          )
        )
        (layer4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (shortcut): Sequential()
          )
        )
        (linear): Linear(in_features=512, out_features=128, bias=True)
        (l2norm): Normalize()
      )
    )


<h5>Net architecture (to be edited with cleaner text)</h5>
<img src="/assets/images/nonparam_files/np_resnet_paintedit.png">
<br />

<h4>Non-parametric softmax classifier, with noise-contrastive estimation - setting memory bank</h4>


```python
class AliasMethod(object):
    '''
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self): 
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj
```


```python
class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)

        # sample positives & negatives
        idx.select(1,0).copy_(y.data)

        # sample corresponding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K+1, inputSize)

        # inner product
        out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))
        out.div_(T).exp_() # batchSize * self.K+1
        x.data.resize_(batchSize, inputSize)

        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K+1)

        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        gradOutput.data.resize_(batchSize, 1, K+1)
        
        # gradient of linear
        gradInput = torch.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None
```


```python
class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        
        self.K = K

        self.register_buffer('params',torch.tensor([K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
 
    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K+1)).view(batchSize, -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out
```


```python
nce_k = 4096
nce_t = 0.07
nce_m = 0.5
```


```python
if nce_k > 0:
    lemniscate = NCEAverage(low_dim, ndata, nce_k, nce_t, nce_m)
else:
    print('nce_k value must be above zero.')
```

<h4>Non-parametric softmax classifier, with noise-contrastive estimation - define objective function</h4>


```python
eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1)-1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)
        
        # eq 6.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        Pmt = x.select(1,0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 6.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
     
        # equation 7
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        
        return loss
```


```python
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    print('nce_k = 0')
```

<h4>Begin training the model</h4>


```python
net.to(device)
lemniscate.to(device)
criterion.to(device)
```




    NCECriterion()




```python
import torch.optim as optim

starting_lr = 0.03

optimizer = optim.SGD(net.parameters(), lr=starting_lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = starting_lr
    if epoch >= 80:
        lr = starting_lr * (0.1 ** ((epoch-80) // 40))
    print('Learning rate of this epoch: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```


```python
# utility class that updates value and average value for display when training the model

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```


```python
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        # inputs - pictures - batch x data_dim - 128x3x32x32
        # targets - class label 0~9 - 128
        # indexes - index of data among the dataset - 128
        optimizer.zero_grad()

        features = net(inputs)
        # features 128x128 - batch x low_dim
        outputs = lemniscate(features, indexes)
        # outputs 128x4097 - batch x nce_k
        
        loss = criterion(outputs, indexes)
        #print(loss) # scalar
        loss.backward()
        optimizer.step()

        # for display
        train_loss.update(loss.item(), inputs.size(0))
        
        if batch_idx >= len(trainloader)-1:
            time_taken = time.time() - end
            print('Epoch {} complete. [batchs trained: {}]\n'
                  'Time taken: {:.3f}, '
                  'Avg Loss: {train_loss.avg:.4f}'.format(
                  epoch, len(trainloader), time_taken, train_loss=train_loss))
```


```python
total_epochs = 200

for epoch in range(total_epochs):
    train(epoch)
```

    
    Epoch: 0
    Learning rate of this epoch: 0.03
    normalization constant Z is set to 110396.4
    Epoch: 0 [batchs trained: 391]
    Time taken: 19.926, Avg Loss: 72.9217
    
    Epoch: 1
    Learning rate of this epoch: 0.03
    Epoch: 1 [batchs trained: 391]
    Time taken: 16.889, Avg Loss: 80.2561
    
    Epoch: 2
    Learning rate of this epoch: 0.03
    Epoch: 2 [batchs trained: 391]
    Time taken: 17.075, Avg Loss: 79.2967
    
    Epoch: 3
    Learning rate of this epoch: 0.03
    Epoch: 3 [batchs trained: 391]
    Time taken: 17.493, Avg Loss: 87.6499
    
    Epoch: 4
    Learning rate of this epoch: 0.03
    Epoch: 4 [batchs trained: 391]
    Time taken: 17.252, Avg Loss: 76.9851
    
    Epoch: 5
    Learning rate of this epoch: 0.03
    Epoch: 5 [batchs trained: 391]
    Time taken: 17.469, Avg Loss: 63.8815
    
    Epoch: 6
    Learning rate of this epoch: 0.03
    Epoch: 6 [batchs trained: 391]
    Time taken: 17.478, Avg Loss: 70.0318
    
    Epoch: 7
    Learning rate of this epoch: 0.03
    Epoch: 7 [batchs trained: 391]
    Time taken: 17.527, Avg Loss: 66.7217
    
    Epoch: 8
    Learning rate of this epoch: 0.03
    Epoch: 8 [batchs trained: 391]
    Time taken: 17.212, Avg Loss: 69.3957
    
    Epoch: 9
    Learning rate of this epoch: 0.03
    Epoch: 9 [batchs trained: 391]
    Time taken: 17.977, Avg Loss: 44.4620
    
    Epoch: 10
    Learning rate of this epoch: 0.03
    Epoch: 10 [batchs trained: 391]
    Time taken: 18.283, Avg Loss: 42.1522
    
    Epoch: 11
    Learning rate of this epoch: 0.03
    Epoch: 11 [batchs trained: 391]
    Time taken: 17.880, Avg Loss: 36.1025
    
    Epoch: 12
    Learning rate of this epoch: 0.03
    Epoch: 12 [batchs trained: 391]
    Time taken: 17.977, Avg Loss: 35.0794
    
    Epoch: 13
    Learning rate of this epoch: 0.03
    Epoch: 13 [batchs trained: 391]
    Time taken: 17.936, Avg Loss: 30.6100
    
    Epoch: 14
    Learning rate of this epoch: 0.03
    Epoch: 14 [batchs trained: 391]
    Time taken: 18.606, Avg Loss: 28.3440
    
    Epoch: 15
    Learning rate of this epoch: 0.03
    Epoch: 15 [batchs trained: 391]
    Time taken: 18.039, Avg Loss: 22.4906
    
    Epoch: 16
    Learning rate of this epoch: 0.03
    Epoch: 16 [batchs trained: 391]
    Time taken: 18.022, Avg Loss: 18.0116
    
    Epoch: 17
    Learning rate of this epoch: 0.03
    Epoch: 17 [batchs trained: 391]
    Time taken: 17.894, Avg Loss: 15.9840
    
    Epoch: 18
    Learning rate of this epoch: 0.03
    Epoch: 18 [batchs trained: 391]
    Time taken: 18.053, Avg Loss: 14.8233
    
    Epoch: 19
    Learning rate of this epoch: 0.03
    Epoch: 19 [batchs trained: 391]
    Time taken: 17.807, Avg Loss: 13.7734
    
    Epoch: 20
    Learning rate of this epoch: 0.03
    Epoch: 20 [batchs trained: 391]
    Time taken: 17.528, Avg Loss: 12.8939
    
    Epoch: 21
    Learning rate of this epoch: 0.03
    Epoch: 21 [batchs trained: 391]
    Time taken: 17.536, Avg Loss: 12.1606
    
    Epoch: 22
    Learning rate of this epoch: 0.03
    Epoch: 22 [batchs trained: 391]
    Time taken: 17.712, Avg Loss: 11.5684
    
    Epoch: 23
    Learning rate of this epoch: 0.03
    Epoch: 23 [batchs trained: 391]
    Time taken: 17.704, Avg Loss: 11.1067
    
    Epoch: 24
    Learning rate of this epoch: 0.03
    Epoch: 24 [batchs trained: 391]
    Time taken: 17.774, Avg Loss: 10.5617
    
    Epoch: 25
    Learning rate of this epoch: 0.03
    Epoch: 25 [batchs trained: 391]
    Time taken: 18.162, Avg Loss: 10.0679
    
    Epoch: 26
    Learning rate of this epoch: 0.03
    Epoch: 26 [batchs trained: 391]
    Time taken: 18.100, Avg Loss: 9.6331
    
    Epoch: 27
    Learning rate of this epoch: 0.03
    Epoch: 27 [batchs trained: 391]
    Time taken: 17.997, Avg Loss: 9.2453
    
    Epoch: 28
    Learning rate of this epoch: 0.03
    Epoch: 28 [batchs trained: 391]
    Time taken: 18.171, Avg Loss: 8.8674
    
    Epoch: 29
    Learning rate of this epoch: 0.03
    Epoch: 29 [batchs trained: 391]
    Time taken: 18.063, Avg Loss: 8.4246
    
    Epoch: 30
    Learning rate of this epoch: 0.03
    Epoch: 30 [batchs trained: 391]
    Time taken: 18.160, Avg Loss: 7.9965
    
    Epoch: 31
    Learning rate of this epoch: 0.03
    Epoch: 31 [batchs trained: 391]
    Time taken: 17.542, Avg Loss: 7.6534
    
    Epoch: 32
    Learning rate of this epoch: 0.03
    Epoch: 32 [batchs trained: 391]
    Time taken: 18.187, Avg Loss: 7.2984
    
    Epoch: 33
    Learning rate of this epoch: 0.03
    Epoch: 33 [batchs trained: 391]
    Time taken: 16.979, Avg Loss: 6.9570
    
    Epoch: 34
    Learning rate of this epoch: 0.03
    Epoch: 34 [batchs trained: 391]
    Time taken: 17.908, Avg Loss: 6.6972
    
    Epoch: 35
    Learning rate of this epoch: 0.03
    Epoch: 35 [batchs trained: 391]
    Time taken: 17.635, Avg Loss: 6.4510
    
    Epoch: 36
    Learning rate of this epoch: 0.03
    Epoch: 36 [batchs trained: 391]
    Time taken: 18.160, Avg Loss: 6.1835
    
    Epoch: 37
    Learning rate of this epoch: 0.03
    Epoch: 37 [batchs trained: 391]
    Time taken: 17.899, Avg Loss: 5.9612
    
    Epoch: 38
    Learning rate of this epoch: 0.03
    Epoch: 38 [batchs trained: 391]
    Time taken: 18.255, Avg Loss: 5.7754
    
    Epoch: 39
    Learning rate of this epoch: 0.03
    Epoch: 39 [batchs trained: 391]
    Time taken: 17.021, Avg Loss: 5.6039
    
    Epoch: 40
    Learning rate of this epoch: 0.03
    Epoch: 40 [batchs trained: 391]
    Time taken: 18.131, Avg Loss: 5.3834
    
    Epoch: 41
    Learning rate of this epoch: 0.03
    Epoch: 41 [batchs trained: 391]
    Time taken: 18.039, Avg Loss: 5.2324
    
    Epoch: 42
    Learning rate of this epoch: 0.03
    Epoch: 42 [batchs trained: 391]
    Time taken: 18.197, Avg Loss: 5.0908
    
    Epoch: 43
    Learning rate of this epoch: 0.03
    Epoch: 43 [batchs trained: 391]
    Time taken: 18.148, Avg Loss: 4.9767
    
    Epoch: 44
    Learning rate of this epoch: 0.03
    Epoch: 44 [batchs trained: 391]
    Time taken: 18.007, Avg Loss: 4.8384
    
    Epoch: 45
    Learning rate of this epoch: 0.03
    Epoch: 45 [batchs trained: 391]
    Time taken: 18.055, Avg Loss: 4.7117
    
    Epoch: 46
    Learning rate of this epoch: 0.03
    Epoch: 46 [batchs trained: 391]
    Time taken: 17.463, Avg Loss: 4.6121
    
    Epoch: 47
    Learning rate of this epoch: 0.03
    Epoch: 47 [batchs trained: 391]
    Time taken: 18.044, Avg Loss: 4.5110
    
    Epoch: 48
    Learning rate of this epoch: 0.03
    Epoch: 48 [batchs trained: 391]
    Time taken: 17.906, Avg Loss: 4.4096
    
    Epoch: 49
    Learning rate of this epoch: 0.03
    Epoch: 49 [batchs trained: 391]
    Time taken: 17.950, Avg Loss: 4.3323
    
    Epoch: 50
    Learning rate of this epoch: 0.03
    Epoch: 50 [batchs trained: 391]
    Time taken: 18.172, Avg Loss: 4.2437
    
    Epoch: 51
    Learning rate of this epoch: 0.03
    Epoch: 51 [batchs trained: 391]
    Time taken: 18.234, Avg Loss: 4.1584
    
    Epoch: 52
    Learning rate of this epoch: 0.03
    Epoch: 52 [batchs trained: 391]
    Time taken: 18.163, Avg Loss: 4.0747
    
    Epoch: 53
    Learning rate of this epoch: 0.03
    Epoch: 53 [batchs trained: 391]
    Time taken: 18.292, Avg Loss: 4.0452
    
    Epoch: 54
    Learning rate of this epoch: 0.03
    Epoch: 54 [batchs trained: 391]
    Time taken: 18.154, Avg Loss: 3.9619
    
    Epoch: 55
    Learning rate of this epoch: 0.03
    Epoch: 55 [batchs trained: 391]
    Time taken: 18.247, Avg Loss: 3.9149
    
    Epoch: 56
    Learning rate of this epoch: 0.03
    Epoch: 56 [batchs trained: 391]
    Time taken: 18.291, Avg Loss: 3.8749
    
    Epoch: 57
    Learning rate of this epoch: 0.03
    Epoch: 57 [batchs trained: 391]
    Time taken: 17.890, Avg Loss: 3.8199
    
    Epoch: 58
    Learning rate of this epoch: 0.03
    Epoch: 58 [batchs trained: 391]
    Time taken: 17.870, Avg Loss: 3.7764
    
    Epoch: 59
    Learning rate of this epoch: 0.03
    Epoch: 59 [batchs trained: 391]
    Time taken: 18.092, Avg Loss: 3.7310
    
    Epoch: 60
    Learning rate of this epoch: 0.03
    Epoch: 60 [batchs trained: 391]
    Time taken: 18.262, Avg Loss: 3.6831
    
    Epoch: 61
    Learning rate of this epoch: 0.03
    Epoch: 61 [batchs trained: 391]
    Time taken: 18.275, Avg Loss: 3.6426
    
    Epoch: 62
    Learning rate of this epoch: 0.03
    Epoch: 62 [batchs trained: 391]
    Time taken: 18.226, Avg Loss: 3.6091
    
    Epoch: 63
    Learning rate of this epoch: 0.03
    Epoch: 63 [batchs trained: 391]
    Time taken: 17.766, Avg Loss: 3.5579
    
    Epoch: 64
    Learning rate of this epoch: 0.03
    Epoch: 64 [batchs trained: 391]
    Time taken: 16.874, Avg Loss: 3.5245
    
    Epoch: 65
    Learning rate of this epoch: 0.03
    Epoch: 65 [batchs trained: 391]
    Time taken: 17.577, Avg Loss: 3.4789
    
    Epoch: 66
    Learning rate of this epoch: 0.03
    Epoch: 66 [batchs trained: 391]
    Time taken: 18.262, Avg Loss: 3.4473
    
    Epoch: 67
    Learning rate of this epoch: 0.03
    Epoch: 67 [batchs trained: 391]
    Time taken: 18.321, Avg Loss: 3.4230
    
    Epoch: 68
    Learning rate of this epoch: 0.03
    Epoch: 68 [batchs trained: 391]
    Time taken: 18.236, Avg Loss: 3.3793
    
    Epoch: 69
    Learning rate of this epoch: 0.03
    Epoch: 69 [batchs trained: 391]
    Time taken: 18.315, Avg Loss: 3.3551
    
    Epoch: 70
    Learning rate of this epoch: 0.03
    Epoch: 70 [batchs trained: 391]
    Time taken: 18.312, Avg Loss: 3.3129
    
    Epoch: 71
    Learning rate of this epoch: 0.03
    Epoch: 71 [batchs trained: 391]
    Time taken: 18.038, Avg Loss: 3.2966
    
    Epoch: 72
    Learning rate of this epoch: 0.03
    Epoch: 72 [batchs trained: 391]
    Time taken: 18.123, Avg Loss: 3.2775
    
    Epoch: 73
    Learning rate of this epoch: 0.03
    Epoch: 73 [batchs trained: 391]
    Time taken: 18.121, Avg Loss: 3.2457
    
    Epoch: 74
    Learning rate of this epoch: 0.03
    Epoch: 74 [batchs trained: 391]
    Time taken: 18.219, Avg Loss: 3.2242
    
    Epoch: 75
    Learning rate of this epoch: 0.03
    Epoch: 75 [batchs trained: 391]
    Time taken: 18.178, Avg Loss: 3.1903
    
    Epoch: 76
    Learning rate of this epoch: 0.03
    Epoch: 76 [batchs trained: 391]
    Time taken: 17.131, Avg Loss: 3.1654
    
    Epoch: 77
    Learning rate of this epoch: 0.03
    Epoch: 77 [batchs trained: 391]
    Time taken: 18.134, Avg Loss: 3.1473
    
    Epoch: 78
    Learning rate of this epoch: 0.03
    Epoch: 78 [batchs trained: 391]
    Time taken: 18.274, Avg Loss: 3.1209
    
    Epoch: 79
    Learning rate of this epoch: 0.03
    Epoch: 79 [batchs trained: 391]
    Time taken: 18.206, Avg Loss: 3.1036
    
    Epoch: 80
    Learning rate of this epoch: 0.03
    Epoch: 80 [batchs trained: 391]
    Time taken: 18.269, Avg Loss: 3.0771
    
    Epoch: 81
    Learning rate of this epoch: 0.03
    Epoch: 81 [batchs trained: 391]
    Time taken: 18.283, Avg Loss: 3.0691
    
    Epoch: 82
    Learning rate of this epoch: 0.03
    Epoch: 82 [batchs trained: 391]
    Time taken: 18.195, Avg Loss: 3.0454
    
    Epoch: 83
    Learning rate of this epoch: 0.03
    Epoch: 83 [batchs trained: 391]
    Time taken: 18.343, Avg Loss: 3.0176
    
    Epoch: 84
    Learning rate of this epoch: 0.03
    Epoch: 84 [batchs trained: 391]
    Time taken: 18.297, Avg Loss: 3.0131
    
    Epoch: 85
    Learning rate of this epoch: 0.03
    Epoch: 85 [batchs trained: 391]
    Time taken: 18.330, Avg Loss: 2.9811
    
    Epoch: 86
    Learning rate of this epoch: 0.03
    Epoch: 86 [batchs trained: 391]
    Time taken: 17.934, Avg Loss: 2.9645
    
    Epoch: 87
    Learning rate of this epoch: 0.03
    Epoch: 87 [batchs trained: 391]
    Time taken: 18.227, Avg Loss: 2.9531
    
    Epoch: 88
    Learning rate of this epoch: 0.03
    Epoch: 88 [batchs trained: 391]
    Time taken: 18.143, Avg Loss: 2.9437
    
    Epoch: 89
    Learning rate of this epoch: 0.03
    Epoch: 89 [batchs trained: 391]
    Time taken: 18.230, Avg Loss: 2.9247
    
    Epoch: 90
    Learning rate of this epoch: 0.03
    Epoch: 90 [batchs trained: 391]
    Time taken: 18.249, Avg Loss: 2.9014
    
    Epoch: 91
    Learning rate of this epoch: 0.03
    Epoch: 91 [batchs trained: 391]
    Time taken: 18.305, Avg Loss: 2.8805
    
    Epoch: 92
    Learning rate of this epoch: 0.03
    Epoch: 92 [batchs trained: 391]
    Time taken: 18.232, Avg Loss: 2.8687
    
    Epoch: 93
    Learning rate of this epoch: 0.03
    Epoch: 93 [batchs trained: 391]
    Time taken: 18.198, Avg Loss: 2.8507
    
    Epoch: 94
    Learning rate of this epoch: 0.03
    Epoch: 94 [batchs trained: 391]
    Time taken: 18.254, Avg Loss: 2.8385
    
    Epoch: 95
    Learning rate of this epoch: 0.03
    Epoch: 95 [batchs trained: 391]
    Time taken: 18.252, Avg Loss: 2.8165
    
    Epoch: 96
    Learning rate of this epoch: 0.03
    Epoch: 96 [batchs trained: 391]
    Time taken: 18.294, Avg Loss: 2.8014
    
    Epoch: 97
    Learning rate of this epoch: 0.03
    Epoch: 97 [batchs trained: 391]
    Time taken: 18.047, Avg Loss: 2.7886
    
    Epoch: 98
    Learning rate of this epoch: 0.03
    Epoch: 98 [batchs trained: 391]
    Time taken: 18.153, Avg Loss: 2.7871
    
    Epoch: 99
    Learning rate of this epoch: 0.03
    Epoch: 99 [batchs trained: 391]
    Time taken: 18.091, Avg Loss: 2.7575
    
    Epoch: 100
    Learning rate of this epoch: 0.03
    Epoch: 100 [batchs trained: 391]
    Time taken: 18.243, Avg Loss: 2.7446
    
    Epoch: 101
    Learning rate of this epoch: 0.03
    Epoch: 101 [batchs trained: 391]
    Time taken: 18.259, Avg Loss: 2.7322
    
    Epoch: 102
    Learning rate of this epoch: 0.03
    Epoch: 102 [batchs trained: 391]
    Time taken: 18.200, Avg Loss: 2.7131
    
    Epoch: 103
    Learning rate of this epoch: 0.03
    Epoch: 103 [batchs trained: 391]
    Time taken: 18.306, Avg Loss: 2.6961
    
    Epoch: 104
    Learning rate of this epoch: 0.03
    Epoch: 104 [batchs trained: 391]
    Time taken: 18.295, Avg Loss: 2.6888
    
    Epoch: 105
    Learning rate of this epoch: 0.03
    Epoch: 105 [batchs trained: 391]
    Time taken: 18.324, Avg Loss: 2.6730
    
    Epoch: 106
    Learning rate of this epoch: 0.03
    Epoch: 106 [batchs trained: 391]
    Time taken: 18.269, Avg Loss: 2.6686
    
    Epoch: 107
    Learning rate of this epoch: 0.03
    Epoch: 107 [batchs trained: 391]
    Time taken: 17.160, Avg Loss: 2.6605
    
    Epoch: 108
    Learning rate of this epoch: 0.03
    Epoch: 108 [batchs trained: 391]
    Time taken: 17.975, Avg Loss: 2.6391
    
    Epoch: 109
    Learning rate of this epoch: 0.03
    Epoch: 109 [batchs trained: 391]
    Time taken: 18.367, Avg Loss: 2.6354
    
    Epoch: 110
    Learning rate of this epoch: 0.03
    Epoch: 110 [batchs trained: 391]
    Time taken: 18.238, Avg Loss: 2.6142
    
    Epoch: 111
    Learning rate of this epoch: 0.03
    Epoch: 111 [batchs trained: 391]
    Time taken: 18.318, Avg Loss: 2.6060
    
    Epoch: 112
    Learning rate of this epoch: 0.03
    Epoch: 112 [batchs trained: 391]
    Time taken: 18.204, Avg Loss: 2.5989
    
    Epoch: 113
    Learning rate of this epoch: 0.03
    Epoch: 113 [batchs trained: 391]
    Time taken: 18.205, Avg Loss: 2.5947
    
    Epoch: 114
    Learning rate of this epoch: 0.03
    Epoch: 114 [batchs trained: 391]
    Time taken: 18.212, Avg Loss: 2.5945
    
    Epoch: 115
    Learning rate of this epoch: 0.03
    Epoch: 115 [batchs trained: 391]
    Time taken: 18.307, Avg Loss: 2.5637
    
    Epoch: 116
    Learning rate of this epoch: 0.03
    Epoch: 116 [batchs trained: 391]
    Time taken: 18.088, Avg Loss: 2.5609
    
    Epoch: 117
    Learning rate of this epoch: 0.03
    Epoch: 117 [batchs trained: 391]
    Time taken: 18.030, Avg Loss: 2.5489
    
    Epoch: 118
    Learning rate of this epoch: 0.03
    Epoch: 118 [batchs trained: 391]
    Time taken: 17.986, Avg Loss: 2.5429
    
    Epoch: 119
    Learning rate of this epoch: 0.03
    Epoch: 119 [batchs trained: 391]
    Time taken: 18.150, Avg Loss: 2.5474
    
    Epoch: 120
    Learning rate of this epoch: 0.003
    Epoch: 120 [batchs trained: 391]
    Time taken: 17.722, Avg Loss: 2.5276
    
    Epoch: 121
    Learning rate of this epoch: 0.003
    Epoch: 121 [batchs trained: 391]
    Time taken: 18.276, Avg Loss: 2.4534
    
    Epoch: 122
    Learning rate of this epoch: 0.003
    Epoch: 122 [batchs trained: 391]
    Time taken: 17.502, Avg Loss: 2.4343
    
    Epoch: 123
    Learning rate of this epoch: 0.003
    Epoch: 123 [batchs trained: 391]
    Time taken: 17.231, Avg Loss: 2.4155
    
    Epoch: 124
    Learning rate of this epoch: 0.003
    Epoch: 124 [batchs trained: 391]
    Time taken: 18.264, Avg Loss: 2.3928
    
    Epoch: 125
    Learning rate of this epoch: 0.003
    Epoch: 125 [batchs trained: 391]
    Time taken: 18.225, Avg Loss: 2.3812
    
    Epoch: 126
    Learning rate of this epoch: 0.003
    Epoch: 126 [batchs trained: 391]
    Time taken: 17.817, Avg Loss: 2.3754
    
    Epoch: 127
    Learning rate of this epoch: 0.003
    Epoch: 127 [batchs trained: 391]
    Time taken: 18.362, Avg Loss: 2.3510
    
    Epoch: 128
    Learning rate of this epoch: 0.003
    Epoch: 128 [batchs trained: 391]
    Time taken: 18.293, Avg Loss: 2.3515
    
    Epoch: 129
    Learning rate of this epoch: 0.003
    Epoch: 129 [batchs trained: 391]
    Time taken: 18.346, Avg Loss: 2.3386
    
    Epoch: 130
    Learning rate of this epoch: 0.003
    Epoch: 130 [batchs trained: 391]
    Time taken: 18.283, Avg Loss: 2.3320
    
    Epoch: 131
    Learning rate of this epoch: 0.003
    Epoch: 131 [batchs trained: 391]
    Time taken: 18.089, Avg Loss: 2.3310
    
    Epoch: 132
    Learning rate of this epoch: 0.003
    Epoch: 132 [batchs trained: 391]
    Time taken: 17.780, Avg Loss: 2.3114
    
    Epoch: 133
    Learning rate of this epoch: 0.003
    Epoch: 133 [batchs trained: 391]
    Time taken: 18.262, Avg Loss: 2.3108
    
    Epoch: 134
    Learning rate of this epoch: 0.003
    Epoch: 134 [batchs trained: 391]
    Time taken: 18.267, Avg Loss: 2.2995
    
    Epoch: 135
    Learning rate of this epoch: 0.003
    Epoch: 135 [batchs trained: 391]
    Time taken: 18.178, Avg Loss: 2.2962
    
    Epoch: 136
    Learning rate of this epoch: 0.003
    Epoch: 136 [batchs trained: 391]
    Time taken: 18.164, Avg Loss: 2.2834
    
    Epoch: 137
    Learning rate of this epoch: 0.003
    Epoch: 137 [batchs trained: 391]
    Time taken: 18.291, Avg Loss: 2.2676
    
    Epoch: 138
    Learning rate of this epoch: 0.003
    Epoch: 138 [batchs trained: 391]
    Time taken: 18.183, Avg Loss: 2.2708
    
    Epoch: 139
    Learning rate of this epoch: 0.003
    Epoch: 139 [batchs trained: 391]
    Time taken: 17.633, Avg Loss: 2.2690
    
    Epoch: 140
    Learning rate of this epoch: 0.003
    Epoch: 140 [batchs trained: 391]
    Time taken: 17.747, Avg Loss: 2.2591
    
    Epoch: 141
    Learning rate of this epoch: 0.003
    Epoch: 141 [batchs trained: 391]
    Time taken: 18.220, Avg Loss: 2.2470
    
    Epoch: 142
    Learning rate of this epoch: 0.003
    Epoch: 142 [batchs trained: 391]
    Time taken: 18.213, Avg Loss: 2.2482
    
    Epoch: 143
    Learning rate of this epoch: 0.003
    Epoch: 143 [batchs trained: 391]
    Time taken: 17.105, Avg Loss: 2.2407
    
    Epoch: 144
    Learning rate of this epoch: 0.003
    Epoch: 144 [batchs trained: 391]
    Time taken: 18.054, Avg Loss: 2.2297
    
    Epoch: 145
    Learning rate of this epoch: 0.003
    Epoch: 145 [batchs trained: 391]
    Time taken: 18.226, Avg Loss: 2.2184
    
    Epoch: 146
    Learning rate of this epoch: 0.003
    Epoch: 146 [batchs trained: 391]
    Time taken: 18.220, Avg Loss: 2.2232
    
    Epoch: 147
    Learning rate of this epoch: 0.003
    Epoch: 147 [batchs trained: 391]
    Time taken: 18.208, Avg Loss: 2.2131
    
    Epoch: 148
    Learning rate of this epoch: 0.003
    Epoch: 148 [batchs trained: 391]
    Time taken: 18.313, Avg Loss: 2.2092
    
    Epoch: 149
    Learning rate of this epoch: 0.003
    Epoch: 149 [batchs trained: 391]
    Time taken: 18.274, Avg Loss: 2.2076
    
    Epoch: 150
    Learning rate of this epoch: 0.003
    Epoch: 150 [batchs trained: 391]
    Time taken: 18.051, Avg Loss: 2.2060
    
    Epoch: 151
    Learning rate of this epoch: 0.003
    Epoch: 151 [batchs trained: 391]
    Time taken: 18.028, Avg Loss: 2.1993
    
    Epoch: 152
    Learning rate of this epoch: 0.003
    Epoch: 152 [batchs trained: 391]
    Time taken: 18.270, Avg Loss: 2.1864
    
    Epoch: 153
    Learning rate of this epoch: 0.003
    Epoch: 153 [batchs trained: 391]
    Time taken: 17.739, Avg Loss: 2.1861
    
    Epoch: 154
    Learning rate of this epoch: 0.003
    Epoch: 154 [batchs trained: 391]
    Time taken: 18.010, Avg Loss: 2.1826
    
    Epoch: 155
    Learning rate of this epoch: 0.003
    Epoch: 155 [batchs trained: 391]
    Time taken: 18.209, Avg Loss: 2.1727
    
    Epoch: 156
    Learning rate of this epoch: 0.003
    Epoch: 156 [batchs trained: 391]
    Time taken: 17.289, Avg Loss: 2.1763
    
    Epoch: 157
    Learning rate of this epoch: 0.003
    Epoch: 157 [batchs trained: 391]
    Time taken: 18.307, Avg Loss: 2.1666
    
    Epoch: 158
    Learning rate of this epoch: 0.003
    Epoch: 158 [batchs trained: 391]
    Time taken: 18.307, Avg Loss: 2.1589
    
    Epoch: 159
    Learning rate of this epoch: 0.003
    Epoch: 159 [batchs trained: 391]
    Time taken: 18.310, Avg Loss: 2.1611
    
    Epoch: 160
    Learning rate of this epoch: 0.0003
    Epoch: 160 [batchs trained: 391]
    Time taken: 17.817, Avg Loss: 2.2168
    
    Epoch: 161
    Learning rate of this epoch: 0.0003
    Epoch: 161 [batchs trained: 391]
    Time taken: 17.048, Avg Loss: 2.2176
    
    Epoch: 162
    Learning rate of this epoch: 0.0003
    Epoch: 162 [batchs trained: 391]
    Time taken: 18.260, Avg Loss: 2.1909
    
    Epoch: 163
    Learning rate of this epoch: 0.0003
    Epoch: 163 [batchs trained: 391]
    Time taken: 18.323, Avg Loss: 2.1916
    
    Epoch: 164
    Learning rate of this epoch: 0.0003
    Epoch: 164 [batchs trained: 391]
    Time taken: 17.246, Avg Loss: 2.1856
    
    Epoch: 165
    Learning rate of this epoch: 0.0003
    Epoch: 165 [batchs trained: 391]
    Time taken: 17.863, Avg Loss: 2.1723
    
    Epoch: 166
    Learning rate of this epoch: 0.0003
    Epoch: 166 [batchs trained: 391]
    Time taken: 17.073, Avg Loss: 2.1670
    
    Epoch: 167
    Learning rate of this epoch: 0.0003
    Epoch: 167 [batchs trained: 391]
    Time taken: 18.252, Avg Loss: 2.1697
    
    Epoch: 168
    Learning rate of this epoch: 0.0003
    Epoch: 168 [batchs trained: 391]
    Time taken: 18.229, Avg Loss: 2.1596
    
    Epoch: 169
    Learning rate of this epoch: 0.0003
    Epoch: 169 [batchs trained: 391]
    Time taken: 18.363, Avg Loss: 2.1796
    
    Epoch: 170
    Learning rate of this epoch: 0.0003
    Epoch: 170 [batchs trained: 391]
    Time taken: 18.219, Avg Loss: 2.1633
    
    Epoch: 171
    Learning rate of this epoch: 0.0003
    Epoch: 171 [batchs trained: 391]
    Time taken: 18.178, Avg Loss: 2.1629
    
    Epoch: 172
    Learning rate of this epoch: 0.0003
    Epoch: 172 [batchs trained: 391]
    Time taken: 18.267, Avg Loss: 2.1570
    
    Epoch: 173
    Learning rate of this epoch: 0.0003
    Epoch: 173 [batchs trained: 391]
    Time taken: 18.099, Avg Loss: 2.1514
    
    Epoch: 174
    Learning rate of this epoch: 0.0003
    Epoch: 174 [batchs trained: 391]
    Time taken: 17.661, Avg Loss: 2.1537
    
    Epoch: 175
    Learning rate of this epoch: 0.0003
    Epoch: 175 [batchs trained: 391]
    Time taken: 17.711, Avg Loss: 2.1534
    
    Epoch: 176
    Learning rate of this epoch: 0.0003
    Epoch: 176 [batchs trained: 391]
    Time taken: 18.154, Avg Loss: 2.1464
    
    Epoch: 177
    Learning rate of this epoch: 0.0003
    Epoch: 177 [batchs trained: 391]
    Time taken: 17.911, Avg Loss: 2.1485
    
    Epoch: 178
    Learning rate of this epoch: 0.0003
    Epoch: 178 [batchs trained: 391]
    Time taken: 18.055, Avg Loss: 2.1413
    
    Epoch: 179
    Learning rate of this epoch: 0.0003
    Epoch: 179 [batchs trained: 391]
    Time taken: 17.693, Avg Loss: 2.1338
    
    Epoch: 180
    Learning rate of this epoch: 0.0003
    Epoch: 180 [batchs trained: 391]
    Time taken: 17.210, Avg Loss: 2.1476
    
    Epoch: 181
    Learning rate of this epoch: 0.0003
    Epoch: 181 [batchs trained: 391]
    Time taken: 17.877, Avg Loss: 2.1408
    
    Epoch: 182
    Learning rate of this epoch: 0.0003
    Epoch: 182 [batchs trained: 391]
    Time taken: 18.350, Avg Loss: 2.1380
    
    Epoch: 183
    Learning rate of this epoch: 0.0003
    Epoch: 183 [batchs trained: 391]
    Time taken: 17.407, Avg Loss: 2.1185
    
    Epoch: 184
    Learning rate of this epoch: 0.0003
    Epoch: 184 [batchs trained: 391]
    Time taken: 17.823, Avg Loss: 2.1279
    
    Epoch: 185
    Learning rate of this epoch: 0.0003
    Epoch: 185 [batchs trained: 391]
    Time taken: 17.776, Avg Loss: 2.1253
    
    Epoch: 186
    Learning rate of this epoch: 0.0003
    Epoch: 186 [batchs trained: 391]
    Time taken: 17.645, Avg Loss: 2.1383
    
    Epoch: 187
    Learning rate of this epoch: 0.0003
    Epoch: 187 [batchs trained: 391]
    Time taken: 18.269, Avg Loss: 2.1289
    
    Epoch: 188
    Learning rate of this epoch: 0.0003
    Epoch: 188 [batchs trained: 391]
    Time taken: 18.258, Avg Loss: 2.1258
    
    Epoch: 189
    Learning rate of this epoch: 0.0003
    Epoch: 189 [batchs trained: 391]
    Time taken: 18.167, Avg Loss: 2.1203
    
    Epoch: 190
    Learning rate of this epoch: 0.0003
    Epoch: 190 [batchs trained: 391]
    Time taken: 18.295, Avg Loss: 2.1319
    
    Epoch: 191
    Learning rate of this epoch: 0.0003
    Epoch: 191 [batchs trained: 391]
    Time taken: 18.227, Avg Loss: 2.1151
    
    Epoch: 192
    Learning rate of this epoch: 0.0003
    Epoch: 192 [batchs trained: 391]
    Time taken: 18.185, Avg Loss: 2.1190
    
    Epoch: 193
    Learning rate of this epoch: 0.0003
    Epoch: 193 [batchs trained: 391]
    Time taken: 18.187, Avg Loss: 2.1163
    
    Epoch: 194
    Learning rate of this epoch: 0.0003
    Epoch: 194 [batchs trained: 391]
    Time taken: 17.930, Avg Loss: 2.1101
    
    Epoch: 195
    Learning rate of this epoch: 0.0003
    Epoch: 195 [batchs trained: 391]
    Time taken: 17.652, Avg Loss: 2.1104
    
    Epoch: 196
    Learning rate of this epoch: 0.0003
    Epoch: 196 [batchs trained: 391]
    Time taken: 18.388, Avg Loss: 2.1152
    
    Epoch: 197
    Learning rate of this epoch: 0.0003
    Epoch: 197 [batchs trained: 391]
    Time taken: 18.241, Avg Loss: 2.1085
    
    Epoch: 198
    Learning rate of this epoch: 0.0003
    Epoch: 198 [batchs trained: 391]
    Time taken: 18.298, Avg Loss: 2.1050
    
    Epoch: 199
    Learning rate of this epoch: 0.0003
    Epoch: 199 [batchs trained: 391]
    Time taken: 17.807, Avg Loss: 2.1091



```python
state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
        }

if not os.path.isdir('checkpoint_'):
    os.mkdir('checkpoint_')

torch.save(state, './checkpoint_/cp.t7')
```

    /home/gorilla/.virtualenvs/jylee_py27/local/lib/python2.7/site-packages/torch/serialization.py:241: UserWarning: Couldn't retrieve source code for container of type NCEAverage. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


<h4>Evaluation - extract candidates</h4>


```python
# optional code: if you have checkpoint saved
import torch
import torch.backends.cudnn as cudnn

checkpoint = torch.load('./checkpoint_/cp.t7')

low_dim = 128
net = ResNet18(low_dim)
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

lemniscate = NCEAverage(128, 50000, 4096, 0.07, 0.5)

net.load_state_dict(checkpoint['net'])
lemniscate = checkpoint['lemniscate']

device = 'cuda'
net.to(device)
lemniscate.to(device)
```




    NCEAverage()




```python
def kNN_unsupervised(net, lemniscate, trainloader, testloader):
    
    net.eval()
    total = 0
    testsize = testloader.dataset.__len__() # 10000 for cifar

    trainFeatures = lemniscate.memory.t() # t() means transpose, features from memory bank (representing every image in the bank - cifar 50k)
    
    #sorted_candidates = torch.LongTensor([]).cuda()
    sorted_candidates = None
    
    timestart = time.time()
    
    with torch.no_grad():
        
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            end = time.time()

            dist = torch.mm(features, trainFeatures) # "cosine-similarity"
            
            yd, yi = torch.sort(dist, dim=1, descending=True) # sort for each row, from maximum cosine similarity to minimum
            
            if sorted_candidates is None:
                sorted_candidates = yi.detach().to('cpu').numpy()
            else:
                sorted_candidates = np.concatenate((sorted_candidates, yi.detach().to('cpu').numpy()), axis=0)
            #sorted_candidates = torch.cat((sorted_candidates, yi), 0)
                        
            total += targets.size(0)
    
    timetaken = time.time() - timestart
    print('Time taken: {:.2f} seconds'.format(timetaken))
        
    return sorted_candidates
```


```python
test_candidates = kNN_unsupervised(net, lemniscate, trainloader, testloader)
```

    Time taken: 50.48 seconds



```python
def imageplot(numpy_arr, name='output.png'):
    x_size = np.max([10, numpy_arr.shape[1] / 100])
    y_size = np.max([7, numpy_arr.shape[0] / 100])
    plt.figure(figsize=(x_size,y_size))
    plt.imshow(numpy_arr)
    plt.axis('off')
    plt.savefig(name)
    plt.show()

def display_candidates(test_data, bank, candidates, test_image_indexes=[1,2,3,4,5], num_candidates=10, desc=False):
    #num_y = len(test_image_indexes)

    total_image_np = None

    for i in test_image_indexes:
        total_image_row = None
        for j in range(num_candidates):
            entry_num = j
            if desc:
                entry_num = candidates.shape[1] - j - 1
            if total_image_row is None:
                total_image_row = np.hstack((test_data[i], bank[candidates[i,entry_num]]))
            else:
                total_image_row = np.hstack((total_image_row, bank[candidates[i,entry_num]]))
        if total_image_np is None:
            total_image_np = total_image_row
        else:
            total_image_np = np.vstack((total_image_np, total_image_row))

    imageplot(total_image_np, name='output.png')
```


```python
display_candidates(testloader.dataset.test_data, trainloader.dataset.train_data, test_candidates,
                  test_image_indexes=list(range(1000,1010)),
                  num_candidates=10)
```


    
![png](/assets/images/nonparam_files/output_37_0.png)
    


<h4>Further application - classification problem (measure accuracy)</h4>


```python
def kNN(net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    
    total = 0
    testsize = testloader.dataset.__len__() # 10000 for cifar

    trainFeatures = lemniscate.memory.t() # t() means transpose, features from memory bank (representing every image in the bank - cifar 50k)
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else: # goes to here for CIFAR10
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
    C = trainLabels.max() + 1 # 9 + 1 = 10

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)

            dist = torch.mm(features, trainFeatures) # "cosine-similarity"
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True) # find K(200) largest values for each row in dist, storing dist values and indexes of where that dist is
            candidates = trainLabels.view(1,-1).expand(batchSize, -1) # copy training data labels (50k) to batchSize (100) amount of rows (100x50000)
            retrieval = torch.gather(candidates, 1, yi) # candidates classes (k=200 so -retrieval: 100x200 - batchSize x k 2dimensional)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) # candidates classes one hotted
            yd_transform = yd.clone().div_(sigma).exp_()
            
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target (top1, top5 acc calculations only)
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)
    
    timetaken = time.time() - end
    print('Time taken: {:.4f} seconds'.format(timetaken))
    print('Top 1 acc: {:.2f} %'.format(top1*100./total))
    print('Top 5 acc: {:.2f} %'.format(top5*100./total))
    
    return top1/total
```


```python
acc = kNN(net, lemniscate, trainloader, testloader, 200, nce_t, 0) # measure topn accuracy based on model from last epoch
```

    Time taken: 1.9918 seconds
    Top 1 acc: 72.10 %
    Top 5 acc: 95.98 %


<br />
References:

(1)
<cite>
wu2018unsupervised,
  title={Unsupervised Feature Learning via Non-Parametric Instance Discrimination},
  author={Wu, Zhirong and Xiong, Yuanjun and Stella, X Yu and Lin, Dahua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
</cite>

Disclaimer:

This work is done purely for educational purposes thus non-commercial. 
