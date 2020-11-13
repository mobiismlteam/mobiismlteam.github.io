---
layout: article
title: NonParam
mathjax: true
toc : true
tags : UnsupervisedFeatureLearning
---



<h1>Zhirong Wu et al. Unsupervised Feature Learning via Non-Parametric Instance Discrimination (2018)$^{(1)}$ - Implementation</h1>
<br />
<br />

<h3>Motivation</h3>
<br />
<img src="/assets/images/nonparam_files/np_motivation.png" width="600">
<br />
<h5>Observation of a classification model</h5>
When the image of a leopard is tested on a supervised learning model, the responses are produced as output above.<br />
The highest responses are leopard, and those that look similar to a leopard,<br />
the lowest responses were those that look nothing like a leopard.
<br />
<h5>Theory Crafting</h5>
Typical discriminative learning methods do not instruct the model to learn the similarity among semantic categories,<br />
but they appear to discover the apparent similarity automatically when learning.
<br />
<h5>Reforming The Theory Crafting Done Above</h5>
As the semantic annotations are mere classes that are independent of each other by principle,<br />
the apparent similarity must be learned from the visual data themselves.
<br />
<h5>Question we would like to answer</h5>
Following the last paragraph..<br />
Can we learn to discriminate the individual instances, without any notion of semantic categories?
<br />
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
<br />

<h3>The Pipeline</h3>
<br />
<img src="/assets/images/nonparam_files/np_implementation_pipeline.png" width="1000">
<br />
<br />
Above is the training network. <br />

In testing, we calculate the similarity between the query and each element in the memory bank and output top "k" candidates. <br />

<h3>Result of Our Training (sample)</h3>
<img src="/assets/images/nonparam_files/np_res_norecompute.png" >
<br />

<h3>Implementions</h3>
<br />

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
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
```


```python
trainset = CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
ndata = trainset.__len__()

testset = CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
```


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

<h5>Net architecture (to be edited with cleaner text)</h5>
<img src="/assets/images/nonparam_files/np_resnet_paintedit.png">
<br />
<br />
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

<h5>Learning curve</h5>
<img src="/assets/images/nonparam_files/np_res_norecompute.png" >
<br />
<br />
<br />

```python
state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
        }

if not os.path.isdir('checkpoint_'):
    os.mkdir('checkpoint_')

torch.save(state, './checkpoint_/cp.t7')
```


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
