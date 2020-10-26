## Settings


```python
%load_ext autoreload
%autoreload 2
```

# Main


```python
import torch
from torch import nn
from torch.nn import functional as F
from ignite.metrics import Accuracy, Loss
```


```python
batch_size = 10000

loss_fn = nn.CrossEntropyLoss()
opt_ = torch.optim.Adam
lr = 0.001
val_metrics = {
        'acc': Accuracy(),
        'loss': Loss(loss_fn)
        }
device = 'cuda:0'
max_epochs = 1000
```

## Load Data


```python
from torchvision.datasets import MNIST
from torchvision import transforms

train_data = MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

print('number of training data: ', len(train_data))
print('number of test data: ', len(test_data))
```

    number of training data:  60000
    number of test data:  10000



```python
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

s = StratifiedShuffleSplit(n_splits=1, test_size=10000)
for train_idx, val_idx in s.split(train_data.data, train_data.targets):
    train_data, val_data = Subset(train_data, train_idx), Subset(train_data, val_idx)
```


```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
```

## Construct Model


```python
class SPPLayer(nn.Module):
    """
    cfg = [(H1, W1), (H2, W2), ...]
    """
    def __init__(self, cfg):
        super(SPPLayer, self).__init__()
        self.layers = []
        for size in cfg:
            self.layers.append(nn.AdaptiveMaxPool2d(size))
        for i, l in enumerate(self.layers):
            self.add_module('l{}'.format(i), l)
            
    def forward(self, x):
        x = torch.cat([l(x).flatten(1, 3) for l in self.layers], 1)
        return x
        
class SPPNet(nn.Module):
    def __init__(self, cfg):
        super(SPPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.spp = SPPLayer(cfg)
        self.fc = nn.Linear(sum([h * w * 8 for h, w in cfg]), 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
        x = self.spp(x)
        x = self.fc(x)
        return x
```


```python
model = SPPNet([(4, 4), (2, 2), (1, 1)])
```


```python
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

def train_net(net, opt, loss_fn, val_metrics, train_loader, val_loader, device):
    net.to(device)
    def prepare_batch(batch, device, non_blocking=False):
        x, y = batch
        return x.to(device), y.to(device)
    def output_transform(x, y, y_pred, loss):
        return (y_pred.max(1)[1], y)
    trainer = create_supervised_trainer(net, opt, loss_fn, device,
            prepare_batch=prepare_batch, output_transform=output_transform)
    evaluator = create_supervised_evaluator(net, val_metrics, device,
            prepare_batch=prepare_batch)
    s = '{}: {:.2f} '
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        print('Epoch {}'.format(trainer.state.epoch))
        message = 'Train - '
        for m in val_metrics.keys():
            message += s.format(m, evaluator.state.metrics[m])
        print(message)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        message = 'Val   - '
        for m in val_metrics.keys():
            message += s.format(m, evaluator.state.metrics[m])
        print(message)
    return trainer
```


```python
opt = opt_(model.parameters(), lr)

trainer = train_net(model, opt, loss_fn, val_metrics,
        train_loader, val_loader, device)
trainer.run(train_loader, max_epochs=max_epochs)
```

    Epoch 1
    Train - acc: 0.26 loss: 2.28 
    Val   - acc: 0.26 loss: 2.28 
    Epoch 2
    Train - acc: 0.22 loss: 2.24 
    Val   - acc: 0.22 loss: 2.24 
    Epoch 3
    Train - acc: 0.32 loss: 2.20 
    Val   - acc: 0.31 loss: 2.20 
    Epoch 4
    Train - acc: 0.55 loss: 2.13 
    Val   - acc: 0.55 loss: 2.13 
    Epoch 5
    Train - acc: 0.65 loss: 2.03 
    Val   - acc: 0.65 loss: 2.04 
    Epoch 6
    Train - acc: 0.67 loss: 1.91 
    Val   - acc: 0.67 loss: 1.91 
    Epoch 7
    Train - acc: 0.70 loss: 1.74 
    Val   - acc: 0.69 loss: 1.75 
    Epoch 8
    Train - acc: 0.71 loss: 1.55 
    Val   - acc: 0.71 loss: 1.55 
    Epoch 9
    Train - acc: 0.73 loss: 1.34 
    Val   - acc: 0.73 loss: 1.35 
    Epoch 10
    Train - acc: 0.75 loss: 1.15 
    Val   - acc: 0.75 loss: 1.15 
    Epoch 11
    Train - acc: 0.77 loss: 0.97 
    Val   - acc: 0.76 loss: 0.98 
    Epoch 12
    Train - acc: 0.79 loss: 0.83 
    Val   - acc: 0.79 loss: 0.83 
    Epoch 13
    Train - acc: 0.81 loss: 0.71 
    Val   - acc: 0.81 loss: 0.72 
    Epoch 14
    Train - acc: 0.83 loss: 0.62 
    Val   - acc: 0.83 loss: 0.63 
    Epoch 15
    Train - acc: 0.85 loss: 0.55 
    Val   - acc: 0.85 loss: 0.56 
    Epoch 16
    Train - acc: 0.87 loss: 0.50 
    Val   - acc: 0.87 loss: 0.51 
    Epoch 17
    Train - acc: 0.88 loss: 0.45 
    Val   - acc: 0.88 loss: 0.46 
    Epoch 18
    Train - acc: 0.89 loss: 0.41 
    Val   - acc: 0.89 loss: 0.43 
    Epoch 19
    Train - acc: 0.90 loss: 0.38 
    Val   - acc: 0.89 loss: 0.40 
    Epoch 20
    Train - acc: 0.90 loss: 0.36 
    Val   - acc: 0.90 loss: 0.37 
    Epoch 21
    Train - acc: 0.91 loss: 0.34 
    Val   - acc: 0.91 loss: 0.35 
    Epoch 22
    Train - acc: 0.91 loss: 0.32 
    Val   - acc: 0.91 loss: 0.33 
    Epoch 23
    Train - acc: 0.92 loss: 0.30 
    Val   - acc: 0.92 loss: 0.31 
    Epoch 24
    Train - acc: 0.92 loss: 0.29 
    Val   - acc: 0.92 loss: 0.30 
    Epoch 25
    Train - acc: 0.92 loss: 0.27 
    Val   - acc: 0.92 loss: 0.29 
    Epoch 26
    Train - acc: 0.93 loss: 0.26 
    Val   - acc: 0.93 loss: 0.27 
    Epoch 27
    Train - acc: 0.93 loss: 0.25 
    Val   - acc: 0.93 loss: 0.26 
    Epoch 28
    Train - acc: 0.93 loss: 0.24 
    Val   - acc: 0.93 loss: 0.25 
    Epoch 29
    Train - acc: 0.93 loss: 0.23 
    Val   - acc: 0.93 loss: 0.24 
    Epoch 30
    Train - acc: 0.94 loss: 0.23 
    Val   - acc: 0.93 loss: 0.24 
    Epoch 31
    Train - acc: 0.94 loss: 0.22 
    Val   - acc: 0.94 loss: 0.23 
    Epoch 32
    Train - acc: 0.94 loss: 0.21 
    Val   - acc: 0.94 loss: 0.22 
    Epoch 33
    Train - acc: 0.94 loss: 0.20 
    Val   - acc: 0.94 loss: 0.21 
    Epoch 34
    Train - acc: 0.94 loss: 0.20 
    Val   - acc: 0.94 loss: 0.21 
    Epoch 35
    Train - acc: 0.94 loss: 0.19 
    Val   - acc: 0.94 loss: 0.20 
    Epoch 36
    Train - acc: 0.95 loss: 0.19 
    Val   - acc: 0.95 loss: 0.19 
    Epoch 37
    Train - acc: 0.95 loss: 0.18 
    Val   - acc: 0.95 loss: 0.19 
    Epoch 38
    Train - acc: 0.95 loss: 0.17 
    Val   - acc: 0.95 loss: 0.18 
    Epoch 39
    Train - acc: 0.95 loss: 0.17 
    Val   - acc: 0.95 loss: 0.18 
    Epoch 40
    Train - acc: 0.95 loss: 0.17 
    Val   - acc: 0.95 loss: 0.18 
    Epoch 41
    Train - acc: 0.95 loss: 0.16 
    Val   - acc: 0.95 loss: 0.17 
    Epoch 42
    Train - acc: 0.95 loss: 0.16 
    Val   - acc: 0.95 loss: 0.17 
    Epoch 43
    Train - acc: 0.95 loss: 0.15 
    Val   - acc: 0.95 loss: 0.16 
    Epoch 44
    Train - acc: 0.96 loss: 0.15 
    Val   - acc: 0.95 loss: 0.16 
    Epoch 45
    Train - acc: 0.96 loss: 0.15 
    Val   - acc: 0.95 loss: 0.16 
    Epoch 46
    Train - acc: 0.96 loss: 0.14 
    Val   - acc: 0.96 loss: 0.15 
    Epoch 47
    Train - acc: 0.96 loss: 0.14 
    Val   - acc: 0.96 loss: 0.15 
    Epoch 48
    Train - acc: 0.96 loss: 0.14 
    Val   - acc: 0.96 loss: 0.15 
    Epoch 49
    Train - acc: 0.96 loss: 0.14 
    Val   - acc: 0.96 loss: 0.15 
    Epoch 50
    Train - acc: 0.96 loss: 0.13 
    Val   - acc: 0.96 loss: 0.14 
    Epoch 51
    Train - acc: 0.96 loss: 0.13 
    Val   - acc: 0.96 loss: 0.14 
    Epoch 52
    Train - acc: 0.96 loss: 0.13 
    Val   - acc: 0.96 loss: 0.14 
    Epoch 53
    Train - acc: 0.96 loss: 0.13 
    Val   - acc: 0.96 loss: 0.14 
    Epoch 54
    Train - acc: 0.96 loss: 0.13 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 55
    Train - acc: 0.96 loss: 0.12 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 56
    Train - acc: 0.96 loss: 0.12 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 57
    Train - acc: 0.96 loss: 0.12 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 58
    Train - acc: 0.96 loss: 0.12 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 59
    Train - acc: 0.96 loss: 0.12 
    Val   - acc: 0.96 loss: 0.13 
    Epoch 60
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 61
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 62
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 63
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 64
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 65
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 66
    Train - acc: 0.97 loss: 0.11 
    Val   - acc: 0.96 loss: 0.12 
    Epoch 67
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.12 
    Epoch 68
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 69
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 70
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 71
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 72
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 73
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 74
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 75
    Train - acc: 0.97 loss: 0.10 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 76
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.11 
    Epoch 77
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 78
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 79
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 80
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 81
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 82
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 83
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 84
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 85
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 86
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 87
    Train - acc: 0.97 loss: 0.09 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 88
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 89
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.10 
    Epoch 90
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 91
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 92
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 93
    Train - acc: 0.97 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 94
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 95
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 96
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 97
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 98
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 99
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 100
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 101
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 102
    Train - acc: 0.98 loss: 0.08 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 103
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 104
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 105
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 106
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 107
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 108
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.09 
    Epoch 109
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 110
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 111
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 112
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 113
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 114
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 115
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 116
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 117
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.97 loss: 0.08 
    Epoch 118
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 119
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 120
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 121
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 122
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 123
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 124
    Train - acc: 0.98 loss: 0.07 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 125
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 126
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 127
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 128
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 129
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 130
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 131
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 132
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 133
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 134
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 135
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 136
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 137
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 138
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.08 
    Epoch 139
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 140
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 141
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 142
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 143
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 144
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 145
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 146
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 147
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 148
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 149
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 150
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 151
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 152
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 153
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 154
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 155
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 156
    Train - acc: 0.98 loss: 0.06 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 157
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 158
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 159
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 160
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 161
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 162
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 163
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 164
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 165
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 166
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 167
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 168
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 169
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 170
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 171
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 172
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 173
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 174
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 175
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 176
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 177
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 178
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 179
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 180
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 181
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 182
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 183
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 184
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 185
    Train - acc: 0.98 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 186
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 187
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 188
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 189
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 190
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 191
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 192
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 193
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.07 
    Epoch 194
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 195
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 196
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 197
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 198
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 199
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 200
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 201
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 202
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 203
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 204
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 205
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 206
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 207
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 208
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 209
    Train - acc: 0.99 loss: 0.05 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 210
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 211
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 212
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 213
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 214
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 215
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 216
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 217
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 218
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 219
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 220
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 221
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 222
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 223
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 224
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 225
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 226
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 227
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 228
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 229
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 230
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 231
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 232
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 233
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 234
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 235
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 236
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 237
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 238
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 239
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 240
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 241
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 242
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 243
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 244
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 245
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 246
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 247
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 248
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 249
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 250
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 251
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 252
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 253
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 254
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 255
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 256
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 257
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 258
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 259
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 260
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 261
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 262
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 263
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 264
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 265
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 266
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 267
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 268
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 269
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 270
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 271
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 272
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 273
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 274
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 275
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 276
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 277
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 278
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 279
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 280
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 281
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 282
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 283
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 284
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 285
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 286
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 287
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 288
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 289
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 290
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 291
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 292
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 293
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 294
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 295
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 296
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 297
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 298
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 299
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 300
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 301
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 302
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 303
    Train - acc: 0.99 loss: 0.04 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 304
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 305
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 306
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 307
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 308
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 309
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 310
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 311
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 312
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 313
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 314
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 315
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 316
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 317
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 318
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 319
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 320
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 321
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 322
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 323
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 324
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 325
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 326
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 327
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 328
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 329
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.06 
    Epoch 330
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 331
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 332
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 333
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 334
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 335
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 336
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 337
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 338
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 339
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 340
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 341
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 342
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 343
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 344
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 345
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 346
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 347
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 348
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 349
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 350
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 351
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 352
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 353
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 354
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 355
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 356
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 357
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 358
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 359
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 360
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 361
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 362
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 363
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 364
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 365
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 366
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 367
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 368
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 369
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 370
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 371
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 372
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 373
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 374
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 375
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 376
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 377
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 378
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 379
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 380
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.98 loss: 0.05 
    Epoch 381
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 382
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 383
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 384
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 385
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 386
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 387
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 388
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 389
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 390
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 391
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 392
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 393
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 394
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 395
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 396
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 397
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 398
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 399
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 400
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 401
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 402
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 403
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 404
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 405
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 406
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 407
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 408
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 409
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 410
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 411
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 412
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 413
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 414
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 415
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 416
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 417
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 418
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 419
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 420
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 421
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 422
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 423
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 424
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 425
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 426
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 427
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 428
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 429
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 430
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 431
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 432
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 433
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 434
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 435
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 436
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 437
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 438
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 439
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 440
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 441
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 442
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 443
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 444
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 445
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 446
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 447
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 448
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 449
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 450
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 451
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 452
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 453
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 454
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 455
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 456
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 457
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 458
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 459
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 460
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 461
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 462
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 463
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 464
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 465
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 466
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 467
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 468
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 469
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 470
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 471
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 472
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 473
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 474
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 475
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 476
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 477
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 478
    Train - acc: 0.99 loss: 0.03 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 479
    Train - acc: 0.99 loss: 0.02 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 480
    Train - acc: 0.99 loss: 0.02 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 481
    Train - acc: 0.99 loss: 0.02 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 482
    Train - acc: 0.99 loss: 0.02 
    Val   - acc: 0.99 loss: 0.05 
    Epoch 483
    Train - acc: 0.99 loss: 0.02 
    Val   - acc: 0.99 loss: 0.05 


    Engine run is terminating due to exception: .
    Engine run is terminating due to exception: .



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-10-8ac1a0a79e01> in <module>
          3 trainer = train_net(model, opt, loss_fn, val_metrics,
          4         train_loader, val_loader, device)
    ----> 5 trainer.run(train_loader, max_epochs=max_epochs)
    

    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in run(self, data, max_epochs, epoch_length, seed)
        656 
        657         self.state.dataloader = data
    --> 658         return self._internal_run()
        659 
        660     @staticmethod


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _internal_run(self)
        720             self._dataloader_iter = None
        721             self.logger.error("Engine run is terminating due to exception: %s.", str(e))
    --> 722             self._handle_exception(e)
        723 
        724         self._dataloader_iter = None


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _handle_exception(self, e)
        435             self._fire_event(Events.EXCEPTION_RAISED, e)
        436         else:
    --> 437             raise e
        438 
        439     @property


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _internal_run(self)
        708                     self.logger.info(elapsed_time_message)
        709                     break
    --> 710                 self._fire_event(Events.EPOCH_COMPLETED)
        711                 self.logger.info(elapsed_time_message)
        712 


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _fire_event(self, event_name, *event_args, **event_kwargs)
        391                 kwargs.update(event_kwargs)
        392                 first, others = ((args[0],), args[1:]) if (args and args[0] == self) else ((), args)
    --> 393                 func(*first, *(event_args + others), **kwargs)
        394 
        395     def fire_event(self, event_name: Any) -> None:


    <ipython-input-9-05b4a011b4ff> in log_training_results(trainer)
         15     @trainer.on(Events.EPOCH_COMPLETED)
         16     def log_training_results(trainer):
    ---> 17         evaluator.run(train_loader)
         18         print('Epoch {}'.format(trainer.state.epoch))
         19         message = 'Train - '


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in run(self, data, max_epochs, epoch_length, seed)
        656 
        657         self.state.dataloader = data
    --> 658         return self._internal_run()
        659 
        660     @staticmethod


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _internal_run(self)
        720             self._dataloader_iter = None
        721             self.logger.error("Engine run is terminating due to exception: %s.", str(e))
    --> 722             self._handle_exception(e)
        723 
        724         self._dataloader_iter = None


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _handle_exception(self, e)
        435             self._fire_event(Events.EXCEPTION_RAISED, e)
        436         else:
    --> 437             raise e
        438 
        439     @property


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _internal_run(self)
        695                     self._setup_engine()
        696 
    --> 697                 time_taken = self._run_once_on_dataset()
        698                 self.state.times[Events.EPOCH_COMPLETED.name] = time_taken
        699                 hours, mins, secs = _to_hours_mins_secs(time_taken)


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/ignite/engine/engine.py in _run_once_on_dataset(self)
        737                     if self.last_event_name != Events.DATALOADER_STOP_ITERATION:
        738                         self._fire_event(Events.GET_BATCH_STARTED)
    --> 739                     self.state.batch = next(self._dataloader_iter)
        740                     self._fire_event(Events.GET_BATCH_COMPLETED)
        741                     iter_counter += 1


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torch/utils/data/dataloader.py in __next__(self)
        343 
        344     def __next__(self):
    --> 345         data = self._next_data()
        346         self._num_yielded += 1
        347         if self._dataset_kind == _DatasetKind.Iterable and \


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torch/utils/data/dataloader.py in _next_data(self)
        383     def _next_data(self):
        384         index = self._next_index()  # may raise StopIteration
    --> 385         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        386         if self._pin_memory:
        387             data = _utils.pin_memory.pin_memory(data)


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torch/utils/data/dataset.py in __getitem__(self, idx)
        255 
        256     def __getitem__(self, idx):
    --> 257         return self.dataset[self.indices[idx]]
        258 
        259     def __len__(self):


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torchvision/datasets/mnist.py in __getitem__(self, index)
         95 
         96         if self.transform is not None:
    ---> 97             img = self.transform(img)
         98 
         99         if self.target_transform is not None:


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torchvision/transforms/transforms.py in __call__(self, pic)
         99             Tensor: Converted image.
        100         """
    --> 101         return F.to_tensor(pic)
        102 
        103     def __repr__(self):


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torchvision/transforms/functional.py in to_tensor(pic)
         98     img = img.transpose(0, 1).transpose(0, 2).contiguous()
         99     if isinstance(img, torch.ByteTensor):
    --> 100         return img.float().div(255)
        101     else:
        102         return img


    KeyboardInterrupt: 



```python

```
