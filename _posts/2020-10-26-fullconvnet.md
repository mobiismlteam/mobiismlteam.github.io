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
from torchvision import models
from ignite.metrics import Accuracy, Loss
```


```python
batch_size = 1

loss_fn = nn.CrossEntropyLoss() # loss 조정해
opt_ = torch.optim.Adam
lr = 0.001
val_metrics = {
        #'acc': Accuracy(),
        'loss': Loss(loss_fn)
        }
device = 'cuda:1'
max_epochs = 1000
```

## Load Data


```python
import numpy as np
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor, CenterCrop

class ToTensor_(ToTensor):
    def __call__(self, pic):
        w, h = pic.size
        w, h = w - (w % 32), h - (h % 32)
        return ToTensor()(CenterCrop((h, w))(pic))

class PILToTensor(ToTensor):
    def __call__(self, pic):
        w, h = pic.size
        w, h = w - (w % 32), h - (h % 32)
        pic = CenterCrop((h, w))(pic)
        img = torch.as_tensor(np.asarray(pic))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2, 0, 1))
        for i, x in np.ndenumerate(img):
            if x == 255:
                img[i] = 21
        return img.long().view(-1)

train_dataset = VOCSegmentation(root='data/', image_set='train', transform=ToTensor_(), target_transform=PILToTensor(), download=False)
val_dataset = VOCSegmentation(root='data/', image_set='val', transform=ToTensor_(), target_transform=PILToTensor(), download=False)

print('# train data:', len(train_dataset))
print('# val data  :', len(val_dataset))
```

    # train data: 1464
    # val data  : 1449



```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
```

## Construct Model


```python
class FullConvNet(nn.Module):
    """
    cfg = [(n_channel, ), (), ...]
    """
    def __init__(self, n_class):
        super(FullConvNet, self).__init__()
        self.layers = []
        for layer in models.vgg19(pretrained=True).features:
            self.layers.append(layer)
            for param in layer.parameters():
                param.requires_grad = False
        n_currch = 512
        for n_nextch in [512, 256, 128, 64, 32]:
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.ConvTranspose2d(n_currch, n_nextch, kernel_size=3, stride=2, padding=1, output_padding=1))
            n_currch = n_nextch
        self.layers.append(nn.Conv2d(n_currch, n_class, kernel_size=1))
        for i, layer in enumerate(self.layers):
            self.add_module('l{}'.format(i), layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.view(1, 22, -1)
```


```python
model = FullConvNet(22)
```

## Train


```python
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

def train_net(net, opt, loss_fn, val_metrics, train_loader, val_loader, device):
    net.to(device)
    def prepare_batch(batch, device, non_blocking=False):
        x, y = batch
        return x.to(device), y.to(device)
    def output_transform(x, y, y_pred, loss):
        return (y_pred.max(1)[1], y) # =======================================
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
    Train - loss: 1.17 
    Val   - loss: 1.20 
    Epoch 2
    Train - loss: 0.97 
    Val   - loss: 1.02 
    Epoch 3
    Train - loss: 0.87 
    Val   - loss: 0.95 
    Epoch 4
    Train - loss: 0.79 
    Val   - loss: 0.92 
    Epoch 5
    Train - loss: 0.73 
    Val   - loss: 0.94 
    Epoch 6
    Train - loss: 0.68 
    Val   - loss: 0.98 
    Epoch 7
    Train - loss: 0.65 
    Val   - loss: 1.08 
    Epoch 8
    Train - loss: 0.60 
    Val   - loss: 1.15 
    Epoch 9
    Train - loss: 0.58 
    Val   - loss: 1.27 
    Epoch 10
    Train - loss: 0.56 
    Val   - loss: 1.29 
    Epoch 11
    Train - loss: 0.53 
    Val   - loss: 1.38 
    Epoch 12
    Train - loss: 0.49 
    Val   - loss: 1.47 
    Epoch 13
    Train - loss: 0.46 
    Val   - loss: 1.48 
    Epoch 14
    Train - loss: 0.47 
    Val   - loss: 1.57 
    Epoch 15
    Train - loss: 0.48 
    Val   - loss: 1.67 
    Epoch 16
    Train - loss: 0.49 
    Val   - loss: 1.66 
    Epoch 17
    Train - loss: 0.47 
    Val   - loss: 1.76 
    Epoch 18
    Train - loss: 0.37 
    Val   - loss: 1.74 
    Epoch 19
    Train - loss: 0.36 
    Val   - loss: 1.80 
    Epoch 20
    Train - loss: 0.34 
    Val   - loss: 1.77 
    Epoch 21
    Train - loss: 0.32 
    Val   - loss: 1.83 
    Epoch 22
    Train - loss: 0.38 
    Val   - loss: 1.95 
    Epoch 23
    Train - loss: 0.31 
    Val   - loss: 1.91 
    Epoch 24
    Train - loss: 0.28 
    Val   - loss: 1.89 
    Epoch 25
    Train - loss: 0.28 
    Val   - loss: 2.05 
    Epoch 26
    Train - loss: 0.26 
    Val   - loss: 1.98 
    Epoch 27
    Train - loss: 0.20 
    Val   - loss: 1.93 
    Epoch 28
    Train - loss: 0.21 
    Val   - loss: 2.01 
    Epoch 29
    Train - loss: 0.22 
    Val   - loss: 2.20 
    Epoch 30
    Train - loss: 0.19 
    Val   - loss: 2.15 


    Engine run is terminating due to exception: .
    Engine run is terminating due to exception: .



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-8ac1a0a79e01> in <module>
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


    <ipython-input-8-05b4a011b4ff> in log_training_results(trainer)
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


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torchvision/datasets/voc.py in __getitem__(self, index)
        123 
        124         if self.transforms is not None:
    --> 125             img, target = self.transforms(img, target)
        126 
        127         return img, target


    ~/.virtualenvs/BaeJR_py36/lib/python3.6/site-packages/torchvision/datasets/vision.py in __call__(self, input, target)
         61             input = self.transform(input)
         62         if self.target_transform is not None:
    ---> 63             target = self.target_transform(target)
         64         return input, target
         65 


    <ipython-input-4-e0a946e58f28> in __call__(self, pic)
         18         img = img.permute((2, 0, 1))
         19         for i, x in np.ndenumerate(img):
    ---> 20             if x == 255:
         21                 img[i] = 21
         22         return img.long().view(-1)


    KeyboardInterrupt: 


## Samples


```python
from torchvision.transforms import ToPILImage

tpi = ToPILImage()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30, 15))
ax = fig.subplots(2, 4)

model.to('cpu')
model.eval()
for i, (image_, _) in zip(range(4), val_loader):
    image = image_.squeeze()
    ax[0][i].imshow(tpi(image))
    ax[1][i].imshow(torch.argmax(model(image_), dim=1).view([i for i in image.shape[1: ]]))

fig.show()
```


![png](/assets/images/fullconvnet_files/fullconvnet_15_0.png)



```python

```
