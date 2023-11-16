# Results from training all the models from paper 'ConvNet for 2020s'

## 1. Resnet:

```Optim = SGD```
```LR = 0.01```

...
```Epoch 10/10, Training Loss: 0.4447, Training Accuracy: 84.82%, Validation Loss: 0.5558, Validation Accuracy: 83.48%```

### Analysis: 

Converge faster but trains very long. 

-----

## 2. Patchify + AdamW + Changing stage compute ratio 

```Optim = Adam```
```LR = 0.005```
```Epoch 20```

...
```Epoch 15/20, Training Loss: 0.9100, Training Accuracy: 68.13%, Validation Loss: 0.8360, Validation Accuracy: 71.11```

### Analysis: 

Converge slow, decrease in accuracy which is expected 

------

## 3. ResNeXt-ify + Inverted Bottleneck

```Optim = Adam```
```LR = 0.005```
```Epoch 20```

... 
```Epoch 17/20, Training Loss: 0.5807, Training Accuracy: 80.17%, Validation Loss: 0.5448, Validation Accuracy: 81.08```

### Analysis: 
Works good!
-----

## 4. Larger Kernel Sizes

```Optim = Adam```
```LR = 0.005```
```Epoch 20```

....
```Epoch 19/20, Training Loss: 0.5628, Training Accuracy: 80.58%, Validation Loss: 0.6100, Validation Accuracy: 78.99```


## 5. Micro design
```Epoch 300```

...
```Epoch 283/300, Validation Accuracy: 76.49%```
