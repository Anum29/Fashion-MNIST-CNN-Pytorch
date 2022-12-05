# Fashion-MNIST-CNN-Pytorch

We'll build a simple convolutional neural network in PyTorch and train it to recognize clothes using the Fashion MNIST dataset.

## Installation

Install the dataset

```bash
train_set = torchvision.datasets.FashionMNIST(root = ".", train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train=False,download=True, transform=transforms.ToTensor())

```

## General Information

The default code runs on relu activation function with learning rate 0.1. One test data batch is a  tensor of shape: . This means we have 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one). We can plot the test output.
```
with torch.no_grad():
  output = network(test_data)
```

<img src="https://github.com/Anum29/Fashion-MNIST-CNN-Pytorch/blob/main/labeled_output.png">


## Requirements:
```
torch == 0.4.1
torchvision >= 0.1.9
numpy >= 1.13.0
matplotlib >= 1.5
```

## References:

[MNIST Handwritten Digit Recognition in PyTorch](https://nextjournal.com/gkoehler/pytorch-mnist)
