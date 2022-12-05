# Fashion-MNIST-CNN-Pytorch

We'll build a simple convolutional neural network in PyTorch and train it to recognize clothes using the Fashion MNIST dataset.

## Installation

Install the dataset

```bash
train_set = torchvision.datasets.FashionMNIST(root = ".", train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train=False,download=True, transform=transforms.ToTensor())

```

## General Information

The given neural network configuration is:

<img src="https://github.com/Anum29/Fashion-MNIST-CNN-Pytorch/blob/main/nn class.png">

Now, we can define the network class
```
class Net(nn.Module): 
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        #self.conv2_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def init_weights(self, m):
        """Define the weights"""
        return nn.init.xavier_uniform(m.weight)  # xavier initialization of weights

    def forward(self, x):
        """Define the forward function"""
        #m = nn.Sigmoid()  sigmoid activation function
        #m = nn.Tanh()  tanh activation function
        #x = F.elu(F.max_pool2d(self.conv1(x), 2)) elu activation function
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # relu activation function - default activation function
        #x = F.dropout(x, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return F.log_softmax(x)
```
During training the network `train(epoch)`, we calculate the accuracy on the training and test set.


<img src="https://github.com/Anum29/Fashion-MNIST-CNN-Pytorch/blob/main/accuracy.png">


We have 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one). We can plot the test output.
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
