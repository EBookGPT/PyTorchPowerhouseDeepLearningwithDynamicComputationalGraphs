# Chapter 4: Computation Graphs in PyTorch

Welcome to the fourth chapter of our journey through the realm of PyTorch Powerhouse: Deep Learning with Dynamic Computational Graphs. In the previous chapter, we learned about the fundamental data structure of PyTorch - tensors, and how to work with them.  Now, we'll explore the power of computation graphs in PyTorch.

Computation graphs are the backbone of PyTorch's automatic differentiation engine, which allows us to train complex neural networks with ease. In this chapter, we'll delve into the basics of computation graphs, and see how PyTorch utilizes them to make our lives as deep learning practitioners easier.

We'll start by discussing what computation graphs are, and why they are useful in deep learning. We'll then move to different types of computation graphs - static and dynamic - and explain why PyTorch has chosen to go with the latter. We'll also describe the different components of a PyTorch computation graph, such as nodes and edges.

Finally, we'll get our hands dirty with some code, and see how we can use PyTorch's computation graphs to implement a simple neural network that can classify images. We'll use all the knowledge from previous chapters, and take our first step towards becoming proficient PyTorch programmers.

So, sharpen your fangs, and let's sink our teeth into the fascinating world of PyTorch computation graphs!
# Chapter 4: Computation Graphs in PyTorch

Count Dracula, the master of the dark arts, sat at his desk deep in thought. He had been contemplating, for quite some time now, the best way to automate the complex neural network that he had been working on for so long. The Count was known for his innovative use of technology in his dark obsession with deep learning. But he was struggling with the fact that the process of updating thousands of parameters in a neural network was not only time-consuming, but also prone to errors. 

As he sat there, staring into the abyss, he remembered a rumor he had heard about an advanced technology called "computation graphs." Intrigued, he began his research and soon discovered PyTorch - a new and exciting deep learning framework that boasted dynamic computational graphs as one of its key features.

The Count was amazed! Finally, he had found the answer to his problems. With PyTorch's dynamic computational graphs, he would be able to track every operation in his neural network, automatically calculate the gradients at each step, and make training his model much easier. 

Over the next few days, the Count locked himself in his laboratory, working tirelessly on his PyTorch neural network. He spent hours carefully crafting the computation graph, adding each layer and node one by one. The complex graph grew larger and more intricate by the hour, but the Count was not deterred. He was determined to make this the most powerful neural network in the world.

As he worked, strange and incredible things started happening inside the graph. The nodes and edges began to pulse with a fiery energy, and the Count could feel the raw power of the network coursing through his veins. He knew that this was it - he had finally achieved his goal of automating his neural network with PyTorch's dynamic computational graphs.

And so, the Count emerged from his laboratory a changed man. With the power of PyTorch, and the knowledge of computational graphs, he was now unstoppable. He went on to revolutionize the world of deep learning, and his neural network became renowned as one of the most powerful models in history.

Remember, just as Count Dracula was able to harness the power of PyTorch's computation graphs to automate his neural network and achieve his goals, you too can use PyTorch to create powerful machine learning models that will change the world.
# Code Explanation

Now that we have been introduced to Dracula's story and the role of computation graphs in PyTorch, let's dive deeper into the code that was used to automate his neural network.

PyTorch offers a range of methods to create computation graphs, but in this story, we will be using the `nn.Module` class to define our neural network architecture. The `nn.Module` is a base class that defines a set of methods and attributes that are essential for defining a PyTorch network.

First, we import the necessary libraries and define our neural network class, `DeepLearningModel`, which inherits from the `nn.Module` base class.

```python
import torch
import torch.nn as nn

class DeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
```

Our neural network has three layers - a fully connected input layer with 784 nodes, a hidden layer with 512 nodes, and another hidden layer with 256 nodes. It also has an output layer with 10 nodes, representing the 10 possible classes in our classification task.

After defining the layers, we add activation functions to each layer. In this case, we use the `ReLU` activation function for the hidden layers and `Softmax` for the output layer. These activation functions help prevent overfitting and improve the accuracy of our model.

```python
def forward(self, input):
        output = self.relu(self.layer1(input))
        output = self.relu(self.layer2(output))
        output = self.softmax(self.layer3(output))
        return output
```

Next, we define the `forward` method, which takes an input and passes it through the layers, applying the activation functions at each step. The `forward` method is what actually performs the forward pass of our computation graph, and it is automatically traced by PyTorch's automatic differentiation engine.

Finally, we define an instance of our `DeepLearningModel` class and pass in some sample data to test our network.

```python
model = DeepLearningModel()

input_tensor = torch.randn(1, 784)  # Random input tensor with shape (1, 784)
output = model(input_tensor)

print(output)
```

We create a new instance of our `DeepLearningModel` class called `model`. We then create a random input tensor with shape `(1, 784)` and pass it to our `model`.

The output that our model produces is a tensor of shape `(1, 10)` - this is because we have 10 classes in our classification task. The `argmax` method is used to find the index of the maximum value in the output tensor, which corresponds to the predicted class of the input.

```python
print(output.argmax(dim=1))
```

And that's it! With just a few lines of PyTorch code, we have created and tested a deep neural network that can classify images. Thanks to PyTorch's dynamic computational graph, our neural network is automatically traced, and the gradients are calculated, making training our model a breeze.


[Next Chapter](05_Chapter05.md)