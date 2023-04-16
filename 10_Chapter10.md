# Chapter 10: Transfer Learning with PyTorch 
<img src="https://images.unsplash.com/photo-1560087632-a6578e40d1b0?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60" alt="transfer learning" width="500"/>

As we dive deeper into the fascinating world of PyTorch, we come to the topic of transfer learning. In the previous chapter, we learned about advanced PyTorch modules such as Dropout and BatchNorm, and Activation Functions which played an important role in building deep neural networks. Transfer learning, on the other hand, allows us to use pre-trained models and their weights to build our custom models.

Have you ever wondered how you could modify a model that can classify different types of cars, to classify different types of fruits? Instead of building a neural network from scratch for fruit classification, we can take advantage of the pre-trained car classification model, and "transfer" its knowledge to our fruit classification problem. This not only saves us time in building and training models but also prevents the need for a large dataset.

Transfer learning has proven to be a valuable technique in deep learning, especially in cases where there are limited datasets for training models. In fact, it has shown to enhance the performance of models, improve their generalization, and reduce overfitting. 

In this chapter, we will learn how to load and use pre-trained models in PyTorch, modify them for our custom problem, and save them for future use. We will also discuss several applications of transfer learning, including image classification, object detection, and natural language processing.

Are you ready to dive into the world of transfer learning with PyTorch Powerhouse? Let's get started!
# Chapter 10: Transfer Learning with PyTorch - Dracula Story 

The night was dark and stormy as Dracula sat in his castle, brooding about his inability to classify different types of monsters accurately. He had tried everything, from building complex neural networks to collecting more data, but nothing seemed to work. It was then that he heard a knock on his door.

"Who could that be?" Dracula muttered to himself as he went to answer the door. To his surprise, it was a group of travelers who had lost their way in the storm. As he welcomed them into his castle, one of the travelers mentioned that he was a deep learning expert, and his name was PyTorch Powerhouse.

Dracula's eyes widened in excitement. He had heard of PyTorch Powerhouse before and knew that he was just the expert he needed to solve his classification problem. The two of them got talking, and Dracula explained his problem with monster classification.

PyTorch Powerhouse listened intently and suggested that they use transfer learning. He explained that transfer learning is a technique in which pre-trained models and their weights are used to build custom models for a specific problem.

"I have a pre-trained model that can classify different types of bats" said PyTorch Powerhouse. "We can modify it to classify different types of monsters for you. This will save us the time and effort required to build a new model from scratch, and also prevent overfitting."

Dracula was skeptical at first but decided to give it a try. PyTorch Powerhouse immediately got to work and showed Dracula how to load and modify the pre-trained model using PyTorch.

After a few hours, they had a new model that could accurately classify different types of monsters. Dracula was impressed by the technique and thanked PyTorch Powerhouse for his help.

"Transfer learning has opened up new possibilities for me," said Dracula. "I can now use pre-trained models to build custom models for various classification problems. Thank you, PyTorch Powerhouse, for showing me the way."

The travelers bid farewell to Dracula and PyTorch Powerhouse, as they rode off into the stormy night. Dracula knew that he had gained a valuable tool, thanks to PyTorch Powerhouse, and he could now classify monsters accurately.
# Code Explanation for the Dracula Story: Transfer Learning with PyTorch

In the Dracula Story, PyTorch Powerhouse used transfer learning to solve Dracula's monster classification problem. Transfer learning is a widely used technique in deep learning, and PyTorch provides several pre-trained models that we can load and modify for our specific problem.

Here is an overview of the code used to resolve Dracula's problem:

## Loading the Pre-trained Model
```
import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
```
In the code above, we first import the required libraries, including PyTorch and torchvision. We then load a pre-trained ResNet18 model using the `resnet18` method from torchvision.models. The `pretrained=True` parameter ensures that we download the pre-trained model from the internet.

## Modifying the Pre-trained Model
```
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
```
In this step, we modify the pre-trained model for Dracula's problem. We take the fully connected layer (`model.fc`) and replace it with a new custom layer that has 2 output neurons (to classify between monsters and humans). We also obtain the input features from the previous layer using `num_ftrs`.

## Training the Model
```
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    for inputs, labels in dataloaders['train']:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
After modifying the pre-trained model, we train it using the standard PyTorch training loop. We define a loss function (`torch.nn.CrossEntropyLoss()`) and an optimizer (`torch.optim.SGD`). We then loop over the data using the `dataloaders` and compute the loss and gradients for each batch. 

## Evaluating the Model
```
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_predictions) * 100
print(f"Model accuracy is {accuracy}%")
```
Finally, we evaluate the model using the test data. We compute the accuracy of the model by comparing the predicted labels with the true labels. The `torch.no_grad()` statement ensures that we don't compute gradients during inference, and the `torch.max()` function returns the index of the maximum value in the output tensor.

Using transfer learning with PyTorch, Dracula was able to classify monsters accurately and enhance his monster hunting skills.


[Next Chapter](11_Chapter11.md)