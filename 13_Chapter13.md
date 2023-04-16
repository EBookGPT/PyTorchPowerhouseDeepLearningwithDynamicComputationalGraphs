# Chapter 13: Saving and Loading PyTorch Models

Welcome back to PyTorch Powerhouse: Deep Learning with Dynamic Computational Graphs! In the previous chapter, we learned about deploying PyTorch models to production. Now, we will discuss how to save and load these models with ease.

Joining us for this chapter is the one and only Jeremy Howard. Jeremy is a renowned data scientist and a co-founder of fast.ai. He is also responsible for launching the Deep Learning course at the University of San Francisco. 

In this chapter, we'll explore how to save PyTorch models to various file formats, including pickles and protobufs. We'll also investigate the various methods to restore these models for use in later applications. Jeremy will take us through the benefits and drawbacks of each technique and provide insights on best practices.

As we know, training deep learning models is resource-heavy and time-consuming. Saving and loading trained models can help us avoid the need for retraining from scratch each time we re-run the model. By following the guidelines in this chapter, you'll learn how to optimize your workflow by leveraging the powerful capabilities offered by PyTorch.

So, let's dive right into Chapter 13 of the PyTorch Powerhouse: Deep Learning with Dynamic Computational Graphs!
# Chapter 13: Saving and Loading PyTorch Models - Dracula's Dilemma

Dracula had just achieved great success in training his deep learning model. His model was capable of identifying potential threats to his rule in the towns around his castle. But as the model grew more complex, it was becoming too difficult and time-consuming to retrain it each time there was new data to consider. It was draining his resources and tests showed retraining caused a significant increase in errors.

Dracula knew there had to be a better way. So, he contacted the PyTorch Powerhouse team, and they assigned him a top expert in the field - Jeremy Howard - to help him find a solution.

Together, Dracula and Jeremy discussed the various techniques for saving and loading PyTorch models. Jeremy first explained how to save models as pickles, a file format that Python users are familiar with. With this, Dracula was able to save even the most complex PyTorch models with minimal fuss.

However, Jeremy was quick to point out that pickling PyTorch models can have performance drawbacks, and recommended an alternate way - saving as protobuf. It was a more complicated technique to wrap his mind around, but with Jeremy's guidance, Dracula learned that saving in the protobuf format was much more efficient for his workflow as it allowed models to be loaded much more quickly than pickles. 

Jeremy also taught Dracula how to save multiple models and their parameters in a single file. This would be useful when the models had dependencies on one another.

Finally, Dracula learned a neat trick for saving models only once during his algorithm's training life cycle. After training his model, he could save the state dictionaries and load them during inference.

With Jeremy's guidance, Dracula was able to gain a deep understanding of saving and loading PyTorch models. He could now save precious time and avoid unnecessary retraining of models. His deep learning model worked more efficiently than ever before, and he continued to use PyTorch for all his future models.

Remember the story of Dracula's dilemma whenever you're faced with the challenge of managing a large number of deep learning models. The PyTorch Powerhouse: Deep Learning with Dynamic Computational Graphs and the insights shared by Jeremy Howard will guide you through the process with ease.
In this chapter, Dracula was faced with the challenge of efficiently retraining his deep learning models. The PyTorch Powerhouse team assigned him a top expert in the field, Jeremy Howard, to help him find a solution. Together, they explored techniques for saving and loading PyTorch models.

Here, we'll explore the code used to save and load PyTorch models. Suppose Dracula had trained his PyTorch model using the following code:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class Dracula model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Setup model and optimizer 
model = Dracula_model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(num_epochs):
    # Perform training steps here 

# Save model using pickle
torch.save(model.state_dict(), 'dracula_model.pickle')

# Save model using protobuf
torch.save(model.state_dict(), 'dracula_model.pb')
```

Here, we define the deep learning model, Dracula_model, and train it using the optim.Adam optimizer. Once the model is fully trained, we save it as a pickle file using `torch.save(model.state_dict(), 'dracula_model.pickle')`. Alternatively, we can also save the model using the protobuf format with `torch.save(model.state_dict(), 'dracula_model.pb')`.

To load a saved model, we use the following code:

```python
# Load the pickle file
loaded_model = Dracula_model()
loaded_model.load_state_dict(torch.load('dracula_model.pickle'))
loaded_model.eval()

# Load the protobuf file
loaded_model = Dracula_model()
loaded_model.load_state_dict(torch.load('dracula_model.pb'))
loaded_model.eval()
```

Here, we first define a new model instance and load the saved state dictionary from the pickle file with `loaded_model.load_state_dict(torch.load('dracula_model.pickle'))`. We can also load the state dictionary from the protobuf file with `loaded_model.load_state_dict(torch.load('dracula_model.pb'))`. After loading the model, we can continue to use it for inference by setting `loaded_model.eval()`

By saving and loading PyTorch models, Dracula can now save precious time and avoid unnecessary retraining of models. With the insights shared by Jeremy Howard, Dracula was able to optimize his workflow and achieve an efficient deep learning algorithm.


[Next Chapter](14_Chapter14.md)