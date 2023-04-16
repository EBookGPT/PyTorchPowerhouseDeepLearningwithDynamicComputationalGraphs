# Chapter 2: Tensor Basics in PyTorch

As we delve deeper into the world of PyTorch and its dynamic computational graphs, we must first explore the fundamental building blocks of the framework: tensors. Tensors, as the name suggests, are mathematical entities that generalize vectors and matrices to multiple dimensions. But what exactly are tensors, and why are they so crucial in deep learning?

In this chapter, we will answer these questions and more. We will learn about the basics of PyTorch tensors, including how to create them, manipulate them, and perform mathematical operations on them. We will also examine some of the key features of PyTorch tensors, such as their ability to work with both CPUs and GPUs, as well as their seamless integration with autograd, PyTorch's automatic differentiation engine.

But before we can begin our exploration of tensors, we must first introduce some key concepts and terminology. In the previous chapter, we talked about computational graphs and how they form the backbone of PyTorch's dynamic computation. In this chapter, we will need to extend our understanding of computational graphs to accomodate this new tensor concept.

So, don your capes and sharpen your fangs, for we are about to embark on a journey into the heart of PyTorch, guided by none other than Dracula himself! But fear not, dear readers, for while the subject matter may be complex and challenging, the Count is a most capable instructor. Together, we shall master the intricacies of PyTorch's tensor framework and once again emerge victorious over the forces of darkness and confusion.
## The Tale of Count PyTorch and the Tensors of Transylvania

Once upon a time, in a dark and mysterious land known as Transylvania, there lived a powerful and feared count named PyTorch. Count PyTorch was known throughout the land for his mastery of the dark arts of deep learning, and his ability to conjure up wondrous algorithms that could predict the future with uncanny accuracy.

One day, a young traveller named Jonathan stumbled upon the Count's castle, seeking refuge from the cold and the darkness that blanketed the land. The Count took pity on poor Jonathan and welcomed him into his lair, offering him warm blankets and hot tea.

As they sipped their tea, the Count noticed that Jonathan looked sad and troubled. "What troubles you, my young friend?" asked the Count.

"I am but a lowly programmer," replied Jonathan. "I know nothing of the dark arts, and the algorithms I create are feeble and powerless. I wish to learn from you, great Count, so that I too may wield the power of deep learning."

The Count smiled. "Ah, you wish to learn the secrets of PyTorch, do you? Then come, let us begin with the most basic building block of our framework: the tensor."

And with that, the Count led Jonathan to his laboratory, where he conjured up a glowing blue orb that floated in mid-air. "Behold, young Jonathan, the tensor!" exclaimed the Count.

Jonathan was confused. "But Count PyTorch, what is this... tensor? I see nothing but a glowing orb!"

The Count chuckled. "The tensor is a mathematical entity that can be represented by an array of numbers. It is the most basic building block of deep learning – everything else is built on top of it."

And so the Count began to teach Jonathan the ways of the tensor. He showed him how to create them, how to manipulate them, and how to perform mathematical operations on them. He demonstrated their ability to work with both CPUs and GPUs, and he explained their seamless integration with PyTorch's automatic differentiation engine, autograd.

Jonathan was amazed. He had never seen anything like this before! Under the guidance of Count PyTorch, he learned to create tensors of all shapes and sizes, to slice and dice them with ease, and to transform them using a myriad of mathematical functions.

And so it went, day after day, week after week, until Jonathan became a master of the tensor. He could create them with ease, manipulate them with finesse, and wield them like a sword against his foes. With the power of the tensor at his fingertips, he was unstoppable!

But as much as he had learned, Jonathan knew that there was still much more to discover. With a grateful heart, he bade farewell to Count PyTorch and set out on his own, eager to explore the mysteries of deep learning and the power of the dynamic computational graph.

And so the tale of Count PyTorch and the tensors of Transylvania comes to a close, but the adventure continues on. For as we have learned, the tensor is but the first step on a long and winding road. The path ahead is fraught with peril and uncertainty, but with the Count as our guide, we shall emerge victorious in the end.
# Code Explanation

The Dracula story in this chapter was meant to convey the basics of PyTorch tensors, their creation and manipulation, and their mathematical operations. While the story itself is fictional, the concepts it conveys are very real, and we can see their practical usage in PyTorch code.

## Creating Tensors

To create a PyTorch tensor, we use the `torch.tensor()` function. We can pass it a list of values, and it will create a tensor with the specified shape:

```
import torch

my_tensor = torch.tensor([1, 2, 3, 4])
print(my_tensor)
```

This will output `[1, 2, 3, 4]`, which is a 1-dimensional tensor of shape `(4,)`.

We can also create tensors of different shapes by passing in nested lists:

```
my_tensor = torch.tensor([[1, 2], [3, 4]])
print(my_tensor)
```

This will output `[[1, 2], [3, 4]]`, which is a 2-dimensional tensor of shape `(2, 2)`.

## Manipulating Tensors

We can manipulate tensors in several ways, such as by indexing, slicing, and concatenation. For example, to access a specific element in a tensor, we can use indexing:

```
my_tensor = torch.tensor([1, 2, 3, 4])
print(my_tensor[0])  # outputs 1
```

We can also slice tensors to extract a subset of the data:

```
my_tensor = torch.tensor([1, 2, 3, 4])
subset = my_tensor[1:3]
print(subset)  # outputs [2, 3]
```

To concatenate two tensors, we can use the `torch.cat()` function:

```
tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([3, 4])
concatenated = torch.cat((tensor1, tensor2))
print(concatenated)  # outputs [1, 2, 3, 4]
```

## Mathematical Operations

Finally, PyTorch tensors allow for all kinds of mathematical operations, ranging from the basic element-wise addition and multiplication to more complex mathematical functions. For example, to perform element-wise addition or multiplication, we can use the `+` and `*` operators:

```
tensor1 = torch.tensor([1, 2, 3, 4])
tensor2 = torch.tensor([2, 4, 6, 8])
summed = tensor1 + tensor2  # [3, 6, 9, 12]
product = tensor1 * tensor2  # [2, 8, 18, 32]
```

PyTorch provides a wide array of mathematical functions, such as `torch.matmul()` for matrix multiplication, `torch.sin()` for trigonometric functions, and many others.

That concludes our brief overview of tensor creation, manipulation, and mathematical operations in PyTorch. This is just the tip of the iceberg, however. Tensors are a powerful and versatile tool in the deep learning toolbox, and there is much more to explore and discover.


[Next Chapter](03_Chapter03.md)